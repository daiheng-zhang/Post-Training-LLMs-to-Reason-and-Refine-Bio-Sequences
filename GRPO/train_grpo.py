import os
import shlex
import subprocess
from typing import List

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


def _bool_flag(value: bool) -> str:
    return "true" if value else "false"


def _as_flag_value(value) -> str:
    if isinstance(value, bool):
        return _bool_flag(value)
    return str(value)


def _format_optional_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, ListConfig)):
        parts = [str(item) for item in value if item is not None]
        return ",".join(parts)
    return str(value)


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        return [value]
    return [str(value)]


def _ensure_dict_config(config):
    if isinstance(config, DictConfig):
        return config
    if isinstance(config, dict):
        return OmegaConf.create(config)
    return config


def _normalize_logger(logger) -> List[str]:
    if not logger:
        return []
    if isinstance(logger, (list, tuple)):
        return [str(item).lower() for item in logger]
    return [str(logger).lower()]


def _wandb_env_enabled() -> bool:
    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if mode in {"disabled", "offline"}:
        return False
    for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_API_KEY", "WANDB_RUN_ID", "WANDB_NAME"):
        if os.environ.get(key):
            return True
    return False


def _resolve_path(base_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def _resolve_list_paths(base_dir: str, paths) -> List[str]:
    if not paths:
        return paths
    return [_resolve_path(base_dir, path) for path in paths]


def _get_world_size() -> int:
    for key in ("WORLD_SIZE", "NPROC_PER_NODE", "LOCAL_WORLD_SIZE"):
        value = os.environ.get(key)
        if value:
            try:
                size = int(value)
            except ValueError:
                continue
            if size > 0:
                return size
    return 1


def _in_distributed_env() -> bool:
    for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
        if os.environ.get(key) is not None:
            return True
    return False


def _detect_nproc_per_node() -> int:
    for key in ("NPROC_PER_NODE", "LOCAL_WORLD_SIZE"):
        value = os.environ.get(key)
        if value:
            try:
                count = int(value)
            except ValueError:
                continue
            if count > 0:
                return count
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass
    return 1


def _build_launcher_cmd() -> List[str]:
    if _in_distributed_env():
        return ["swift", "rlhf"]
    nproc = _detect_nproc_per_node()
    if nproc > 1:
        cmd = ["torchrun", f"--nproc_per_node={nproc}"]
        master_port = os.environ.get("MASTER_PORT")
        if master_port:
            cmd.extend(["--master_port", master_port])
        cmd.extend(["-m", "swift.cli.rlhf"])
        return cmd
    return ["swift", "rlhf"]


def _warn_generation_divisibility(
    num_generations: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    per_device_eval_batch_size: int = 0,
) -> None:
    world_size = _get_world_size()
    effective_train = world_size * per_device_train_batch_size * gradient_accumulation_steps
    if effective_train > 0 and effective_train % num_generations != 0:
        print(
            "Warning: num_generations does not divide the effective train batch "
            f"(num_generations={num_generations}, effective_train={effective_train}). "
            "Adjust num_generations or batch/accumulation to avoid GRPO sampling errors."
        )
    if per_device_eval_batch_size:
        effective_eval = world_size * per_device_eval_batch_size
        if effective_eval > 0 and effective_eval % num_generations != 0:
            print(
                "Warning: num_generations does not divide the effective eval batch "
                f"(num_generations={num_generations}, effective_eval={effective_eval}). "
                "Adjust num_generations or eval batch size to avoid GRPO eval errors."
            )


def resolve_data_paths(config: DictConfig) -> DictConfig:
    config = _ensure_dict_config(config)
    # Resolve relative paths from repository root.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if getattr(config, "data", None):
        config.data.train_files = _resolve_list_paths(base_dir, config.data.train_files)
        config.data.val_files = _resolve_list_paths(base_dir, config.data.val_files)
    if getattr(config, "trainer", None):
        if getattr(config.trainer, "default_local_dir", None):
            config.trainer.default_local_dir = _resolve_path(base_dir, config.trainer.default_local_dir)
        if getattr(config.trainer, "resume_from_path", None):
            config.trainer.resume_from_path = _resolve_path(base_dir, config.trainer.resume_from_path)
    if "swift" in config and getattr(config.swift, "external_plugin", None):
        config.swift.external_plugin = _resolve_path(base_dir, config.swift.external_plugin)
    return config


def _get_first(paths):
    if isinstance(paths, (list, tuple, ListConfig)) and paths:
        return paths[0]
    if isinstance(paths, str):
        return paths
    return None


def _get_data_field(config, field: str):
    if config is None:
        return None
    data = None
    if hasattr(config, "data"):
        data = getattr(config, "data", None)
    elif isinstance(config, dict):
        data = config.get("data")
    if data is None:
        return None
    if isinstance(data, (dict, DictConfig)):
        return data.get(field)
    if hasattr(data, field):
        return getattr(data, field)
    return None


def _get_actor_rollout_model_path(config):
    if config is None:
        return None
    actor_rollout_ref = None
    if hasattr(config, "actor_rollout_ref"):
        actor_rollout_ref = getattr(config, "actor_rollout_ref", None)
    elif isinstance(config, dict):
        actor_rollout_ref = config.get("actor_rollout_ref")
    if actor_rollout_ref is None:
        return None
    if isinstance(actor_rollout_ref, (dict, DictConfig)):
        model = actor_rollout_ref.get("model")
    else:
        model = getattr(actor_rollout_ref, "model", None)
    if model is None:
        return None
    if isinstance(model, (dict, DictConfig)):
        return model.get("path")
    return getattr(model, "path", None)


def build_swift_command(config: DictConfig) -> List[str]:
    config = _ensure_dict_config(config)
    train_files = _get_data_field(config, "train_files")
    train_path = _get_first(train_files)
    if not train_path:
        print("==== DEBUG build_swift_command ====")
        print("config.data =", getattr(config, "data", None))
        print("config.data.train_files =", getattr(config.data, "train_files", None))
        print("config.swift =", getattr(config, "swift", None))
        print("===================================")
        raise ValueError("Missing data.train_files[0] for Swift GRPO dataset.")
    val_files = _get_data_field(config, "val_files")
    val_path = _get_first(val_files)

    if "swift" not in config:
        raise ValueError("Missing swift config block for ms-swift GRPO launch.")
    swift_cfg = config.swift

    num_generations = getattr(swift_cfg, "num_generations", None)
    if num_generations is None:
        num_generations = config.actor_rollout_ref.rollout.n

    _warn_generation_divisibility(
        num_generations=int(num_generations),
        per_device_train_batch_size=int(swift_cfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(swift_cfg.gradient_accumulation_steps),
        per_device_eval_batch_size=int(getattr(swift_cfg, "per_device_eval_batch_size", 0) or 0),
    )

    resume_path = None
    if getattr(config.trainer, "resume_mode", None) == "resume_path":
        resume_path = getattr(config.trainer, "resume_from_path", None)

    model_path = config.model.path
    if resume_path:
        base_model_path = getattr(config.model, "base_path", None) or _get_actor_rollout_model_path(config)
        if base_model_path:
            model_path = base_model_path

    attn_impl = getattr(swift_cfg, "attn_impl", None)
    if not attn_impl and config.model.use_remove_padding:
        attn_impl = "flash_attn"
    include_padding_free = attn_impl != "eager"
    temperature = getattr(swift_cfg, "temperature", None)

    max_steps = getattr(config.trainer, "max_steps", None)
    max_steps_value = None
    if max_steps is not None:
        try:
            max_steps_value = int(max_steps)
        except (TypeError, ValueError):
            max_steps_value = None

    cmd = _build_launcher_cmd()
    cmd.extend([
        "--rlhf_type",
        "grpo",
        "--do_train",
        "--model",
        model_path,
        "--model_type",
        "qwen3",
        "--train_type",
        str(getattr(swift_cfg, "train_type", "lora")),
    ])

    importance_sampling_level = getattr(config, "importance_sampling_level", None)
    if importance_sampling_level is None and getattr(config, "algorithm", None) is not None:
        importance_sampling_level = getattr(getattr(config.algorithm, "grpo", None), "importance_sampling_level", None)
    if importance_sampling_level is None and getattr(config, "swift", None) is not None:
        importance_sampling_level = getattr(config.swift, "importance_sampling_level", None)
    if importance_sampling_level is not None:
        value = str(importance_sampling_level).strip()
        if value:
            cmd.extend(["--importance_sampling_level", value])

    lora_rank = getattr(swift_cfg, "lora_rank", None)
    lora_alpha = getattr(swift_cfg, "lora_alpha", None)
    target_modules = getattr(swift_cfg, "target_modules", None)
    if lora_rank is None:
        lora_rank = getattr(config.model, "lora_rank", None)
    if lora_alpha is None:
        lora_alpha = getattr(config.model, "lora_alpha", None)
    if target_modules is None:
        target_modules = getattr(config.model, "target_modules", None)
    target_modules_list = _as_list(target_modules)
    if lora_rank is None or lora_alpha is None or not target_modules_list:
        raise ValueError(
            "Missing LoRA config. Set `swift.lora_rank`, `swift.lora_alpha`, and `swift.target_modules` "
            "(preferred), or keep the legacy `model.lora_*` fields."
        )

    adapters = _format_optional_list(getattr(swift_cfg, "adapters", None))
    if adapters:
        cmd.extend(["--adapters", adapters])
    ref_adapters = _format_optional_list(getattr(swift_cfg, "ref_adapters", None))
    if ref_adapters:
        cmd.extend(["--ref_adapters", ref_adapters])

    cmd.extend([
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        "--target_modules",
        *target_modules_list,
        "--dataset",
        train_path,
        "--split_dataset_ratio",
        str(getattr(swift_cfg, "split_dataset_ratio", 0)),
        "--external_plugins",
        str(swift_cfg.external_plugin),
        "--reward_funcs",
        str(swift_cfg.reward_func),
        "--beta",
        str(config.algorithm.kl_ctrl.kl_coef),
        "--epsilon",
        str(config.algorithm.grpo.epsilon),
        "--num_generations",
        str(num_generations),
        "--max_length",
        str(config.data.max_prompt_length),
        "--max_completion_length",
        str(config.data.max_response_length),
        "--learning_rate",
        str(config.trainer.learning_rate),
        "--gradient_checkpointing",
        _bool_flag(config.model.enable_gradient_checkpointing),
        "--per_device_train_batch_size",
        str(swift_cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(swift_cfg.gradient_accumulation_steps),
        "--output_dir",
        str(config.trainer.default_local_dir),
    ])
    loss_type = getattr(getattr(config.algorithm, "grpo", None), "loss_type", None)
    if loss_type is None:
        loss_type = getattr(config.algorithm, "loss_type", None)
    if loss_type is None:
        loss_type = getattr(swift_cfg, "loss_type", None)
    if loss_type is not None:
        value = str(loss_type).strip()
        if value:
            cmd.extend(["--loss_type", value])
    epsilon_high = getattr(config.algorithm.grpo, "epsilon_high", None)
    if epsilon_high is not None:
        cmd.extend(["--epsilon_high", str(epsilon_high)])
    steps_per_generation = getattr(config.algorithm.grpo, "steps_per_generation", None)
    if steps_per_generation is not None:
        cmd.extend(["--steps_per_generation", str(steps_per_generation)])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])

    load_args = getattr(config.trainer, "load_args", None)
    if load_args is not None:
        cmd.extend(["--load_args", _as_flag_value(load_args)])

    if max_steps_value and max_steps_value > 0:
        cmd.extend(["--max_steps", str(max_steps_value)])
    else:
        cmd.extend(["--num_train_epochs", str(config.trainer.total_epochs)])

    if include_padding_free:
        cmd.extend(["--padding_free", _bool_flag(config.model.use_remove_padding)])
    if attn_impl:
        cmd.extend(["--attn_impl", str(attn_impl)])

    if val_path:
        cmd.extend(
            [
                "--do_eval",
                "--val_dataset",
                val_path,
                "--per_device_eval_batch_size",
                str(swift_cfg.per_device_eval_batch_size),
            ]
        )

    if getattr(config.trainer, "save_freq", 0):
        cmd.extend(
            [
                "--save_strategy",
                "steps",
                "--save_steps",
                str(config.trainer.save_freq),
            ]
        )
        if getattr(config.trainer, "save_total_limit", None):
            cmd.extend(["--save_total_limit", str(config.trainer.save_total_limit)])

    save_only_model = getattr(config.trainer, "save_only_model", None)
    if save_only_model is not None:
        cmd.extend(["--save_only_model", _as_flag_value(save_only_model)])

    if val_path and getattr(config.trainer, "test_freq", 0):
        cmd.extend(
            [
                "--eval_strategy",
                "steps",
                "--eval_steps",
                str(config.trainer.test_freq),
            ]
        )

    if getattr(swift_cfg, "logging_steps", 0):
        cmd.extend(["--logging_steps", str(swift_cfg.logging_steps)])

    log_completions = getattr(swift_cfg, "log_completions", None)
    if log_completions is not None:
        cmd.extend(["--log_completions", _as_flag_value(log_completions)])

    report_to = "wandb"
    if report_to:
        cmd.extend(["--report_to", report_to])

    if resume_path:
        cmd.extend(["--resume_from_checkpoint", str(resume_path)])

    return cmd


@hydra.main(config_path="../configs/rl", config_name="chem_grpo", version_base=None)
def main(config: DictConfig) -> None:
    config = resolve_data_paths(config)
    print("Training Config:\n", OmegaConf.to_yaml(config))
    cmd = build_swift_command(config)
    print("Swift command:\n", " ".join(shlex.quote(arg) for arg in cmd))
    if os.environ.get("STRIDE_DRY_RUN", "").strip() in {"1", "true", "True"}:
        print("STRIDE_DRY_RUN is enabled. Command execution skipped.")
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
