#!/usr/bin/env python3
"""Generate optimized molecules using vLLM (optionally with LoRA)."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer molecule edits with vLLM.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--input-smiles", required=True, help="TXT file with one SMILES per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruction", default="make this molecule more like a drug")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_prompt(instruction: str, smiles: str) -> List[Dict[str, str]]:
    system = (
        "You are a molecular SMILES editing assistant. Use only INSERT/DELETE/REPLACE in <think>."
    )
    user = (
        f"Can you {instruction}? The output molecule should be similar to the input molecule.\n\n"
        f"input_smiles:\n{smiles}\n\n"
        "Please return:\n- edit_traj (using only INSERT/DELETE/REPLACE)\n- output_smiles"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_smiles(path: str, max_samples: int) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            items.append(s)
            if max_samples > 0 and len(items) >= max_samples:
                break
    return items


def main() -> None:
    args = parse_args()
    smiles = read_smiles(args.input_smiles, args.max_samples)

    if args.dry_run:
        print("Dry-run enabled. Configuration:")
        payload = vars(args).copy()
        payload["loaded_samples"] = len(smiles)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs: Dict[str, object] = {
        "model": args.model,
        "trust_remote_code": True,
        "max_model_len": 4096,
        "seed": args.seed,
    }
    if args.adapter:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1

    llm = LLM(**llm_kwargs)
    lora_request: Optional[LoRARequest] = None
    if args.adapter:
        lora_request = LoRARequest("molecule_adapter", 1, args.adapter)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
    )

    outputs = []
    for smi in smiles:
        prompt = tokenizer.apply_chat_template(
            build_prompt(args.instruction, smi),
            tokenize=False,
            add_generation_prompt=True,
        )
        result = llm.generate([prompt], sampling, use_tqdm=False, lora_request=lora_request)
        text = result[0].outputs[0].text.strip()
        outputs.append({"input_smiles": smi, "raw_output": text})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(outputs, fout, ensure_ascii=True, indent=2)

    print(f"Wrote {len(outputs)} records to {out_path}")


if __name__ == "__main__":
    main()
