#!/usr/bin/env python3
"""Generate GFP variants using vLLM (optionally with a LoRA adapter)."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_WT_SEQUENCE = (
    "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCF"
    "SRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLE"
    "YNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNE"
    "KRDHMVLLEFVTAAGITHGMDELYK"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer GFP variants with vLLM.")
    parser.add_argument("--model", required=True, help="Base model path or HF id")
    parser.add_argument("--adapter", default="", help="Optional LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--total-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wt-seq", default=DEFAULT_WT_SEQUENCE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_prompt(wt_seq: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a protein engineer optimizing fluorescence for GFP variants."
    )
    user_prompt = (
        "Background:\n"
        "- WT is the baseline GFP sequence.\n"
        "- Propose edits that improve fluorescence.\n\n"
        "Output format requirements:\n"
        "1) Put mutation actions in <think>...</think>.\n"
        "2) After </think>, output only {\"protein\": \"<MUTATED_SEQUENCE>\"}.\n\n"
        f"WT sequence:\n{wt_seq}\n"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def parse_output(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "raw_output": text,
        "thinking_process": None,
        "parsed_protein": None,
        "valid_format": False,
    }
    if "</think>" not in text:
        return out
    think_part, rest = text.split("</think>", 1)
    think_part = think_part.replace("<think>", "").strip()
    rest = rest.strip()
    try:
        obj = json.loads(rest)
    except Exception:
        return out
    if isinstance(obj, dict) and isinstance(obj.get("protein"), str):
        out["thinking_process"] = think_part
        out["parsed_protein"] = obj["protein"]
        out["valid_format"] = True
    return out


def main() -> None:
    args = parse_args()
    if args.total_samples <= 0:
        raise ValueError("--total-samples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.dry_run:
        print("Dry-run enabled. Configuration:")
        print(json.dumps(vars(args), indent=2, ensure_ascii=True))
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    messages = build_prompt(args.wt_seq)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm_kwargs = {
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
        lora_request = LoRARequest("protein_adapter", 1, args.adapter)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.batch_size,
    )

    results: List[Dict[str, Any]] = []
    batches = (args.total_samples + args.batch_size - 1) // args.batch_size
    for _ in range(batches):
        outputs = llm.generate([prompt], sampling, use_tqdm=False, lora_request=lora_request)
        for out in outputs[0].outputs:
            results.append(parse_output(out.text.strip()))
            if len(results) >= args.total_samples:
                break
        if len(results) >= args.total_samples:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=True, indent=2)

    print(f"Wrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
