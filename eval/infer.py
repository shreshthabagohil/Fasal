"""eval/infer.py — batched generation from a base model, optionally + a PEFT LoRA adapter.

Two modes:
  (a) BASE only          — load base at pinned SHA, generate.
  (b) BASE + ADAPTER      — load base + PEFT LoRA adapter, generate.

Prompt template (byte-for-byte match to train/config.yaml — DO NOT deviate):
    ### Question:
    {instruction}
    ### Context:
    {input}
    ### Answer:

Usage:
    python eval/infer.py \\
        --base sarvamai/sarvam-1 --base-sha <sha> \\
        --heldout eval/heldout/test.jsonl --out eval/out/base.jsonl \\
        --batch-size 8 --max-new-tokens 512 --device auto
"""
import argparse
import json
import random
import sys

PROMPT_TEMPLATE = "### Question:\n{instruction}\n### Context:\n{input}\n### Answer:"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args():
    ap = argparse.ArgumentParser(description="Batched base(+adapter) generation for eval.")
    ap.add_argument("--base", required=True, help="HF repo id of the base model")
    ap.add_argument("--base-sha", required=True, help="Pinned HF commit SHA (never a floating tag)")
    ap.add_argument("--adapter", default=None, help="Path or HF repo id of a PEFT LoRA adapter (optional)")
    ap.add_argument("--heldout", required=True, help="Input JSONL path")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"], default="auto")
    ap.add_argument("--seed", type=int, default=1729)
    return ap.parse_args()


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(row) -> str:
    return PROMPT_TEMPLATE.format(instruction=row.get("instruction", ""), input=row.get("input", ""))


def postprocess(full_text: str, prompt: str) -> str:
    # Take only newly generated text after the prompt, then split on the
    # "### Answer:" marker and take the tail; truncate at the next "### " if any.
    tail = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    marker = "### Answer:"
    if marker in full_text:
        tail = full_text.split(marker, 1)[1]
    next_section = tail.find("### ")
    if next_section != -1:
        tail = tail[:next_section]
    return tail.strip()


def main():
    args = parse_args()
    set_seed(args.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    def detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():  # Apple Silicon (M1/M2/M3/M4) GPU
            return "mps"
        return "cpu"

    resolved_device = detect_device() if args.device == "auto" else args.device
    device_map = {"": resolved_device}

    rows = load_rows(args.heldout)
    print(f"[infer] loaded {len(rows)} rows from {args.heldout}", flush=True)
    print(f"[infer] resolved device: {resolved_device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.base_sha)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if resolved_device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            revision=args.base_sha,
            quantization_config=bnb_config,
            device_map=device_map,
        )
    elif resolved_device == "mps":
        # bitsandbytes 4-bit NF4 needs a CUDA backend, which Apple Silicon
        # does not have. Load directly on the Mac's GPU (MPS) in fp16 instead
        # of falling all the way back to CPU -- much faster on a MacBook
        # (M1/M2/M3/M4), just without 4-bit quantization.
        print(
            "[infer] no CUDA device — using Apple Silicon MPS GPU (fp16, no 4-bit quant)",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            revision=args.base_sha,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        # CPU fallback (smoke tests / tiny models): slowest path, only used
        # when neither a CUDA nor an MPS GPU is available.
        print(
            "[infer] WARNING: no CUDA/MPS device — loading without 4-bit NF4 on CPU (slow; smoke-test path only)",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            revision=args.base_sha,
            device_map=device_map,
        )

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    n_in = len(rows)
    n_out = 0
    n_err = 0

    from tqdm import tqdm

    with open(args.out, "w", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="infer"):
            batch = rows[start : start + args.batch_size]
            prompts = [build_prompt(r) for r in batch]
            try:
                enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        do_sample=False,
                        num_beams=1,
                        temperature=None,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for row, prompt, full_text in zip(batch, prompts, decoded):
                    answer = postprocess(full_text, prompt)
                    out_f.write(json.dumps({"id": row["id"], "prompt": prompt, "answer": answer}) + "\n")
                    n_out += 1
            except Exception as e:  # noqa: BLE001 - log per-row error and continue
                for row, prompt in zip(batch, prompts):
                    out_f.write(
                        json.dumps({"id": row["id"], "prompt": prompt, "answer": None, "error": str(e)}) + "\n"
                    )
                    n_err += 1
                print(f"[infer] ERROR on batch starting at {start}: {e}", file=sys.stderr)

    print(f"[infer] done. total_in={n_in} total_out={n_out} errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
