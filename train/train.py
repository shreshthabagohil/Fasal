"""train/train.py -- QLoRA SFT for Fasal (Kisan Call Centre farmer-advisory model).

Written 2026-08-18. There was previously no committed training script: the
three Kaggle/Colab sessions that produced the shipped adapter
(Algo-Nova/fasal-sarvam1-lora, HF commit 3d1fa91) ran from an untracked local
script that used trl's SFTTrainer, which broke twice across trl versions
during the event (see requirements.txt's dropped-trl comment). This script
replaces that with plain `transformers.Trainer` -- no trl dependency -- so a
retrain is reproducible from a committed file, not a one-off notebook cell.

Every hyperparameter below is READ from train/config.yaml, not hardcoded here
-- config.yaml is the single source of truth (per train/A0_LOCKED.md: "Changes
require a Planning-session decision + an entry in 08_MILESTONE_LOG.md. Do NOT
silently edit."). If you need to change a hyperparameter, change config.yaml,
not this file.

Prompt template is byte-for-byte identical to eval/infer.py's PROMPT_TEMPLATE
and demo/app.py's -- all three MUST stay in sync (see demo/app.py's module
docstring for why this matters: sarvam-1 is a text-completion base model,
not a chat model, so this Alpaca-style format is what the adapter actually
learns, not a chat template).

Usage (single seed):
    python train/train.py \\
        --dataset dataset/kcc_instruct_v1.jsonl \\
        --config train/config.yaml \\
        --output-dir train/out/iter-0-sarvam-seed1729 \\
        --seed 1729

Usage (the 3-seed final candidate run required by A0_LOCKED.md before
shipping -- "retrain on 3 seeds {1729, 2027, 3141}; median-seed ships"):
    for seed in 1729 2027 3141; do
        python train/train.py --seed "$seed" \\
            --output-dir "train/out/final-seed${seed}"
    done
    # then run eval/run_eval.sh against each of the 3 adapters and ship
    # whichever has the median win-rate, per A0_LOCKED.md's claim bar.

Requires a CUDA GPU (bitsandbytes 4-bit NF4 quantization is CUDA-only --
does not run on Apple Silicon / MPS. This is why Kaggle/Colab/Azure were
used for training while eval/infer.py has a separate MPS fallback path for
GPU-free inference on a Mac).
"""
import argparse
import json
import os
import random
import sys

# Must stay byte-for-byte identical to eval/infer.py's PROMPT_TEMPLATE and
# demo/app.py's PROMPT_TEMPLATE. If this changes, update both of those too.
PROMPT_TEMPLATE = "### Question:\n{instruction}\n### Context:\n{input}\n### Answer:"


def set_seed(seed: int) -> None:
    # Same seeding routine as eval/infer.py -- keep in sync.
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
    ap = argparse.ArgumentParser(description="QLoRA SFT training for Fasal.")
    ap.add_argument("--dataset", default="dataset/kcc_instruct_v1.jsonl",
                     help="Training JSONL (instruction/input/output/lang schema). "
                          "Held-out/gold/reserve rows are already carved OUT of this "
                          "file by data/build_gold_and_heldout.py, so no extra "
                          "leakage filtering is needed here.")
    ap.add_argument("--config", default="train/config.yaml",
                     help="Single source of truth for all hyperparameters.")
    ap.add_argument("--output-dir", required=True,
                     help="Where to save the LoRA adapter + tokenizer.")
    ap.add_argument("--seed", type=int, default=None,
                     help="Overrides config.yaml's seed (for the 3-seed final run). "
                          "Defaults to config.yaml's seed if not given.")
    ap.add_argument("--wandb-run-name", default=None,
                     help="Defaults to '<config wandb.run_name_prefix>-<seed>'.")
    ap.add_argument("--no-wandb", action="store_true",
                     help="Disable W&B logging (e.g. for a quick local smoke test).")
    return ap.parse_args()


def load_config(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_example(tokenizer, row, max_seq_length):
    """Build one training example with prompt tokens masked out of the loss
    (labels = -100 on the prompt span, real token ids on the completion span
    only). Training on prompt tokens is a common QLoRA/SFT mistake -- it
    wastes capacity teaching the model to predict farmer questions it's
    never asked to generate, and dilutes gradient signal on the actual
    answer, the part quality is judged on."""
    prompt = PROMPT_TEMPLATE.format(
        instruction=row.get("instruction", ""), input=row.get("input", "")
    )
    completion = " " + (row.get("output", "") or "").strip() + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + completion_ids)[:max_seq_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class PadCollator:
    """Left-pads input_ids/attention_mask/labels to the longest example in
    the batch. Labels are padded with -100 (ignored by the loss), not the
    pad token id -- padding a text label with a real token id would silently
    train the model to predict pad tokens as content, which is wrong."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append([self.pad_token_id] * pad_len + f["input_ids"])
            attention_mask.append([0] * pad_len + f["attention_mask"])
            labels.append([-100] * pad_len + f["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed as hf_set_seed,
    )

    hf_set_seed(seed)

    if not torch.cuda.is_available():
        print(
            "[train] ERROR: no CUDA GPU detected. QLoRA 4-bit (bitsandbytes NF4) "
            "requires CUDA and does not run on Apple Silicon / MPS or CPU. Run this "
            "on Kaggle/Colab/Azure, not locally on a Mac. (eval/infer.py has a "
            "separate MPS-friendly path for GPU-free *inference*, not training.)",
            file=sys.stderr,
        )
        sys.exit(1)

    base_model = cfg["chosen"]["base_model"]
    base_sha = cfg["chosen"]["base_commit_sha"]
    qlora_cfg = cfg["qlora"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    wandb_cfg = cfg.get("wandb", {})

    print(f"[train] base={base_model}@{base_sha} seed={seed} output_dir={args.output_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=base_sha)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if qlora_cfg["compute_dtype"] == "bf16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["bits"] == 4,
        bnb_4bit_quant_type=qlora_cfg["quant"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg["double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_sha,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"[train] loading dataset from {args.dataset} ...", flush=True)
    rows = load_dataset_rows(args.dataset)
    print(f"[train] {len(rows)} training rows", flush=True)

    max_seq_length = train_cfg["max_seq_length"]
    examples = [build_example(tokenizer, r, max_seq_length) for r in rows]
    hf_dataset = Dataset.from_list(examples)

    run_name = args.wandb_run_name or f"{wandb_cfg.get('run_name_prefix', 'iter')}-seed{seed}"
    report_to = [] if args.no_wandb else ["wandb"]
    if not args.no_wandb:
        os.environ.setdefault("WANDB_ENTITY", wandb_cfg.get("entity", ""))
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "fasal"))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        optim=train_cfg["optim"],
        bf16=(qlora_cfg["compute_dtype"] == "bf16"),
        fp16=(qlora_cfg["compute_dtype"] != "bf16"),
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=report_to,
        run_name=run_name,
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    print("[train] starting training ...", flush=True)
    trainer.train()

    print(f"[train] saving adapter to {args.output_dir} ...", flush=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[train] done.", flush=True)


if __name__ == "__main__":
    main()
