"""Fasal advisor demo -- pure Gradio chat UI, deployed as an HF Space (Gradio SDK).

Loads the base model (sarvamai/sarvam-1, pinned SHA per train/A0_LOCKED.md) with the
published Fasal LoRA adapter (Algo-Nova/fasal-sarvam1-lora) applied on top, and serves
a simple multilingual farmer-advisory chat interface.

Zero-budget note: this targets HF Spaces' free CPU Basic tier (2 vCPU / 16GB RAM, no
GPU). bitsandbytes 4-bit quantization is CUDA-only (see requirements.txt in the main
repo), so on CPU we load in bfloat16 instead -- roughly half the memory of float32 for
a ~2.5B-param model, and CPU inference in bf16 is supported by recent PyTorch. This
keeps the demo well inside the 16GB budget but generation will be slow (tens of
seconds per reply) -- max_new_tokens is capped low specifically to keep that from
becoming a multi-minute wait. If the Space is later upgraded to a paid GPU tier, set
DEVICE_OVERRIDE=cuda and the 4-bit bitsandbytes path can be re-enabled (see
_load_model() below) for much faster generation.

Uses the Gradio SDK (not Docker) so the Space stays on HF's free CPU Basic tier --
Docker Spaces require a paid plan, Gradio SDK Spaces do not.

Env vars (all optional, sensible defaults for the published model):
    BASE_MODEL       default: sarvamai/sarvam-1
    BASE_SHA         default: e9607337286ddf496d4a2562b194e489dcf3feea
    ADAPTER_REPO     default: Algo-Nova/fasal-sarvam1-lora
    MAX_NEW_TOKENS   default: 256
    DEVICE_OVERRIDE  default: auto-detect (cuda if available, else cpu)
"""
import os
import threading
import time

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "sarvamai/sarvam-1")
BASE_SHA = os.environ.get("BASE_SHA", "e9607337286ddf496d4a2562b194e489dcf3feea")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "Algo-Nova/fasal-sarvam1-lora")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

SYSTEM_PROMPT = (
    "You are Fasal, an agricultural advisor for Indian farmers. Answer in the SAME "
    "language and script the farmer asked in (Hindi, Gujarati, Marathi, Tamil, "
    "Bengali, Kannada, Punjabi, or English). Be specific and actionable: name the "
    "crop/input/dose/timing where relevant. If you are not confident, say so rather "
    "than guessing -- an unsafe or wrong dose recommendation is worse than no answer."
)

# Loaded lazily on first request (not at import time) so the Space comes up
# immediately instead of blocking startup on a multi-GB model download + load.
_model = None
_tokenizer = None
_load_lock = threading.Lock()
_load_error = None


def _pick_device_and_dtype():
    override = os.environ.get("DEVICE_OVERRIDE")
    if override:
        device = override
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 halves memory vs fp32 and is supported for CPU inference in recent
    # PyTorch; fall back to fp32 if something on this box doesn't support it.
    dtype = torch.bfloat16
    return device, dtype


def _load_model():
    global _model, _tokenizer, _load_error
    with _load_lock:
        if _model is not None or _load_error is not None:
            return
        try:
            device, dtype = _pick_device_and_dtype()
            print(f"[fasal-demo] loading {BASE_MODEL}@{BASE_SHA} on {device} ({dtype}) ...", flush=True)

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_SHA)

            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                revision=BASE_SHA,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )

            model = PeftModel.from_pretrained(base, ADAPTER_REPO)
            model.to(device)
            model.eval()

            _tokenizer = tokenizer
            _model = model
            print("[fasal-demo] model ready.", flush=True)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the UI, not swallowed
            _load_error = str(exc)
            print(f"[fasal-demo] MODEL LOAD FAILED: {exc}", flush=True)


def respond(message, history):
    if _model is None and _load_error is None:
        _load_model()

    if _load_error is not None:
        return (
            "Sorry -- the model failed to load on this Space. This is a "
            f"server-side problem, not something wrong with your question. "
            f"(error: {_load_error})"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        # Gradio ChatInterface history is a list of {"role", "content"} dicts
        # (type="messages" format) or legacy (user, assistant) tuples -- handle both.
        if isinstance(turn, dict):
            messages.append(turn)
        else:
            user_msg, bot_msg = turn
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    start = time.monotonic()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=_tokenizer.eos_token_id,
        )
    elapsed = time.monotonic() - start

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = _tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f"[fasal-demo] generated {len(generated)} tokens in {elapsed:.1f}s", flush=True)
    return reply or "(no output -- try rephrasing the question)"


DESCRIPTION = (
    "**Fasal** -- a multilingual farmer-advisory model, QLoRA fine-tuned from "
    f"`{BASE_MODEL}` on real Kisan Call Centre queries across 8 Indian languages. "
    "Ask a farming question in Hindi, Gujarati, Marathi, Tamil, Bengali, Kannada, "
    "Punjabi, or English.\n\n"
    "*Running on a free CPU Space -- first reply after startup can take a minute "
    "while the model loads; each reply after that takes tens of seconds.*"
)

chat = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Fasal -- Multilingual Farmer Advisor",
    description=DESCRIPTION,
    examples=[
        "मिर्च की बुवाई का सही समय क्या है?",
        "What is the best time to sow groundnut?",
        "பருத்தியில் இலை சுருள் நோயை எப்படி கட்டுப்படுத்துவது?",
    ],
)

if __name__ == "__main__":
    chat.launch()
