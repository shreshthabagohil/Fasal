"""Fasal advisor demo -- pure Gradio chat UI, deployed as an HF Space (Gradio SDK,
ZeroGPU hardware -- free tier, GPU allocated on demand per request).

Loads the base model (sarvamai/sarvam-1, pinned SHA per train/A0_LOCKED.md) with the
published Fasal LoRA adapter (Algo-Nova/fasal-sarvam1-lora) applied on top, and serves
a simple multilingual farmer-advisory chat interface.

PROMPT FORMAT -- read before touching this file (fixed 2026-08-17):
sarvam-1 is a text-completion BASE model ("cannot be used directly as a chat or an
instruction-following model" -- https://huggingface.co/sarvamai/sarvam-1). It ships a
generic chat_template, but our LoRA adapter was never trained on that template -- it
was trained on the plain Alpaca-style instruction format below (byte-for-byte match to
eval/infer.py and train/config.yaml's dataset). An earlier version of this file called
tokenizer.apply_chat_template(...) instead, which fed the adapter a prompt shape it had
never seen during training -- a silent, high-impact bug (garbage/generic-looking
answers on the live demo that had nothing to do with model or data quality). Do not
reintroduce apply_chat_template here unless the training format itself changes, and if
it does, update this comment + eval/infer.py + train/config.yaml together.

Training data (dataset/kcc_instruct_v1.jsonl) is single-turn, not multi-turn chat, and
every row's "input" field is a populated "State: X | Crop: Y | Season: Z | Type: W"
context string -- never blank. So this UI (a) only feeds the CURRENT message to the
model, ignoring prior chat turns for prompting (multi-turn KCC dialogue was never in
the training distribution, so concatenating history would just add more out-of-
distribution noise, not real context), and (b) exposes optional State/Crop/Season/
Query-type fields so a filled-in query matches the training input distribution instead
of leaving "input" blank (also out-of-distribution).

ZeroGPU note: this Space uses HF's free ZeroGPU hardware. ZeroGPU only attaches a
real GPU for the duration of a function decorated with @spaces.GPU -- outside that
function torch.cuda.is_available() is False. So model load + generation both happen
inside respond(), which is decorated below; that's the only place CUDA is visible.

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
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "sarvamai/sarvam-1")
BASE_SHA = os.environ.get("BASE_SHA", "e9607337286ddf496d4a2562b194e489dcf3feea")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "Algo-Nova/fasal-sarvam1-lora")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

# Must stay byte-for-byte identical to eval/infer.py's PROMPT_TEMPLATE and to the
# format the training data (dataset/kcc_instruct_v1.jsonl) was built from -- this is
# what the LoRA adapter actually learned, not a chat template.
PROMPT_TEMPLATE = "### Question:\n{instruction}\n### Context:\n{input}\n### Answer:"

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
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

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


def _build_context(state: str, crop: str, season: str, query_type: str) -> str:
    """Build the "input" field to match the training data's
    "State: X | Crop: Y | Season: Z | Type: W" shape. Blank fields are dropped
    rather than left as empty placeholders, since the training data never has
    a field present-but-empty -- it either has a real value or omits the field.
    """
    parts = []
    if state.strip():
        parts.append(f"State: {state.strip()}")
    if crop.strip():
        parts.append(f"Crop: {crop.strip()}")
    if season.strip():
        parts.append(f"Season: {season.strip()}")
    if query_type.strip():
        parts.append(f"Type: {query_type.strip()}")
    return " | ".join(parts)


def _postprocess(full_text: str, prompt: str) -> str:
    # Same logic as eval/infer.py's postprocess() -- keep in sync if either changes.
    tail = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    marker = "### Answer:"
    if marker in full_text:
        tail = full_text.split(marker, 1)[1]
    next_section = tail.find("### ")
    if next_section != -1:
        tail = tail[:next_section]
    return tail.strip()


@spaces.GPU
def respond(message, history, state, crop, season, query_type):
    if _model is None and _load_error is None:
        _load_model()

    if _load_error is not None:
        return (
            "Sorry -- the model failed to load on this Space. This is a "
            f"server-side problem, not something wrong with your question. "
            f"(error: {_load_error})"
        )

    # Deliberately NOT folding `history` into the prompt -- see module docstring.
    # Training data is single-turn; the visible chat history is for the user's
    # benefit only, not fed back into the model.
    context = _build_context(state, crop, season, query_type)
    prompt = PROMPT_TEMPLATE.format(instruction=message, input=context)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    start = time.monotonic()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            temperature=None,
            pad_token_id=_tokenizer.pad_token_id,
            # Same repetition-loop fix as eval/infer.py -- keep both in sync.
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
    elapsed = time.monotonic() - start

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_reply = _tokenizer.decode(generated, skip_special_tokens=True).strip()
    reply = _postprocess(prompt + raw_reply, prompt)

    print(f"[fasal-demo] generated {len(generated)} tokens in {elapsed:.1f}s", flush=True)
    return reply or "(no output -- try rephrasing the question)"


DESCRIPTION = (
    "**Fasal** -- a multilingual farmer-advisory model, QLoRA fine-tuned from "
    f"`{BASE_MODEL}` on real Kisan Call Centre queries across 8 Indian languages. "
    "Ask a farming question in Hindi, Gujarati, Marathi, Tamil, Bengali, Kannada, "
    "Punjabi, or English. Filling in the optional fields below (State/Crop/Season/"
    "Query type) matches how the model was trained and usually gives sharper, more "
    "specific answers.\n\n"
    "*Running on free ZeroGPU hardware -- first reply after startup can take a "
    "minute or two while the model loads and a GPU is allocated.*"
)

chat = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Fasal -- Multilingual Farmer Advisor",
    description=DESCRIPTION,
    additional_inputs=[
        gr.Textbox(label="State (optional)", placeholder="e.g. Gujarat"),
        gr.Textbox(label="Crop (optional)", placeholder="e.g. Cotton"),
        gr.Textbox(label="Season (optional)", placeholder="e.g. Kharif"),
        gr.Textbox(label="Query type (optional)", placeholder="e.g. Plant Protection"),
    ],
    examples=[
        ["મારા કપાસના પાકમાં સફેદ માખી આવી છે શું છંટકાવ કરું", "Gujarat", "Cotton", "Kharif", "Plant Protection"],
        ["What is the best time to sow groundnut?", "Tamil Nadu", "Groundnut", "Kharif", "Fertilizer Use"],
        ["பருத்தியில் இலை சுருள் நோயை எப்படி கட்டுப்படுத்துவது?", "", "", "", ""],
    ],
)

if __name__ == "__main__":
    chat.launch()
