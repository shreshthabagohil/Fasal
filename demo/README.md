---
title: Fasal Advisor
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: other
---

# Fasal — Multilingual Farmer Advisor (demo)

Live demo for **Fasal**, a QLoRA-fine-tuned `sarvamai/sarvam-1` adapter trained on
real Kisan Call Centre (KCC) farmer queries across 8 Indian languages. Ask an
agricultural question in Hindi, Gujarati, Marathi, Tamil, Bengali, Kannada, Punjabi,
or English and get an answer in the same language.

- **Model:** https://huggingface.co/Algo-Nova/fasal-sarvam1-lora
- **Dataset:** https://huggingface.co/datasets/Algo-Nova/fasal-kcc-instruct
- **Live Space:** https://huggingface.co/spaces/Shreshthabagohil/fasal-advisor-web
- **Source (this Space + full project):** https://github.com/shreshthabagohil/Fasal

## Running locally

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (defaults to http://localhost:7860).

## Notes

- Runs on HF Spaces' free tier via the **Gradio SDK** (not Docker — Docker Spaces
  require a paid plan, Gradio SDK Spaces don't). Weights load in bfloat16, no GPU
  quantization — see `app.py` for why. First reply after a cold start takes
  roughly a minute while the ~2.5B-param model loads; each reply after that takes
  tens of seconds on CPU. This is a deliberate zero-budget tradeoff, not a bug.
- License: this demo serves a model derived from `sarvamai/sarvam-1`, which ships
  under the Sarvam AI Research License (non-commercial-flavored). See
  `LICENSE_BASE_SARVAM.md` in the main repo for the full text.
- This folder mirrors what's actually deployed on the live Space — see `app.py`'s
  header comment and `requirements.txt`'s inline comments for the real dependency
  gotchas hit while deploying (starlette/jinja2 template-cache bug, huggingface_hub
  `HfFolder` removal, ZeroGPU's supported torch versions, adapter-config/`peft`
  version mismatch).
