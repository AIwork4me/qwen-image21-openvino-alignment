# Prompt Enhancer Artifacts — Out of v2 Core Matrix Scope

The `Comfy-Org/Qwen-Image-2.1` (and related Comfy-Org release repos) also
distribute prompt-enhancer models named along the lines of:

```
qwen3.5_*_qwen_image_2.1_pe_*
```

Classification: **Prompt Enhancer** — a generative LM used to rewrite/expand
user prompts before conditioning.

They are NOT:

- the Qwen3-VL-8B diffusion conditioning Text Encoder of Qwen-Image 2.1, and
- not precision variants of `qwen3vl_8b_{bf16,int8_convrot,w4a8}.safetensors`.

The v2 core alignment matrix covers only the three official conditioning Text
Encoder artifacts (`C_BF16`, `C_INT8`, `C_W4A8`). Prompt-enhancer models are
excluded from every numerical, trajectory, quality, and production experiment
in this validation. No claim in any v2 report extends to them.
