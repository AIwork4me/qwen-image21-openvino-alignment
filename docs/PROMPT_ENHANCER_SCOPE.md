# Prompt-Enhancer Models — Scope Note (v2)

The official ComfyUI Qwen-Image 2.1 artifact repository
(`Comfy-Org/Qwen-Image-2.1`, `text_encoders/`) also ships:

- `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` (9,471,072,252 B)
- `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors` (9,471,072,252 B)

These are **Qwen3.5-9B prompt-enhancer** ("PE") models: optional ComfyUI components
that rewrite/expand the user prompt *before* conditioning. They are NOT the
Qwen3-VL-8B core conditioning encoder of Qwen-Image 2.1.

## v2 scope decision

- The v2 numerical-alignment matrix (`C_BF16` / `C_INT8` / `C_W4A8` vs references and
  OpenVINO) covers **only the core conditioning encoder**
  (`qwen3vl_8b_*` artifacts, SHA256-frozen in
  `artifacts/v2/manifests/comfyui_encoder_artifacts.json`).
- PE models are **out of scope** for embedding comparisons; no PE output is mixed
  into any v2 comparison.
- If a production pipeline uses a prompt enhancer, the conditioning-input
  distribution changes; v2 conclusions apply to the encoder given identical inputs,
  which is the controlled variable of this study.
- PE artifacts were NOT downloaded in v2 (recorded here solely to document their
  existence and exclusion).
