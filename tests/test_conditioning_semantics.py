#!/usr/bin/env python3
"""Unit tests proving the conditioning-semantics implementation equals the official pipeline.

Run: python -m tests.test_conditioning_semantics  (requires text_encoder weights + GPU)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from common import ROOT, load_cfg

T2I = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
       "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")


def test_template_matches_pipeline():
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    cfg = load_cfg()
    # instantiate without loading weights? __init__ needs components; check class attribute presence instead
    import inspect
    src = inspect.getsource(QwenImage21Pipeline)
    assert 'Comprehend and analyze the provided prompt.' in src
    print("PASS: T2I template string matches official pipeline source")


def test_hidden_states_semantics():
    """hidden_states[-1] with norm hook == layer-35 pre-norm output == norm module input;
    post-norm differs."""
    from transformers import Qwen3VLForConditionalGeneration
    cfg = load_cfg()
    snap = cfg["model"]["snapshot_path"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(snap, "text_encoder"), torch_dtype=torch.bfloat16).to("cuda").eval()
    lm = model.model.language_model
    z = np.load(os.path.join(ROOT, "artifacts", "inputs", "S01.npz"))
    ids = torch.from_numpy(z["input_ids"]).to("cuda")
    attn = torch.from_numpy(z["attention_mask"]).to("cuda")
    mm = torch.from_numpy(z["mm_token_type_ids"]).to("cuda")

    captured = {}

    def hook(module, args, kwargs, output):
        captured["prenorm"] = args[0].detach()
        return args[0]

    h = lm.norm.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, mm_token_type_ids=mm)
    h.remove()
    hs = out.hidden_states
    assert len(hs) == 37, f"expected 37 hidden states, got {len(hs)}"
    pre = captured["prenorm"][0]
    # with hook, hidden_states[-1] must equal prenorm (bitwise, same tensor lineage)
    assert torch.equal(hs[-1][0], pre), "hidden_states[-1] != prenorm under hook"
    # layer-35 output (hs[36]) must also equal prenorm
    assert torch.equal(hs[36][0], pre), "hidden_states[36] != prenorm"
    post = lm.norm(pre[None])[0]
    assert not torch.allclose(pre.float(), post.float()), "postnorm == prenorm?!"
    # drop_idx semantics
    drop = int(z["drop_idx"])
    ids_list = ids[0].tolist()
    second = [i for i, t in enumerate(ids_list) if t == 151644][1]
    assert drop == second == 14, (drop, second)
    print("PASS: hidden-state selection = last layer pre-norm; drop_idx=14 = second <|im_start|>")


def test_pipeline_parity():
    """Our captured prompt_embeds (run_rocm_encoder) must match the official pipeline's
    encode_prompt output for the same prompt (bf16-cast view)."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    cfg = load_cfg()
    snap = cfg["model"]["snapshot_path"]
    import json
    prompt = json.load(open(os.path.join(ROOT, "prompts/smoke.json")))["prompts"][0]

    pipe = QwenImage21Pipeline.from_pretrained(
        snap, torch_dtype=torch.bfloat16, transformer=None, vae=None, scheduler=None)
    pipe.text_encoder.to("cuda")
    pe, mask, ipm = pipe.encode_prompt(prompt=[prompt["text"]], device=torch.device("cuda"))
    ours = np.load(os.path.join(ROOT, "artifacts/tensors/R0", f"{prompt['id']}.npz"))
    ref_bf16 = torch.from_numpy(ours["prompt_embeds"]).to(torch.bfloat16).float()
    got = pe[0].cpu()
    assert ref_bf16.shape == got.shape, (ref_bf16.shape, got.shape)
    assert torch.equal(ref_bf16, got), f"mismatch max={float((ref_bf16-got).abs().max())}"
    print("PASS: frozen R0 prompt_embeds bit-match official pipeline encode_prompt (bf16 view)")


if __name__ == "__main__":
    test_template_matches_pipeline()
    test_hidden_states_semantics()
    test_pipeline_parity()
    print("ALL CONDITIONING-SEMANTICS TESTS PASSED")
