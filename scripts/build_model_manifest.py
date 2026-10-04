#!/usr/bin/env python3
"""Freeze the Qwen-Image-2.1 model manifest: revision, config hashes, weight hashes.

Proves that ROCm and OpenVINO paths originate from the SAME source checkpoint.
Also hashes the OpenVINO artifacts (customer INT8, customer FP16, clean export).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json, sha256_file


def hash_dir_files(d, pattern_list=None):
    out = {}
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, d)
            if pattern_list and not any(pat in rel for pat in pattern_list):
                continue
            sz = os.path.getsize(p)
            out[rel] = {"sha256": sha256_file(p), "bytes": sz, "hashed": True}
    return out


def main():
    cfg = load_cfg()
    snap = cfg["model"]["snapshot_path"]
    man = {
        "repo_id": cfg["model"]["repo_id"],
        "snapshot_revision": os.path.basename(snap),
        "snapshot_path": snap,
        "source": "huggingface_hub snapshot_download via hf-mirror (proxy environment)",
        "configs": {},
        "openvino_artifacts": {},
    }
    for sub in ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
                "text_encoder/generation_config.json", "transformer/config.json", "vae/config.json",
                "processor/tokenizer_config.json", "processor/preprocessor_config.json",
                "processor/special_tokens_map.json", "processor/tokenizer.json",
                "processor/vocab.json", "processor/added_tokens.json", "processor/chat_template.jinja"]:
        p = os.path.join(snap, sub)
        if os.path.exists(p):
            man["configs"][sub] = sha256_file(p)
        else:
            # tokenizer.json may live only in cache or with different layout
            alt = os.path.join(snap, sub.replace("processor/", "processor/"))
            man["configs"][sub] = "MISSING"

    # weight shard presence + hashes (big; hash fully for the text encoder which both paths use)
    for sub in ["text_encoder", "transformer", "vae"]:
        d = os.path.join(snap, sub)
        if os.path.isdir(d):
            man[f"{sub}_weights"] = hash_dir_files(d)
    # OpenVINO artifacts
    for label, path in [("customer_int8", cfg["model"]["customer_ov_artifact"]),
                        ("customer_fp16", "/root/models/qwen3vl-openvino-fp16"),
                        ("clean_fp32_export", "/valdata/models/ov_clean_fp32")]:
        if os.path.isdir(path):
            files = {}
            for root, _, fs in os.walk(path):
                for fn in sorted(fs):
                    p = os.path.join(root, fn)
                    rel = os.path.relpath(p, path)
                    if os.path.getsize(p) < 200 * 2 ** 20:
                        files[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
                    else:
                        files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
            man["openvino_artifacts"][label] = {"path": path, "files": files}
        else:
            man["openvino_artifacts"][label] = {"path": path, "status": "NOT_FOUND"}

    save_json(man, os.path.join(ROOT, "artifacts", "model_manifest.json"))
    print("model manifest saved;", len(man.get("text_encoder_weights", {})), "TE shards;",
          len(man.get("transformer_weights", {})), "transformer shards;",
          len(man.get("vae_weights", {})), "vae files")


if __name__ == "__main__":
    main()
