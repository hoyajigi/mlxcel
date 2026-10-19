"""Regenerate the GLiNER2.5 fixtures from mlx-vlm and Hugging Face.

Usage (a Python with mlx-vlm >= 0.7.8 from git main, torch and transformers;
mlx[cpu] is enough):

    python -I generate_gliner2_5_fixture.py <gliner2.5-small-v1 dir> <fixture dir>

Writes three files into ``<fixture dir>``:

- ``gliner2_5_mlx_vlm.json``: the exact ``extract_entities`` /
  ``classify_text`` results of ``mlx_vlm.models.gliner2_5.Model`` on
  ``fastino/gliner2.5-small-v1`` for the README examples, plus the
  ``_prepare`` input ids, word positions, marker positions and offsets each
  case was scored on. ``src/token_classification/gliner2_5_checkpoint_tests.rs``
  compares mlxcel against it.
- ``deberta_v2_layer_fixture.json``: a tiny random DeBERTa-v2 encoder (its
  weights are in the file) run through mlx-vlm's ``DebertaModel``, and the
  Hugging Face ``build_relative_position`` buckets for lengths 8, 600 and 4096.
  ``src/models/deberta_v2_tests.rs`` checks the encoder against both.
- ``deberta_v2_hf_long.json``: ``transformers`` ``DebertaV2Model`` hidden
  states of the real encoder on an input longer than 128 tokens, where
  mlx-vlm's unbucketed relative positions diverge from Hugging Face.

The live ``config.json`` declares ``model_type: gliner2`` and
``architectures: ["Gliner2ForSchemaExtraction"]``, which mlx-vlm does not map;
its loader keys on ``"BoundaryExtractor" in architectures``, so the checkpoint
is loaded through a temporary directory whose ``config.json`` carries that
architecture and whose other files are symlinks. The ``oracle`` field records
this together with the mlx-vlm, mlx, torch and transformers versions.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

README_TEXT = "Apple hired Sam in New York."
README_LABELS = ["company", "person", "location"]
README_DESCRIPTIONS = {
    "company": "a business or corporation",
    "person": "a human being",
    "location": "a city, state or country",
}
CLASSIFY_TEXT = "The launch was excellent."
CLASSIFY_TASKS = {"sentiment": ["positive", "neutral", "negative"]}
CLASSIFY_SPEC_TASKS = {
    "sentiment": {
        "labels": ["positive", "neutral", "negative"],
        "prompt": "the overall tone of the text",
        "label_descriptions": {"positive": "a favourable opinion"},
        "multi_label": True,
        "threshold": 0.3,
    }
}
# Email and URL words, CJK text under the char splitter, and a text with no
# terminal punctuation.
EMAIL_TEXT = "Contact ada.lovelace@example.com or visit https://example.com/team today"
CJK_TEXT = "东京大学的田中教授访问了北京。"
CJK_LABELS = ["organization", "person", "location"]


def version_of(name: str) -> str:
    dist = importlib.metadata.distribution(name)
    direct = dist.read_text("direct_url.json")
    commit = None
    if direct:
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
    return f"{name} {dist.version}" + (f" (git {commit})" if commit else "")


def boundary_extractor_dir(model_dir: Path) -> Path:
    """A copy of the checkpoint mlx-vlm's loader recognizes."""
    tmp = Path(tempfile.mkdtemp(prefix="gliner2_5-oracle-"))
    for entry in model_dir.iterdir():
        if entry.name == "config.json":
            config = json.loads(entry.read_text(encoding="utf-8"))
            config["architectures"] = ["BoundaryExtractor"]
            (tmp / entry.name).write_text(json.dumps(config), encoding="utf-8")
        elif entry.is_file():
            os.symlink(entry.resolve(), tmp / entry.name)
        elif entry.is_dir() and entry.name == "encoder_config":
            os.symlink(entry.resolve(), tmp / entry.name)
    return tmp


def prepared(model, processor, text, schema, word_splitter="whitespace"):
    item = model._prepare(processor, text, schema, None, word_splitter)
    return {
        "input_ids": item.input_ids[0].tolist(),
        "word_positions": list(item.word_positions),
        "marker_positions": list(item.marker_positions),
        "offsets": [list(o) for o in item.offsets],
        "text": item.text,
    }


def gliner_cases(model_dir: Path) -> dict:
    from mlx_vlm.models.gliner2_5.gliner2_5 import _schema_tokens
    from mlx_vlm.utils import load_model, load_processor

    oracle_dir = boundary_extractor_dir(model_dir)
    model = load_model(oracle_dir)
    processor = load_processor(oracle_dir, add_detokenizer=False)

    cases = []

    def extract(name, text, labels, *, descriptions=None, word_splitter="whitespace", **kw):
        entity_types = {**{l: None for l in labels}, **(descriptions or {})} if descriptions else labels
        schema = _schema_tokens("entities", labels, "[E]", descriptions=descriptions or {})
        cases.append(
            {
                "name": name,
                "kind": "extract",
                "text": text,
                "labels": labels,
                "descriptions": descriptions,
                "word_splitter": word_splitter,
                "options": kw,
                "prepared": prepared(model, processor, text, schema, word_splitter),
                "result": model.extract_entities(
                    processor, text, entity_types, word_splitter=word_splitter, **kw
                ),
            }
        )

    def classify(name, text, tasks):
        prepared_tasks = {}
        for task, spec in tasks.items():
            if isinstance(spec, dict):
                schema = _schema_tokens(
                    task,
                    list(spec["labels"]),
                    "[L]",
                    prompt=spec.get("prompt"),
                    descriptions=spec.get("label_descriptions"),
                )
            else:
                schema = _schema_tokens(task, list(spec), "[L]")
            prepared_tasks[task] = prepared(model, processor, text, schema)
        cases.append(
            {
                "name": name,
                "kind": "classify",
                "text": text,
                "tasks": tasks,
                "prepared": prepared_tasks,
                "result": model.classify_text(processor, text, tasks),
            }
        )

    extract("readme", README_TEXT, README_LABELS, include_confidence=True, include_spans=True)
    extract("readme_plain", README_TEXT, README_LABELS)
    extract(
        "readme_described",
        README_TEXT,
        README_LABELS,
        descriptions=README_DESCRIPTIONS,
        include_confidence=True,
        include_spans=True,
    )
    extract(
        "email_url_unterminated",
        EMAIL_TEXT,
        ["email", "url", "date"],
        include_confidence=True,
        include_spans=True,
    )
    extract(
        "cjk_char",
        CJK_TEXT,
        CJK_LABELS,
        word_splitter="char",
        include_confidence=True,
        include_spans=True,
    )
    classify("sentiment", CLASSIFY_TEXT, CLASSIFY_TASKS)
    classify("sentiment_spec", CLASSIFY_TEXT, CLASSIFY_SPEC_TASKS)

    for case in cases:
        length = (
            len(case["prepared"]["input_ids"])
            if case["kind"] == "extract"
            else max(len(p["input_ids"]) for p in case["prepared"].values())
        )
        assert length < 128, f"{case['name']} has {length} tokens; keep parity cases under 128"

    return {
        "oracle": {
            "mlx_vlm": version_of("mlx-vlm"),
            "mlx": mx.__version__,
            "model": "fastino/gliner2.5-small-v1",
            "note": "config.json architectures patched to [\"BoundaryExtractor\"] in a "
            "temporary copy so mlx-vlm's loader recognizes the checkpoint; weights "
            "are float32 and the model ran in float32",
        },
        "cases": cases,
    }


# Geometry of the tiny DeBERTa-v2 fixture.
TINY = {
    "model_type": "deberta-v2",
    "vocab_size": 32,
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 16,
    "hidden_act": "gelu",
    "max_position_embeddings": 64,
    "max_relative_positions": 64,
    "position_buckets": 4,
    "relative_attention": True,
    "pos_att_type": ["p2c", "c2p"],
    "share_att_key": True,
    "position_biased_input": False,
    "type_vocab_size": 0,
    "layer_norm_eps": 1e-7,
    "norm_rel_ebd": "layer_norm",
    "pad_token_id": 0,
}


def tiny_weights(rng: np.random.RandomState) -> dict:
    d, i, v = TINY["hidden_size"], TINY["intermediate_size"], TINY["vocab_size"]
    weights = {}

    def put(name, shape, scale=0.3):
        weights[name] = rng.uniform(-scale, scale, size=shape).astype(np.float32)

    def norm(name):
        weights[f"{name}.weight"] = (1.0 + rng.uniform(-0.2, 0.2, size=(d,))).astype(np.float32)
        weights[f"{name}.bias"] = rng.uniform(-0.1, 0.1, size=(d,)).astype(np.float32)

    put("embeddings.word_embeddings.weight", (v, d), 1.0)
    norm("embeddings.LayerNorm")
    put("encoder.rel_embeddings.weight", (2 * TINY["position_buckets"], d), 1.0)
    norm("encoder.LayerNorm")
    for layer in range(TINY["num_hidden_layers"]):
        base = f"encoder.layer.{layer}"
        for proj in ("query_proj", "key_proj", "value_proj"):
            put(f"{base}.attention.self.{proj}.weight", (d, d))
            put(f"{base}.attention.self.{proj}.bias", (d,), 0.1)
        put(f"{base}.attention.output.dense.weight", (d, d))
        put(f"{base}.attention.output.dense.bias", (d,), 0.1)
        norm(f"{base}.attention.output.LayerNorm")
        put(f"{base}.intermediate.dense.weight", (i, d))
        put(f"{base}.intermediate.dense.bias", (i,), 0.1)
        put(f"{base}.output.dense.weight", (d, i))
        put(f"{base}.output.dense.bias", (d,), 0.1)
        norm(f"{base}.output.LayerNorm")
    return weights


def deberta_layer_fixture() -> dict:
    import torch
    from mlx_vlm.models.gliner2_5.config import EncoderConfig
    from mlx_vlm.models.gliner2_5.deberta import DebertaModel
    from transformers.models.deberta_v2.modeling_deberta_v2 import build_relative_position

    rng = np.random.RandomState(2390)
    weights = tiny_weights(rng)
    model = DebertaModel(EncoderConfig.from_dict(TINY))
    mlx_weights = {}
    for key, value in weights.items():
        key = key.replace("encoder.layer.", "encoder.layers.")
        key = key.replace(".attention.self.", ".attention.self_attn.")
        key = key.replace(".LayerNorm.", ".layer_norm.")
        mlx_weights[key] = mx.array(value)
    model.load_weights(list(mlx_weights.items()), strict=True)

    # Row 0 is full length; row 1 is right-padded to four real tokens. Length
    # 6 exceeds the identity range of a 4-bucket table (|distance| <= 2), so
    # the log buckets are exercised.
    input_ids = np.array([[5, 17, 3, 29, 11, 8], [21, 2, 30, 7, 0, 0]], dtype=np.int32)
    attention_mask = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=np.int32)
    output = model(mx.array(input_ids), mx.array(attention_mask))
    mx.eval(output)

    def buckets(length: int) -> list[int]:
        dummy = torch.zeros(1, length, 1)
        rel = build_relative_position(dummy, dummy, bucket_size=256, max_position=512)[0]
        # rel[i][j] depends only on the distance i - j; row 0 holds the
        # distances 0, -1, ..., -(length-1) and the last row length-1, ..., 0.
        return [int(rel[0][-d]) for d in range(-(length - 1), 0)] + [
            int(rel[length - 1][length - 1 - d]) for d in range(length)
        ]

    return {
        "oracle": {
            "mlx_vlm": version_of("mlx-vlm"),
            "mlx": mx.__version__,
            "torch": torch.__version__,
            "transformers": version_of("transformers"),
        },
        "config": TINY,
        "weights": {k: {"shape": list(v.shape), "data": v.flatten().tolist()} for k, v in weights.items()},
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "output": np.array(output).astype(np.float32).tolist(),
        "hf_relative_buckets": {
            "note": "Hugging Face build_relative_position(bucket_size=256, max_position=512) "
            "for distances -(L-1)..=(L-1), the live gliner2.5-small-v1 geometry "
            "(position_buckets 256, max_relative_positions -1 -> max_position_embeddings 512)",
            "8": buckets(8),
            "600": buckets(600),
            "4096": buckets(4096),
        },
    }


LONG_TEXT = (
    "The Rust port of a DeBERTa-v2 encoder must bucket relative positions the way "
    "Hugging Face does, because a disentangled attention head reads one relative "
    "embedding per bucket and a distance past the half bucket size is mapped "
    "logarithmically rather than clipped. "
) * 6


def hf_long_fixture(model_dir: Path) -> dict:
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from transformers import DebertaV2Config, DebertaV2Model

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    encoder_config = config.get("encoder_config")
    if encoder_config is None:
        encoder_config = json.loads(
            (model_dir / "encoder_config" / "config.json").read_text(encoding="utf-8")
        )
    hf = DebertaV2Model(DebertaV2Config(**encoder_config)).eval()
    state = {
        key[len("encoder.") :]: value.float()
        for key, value in load_file(str(model_dir / "model.safetensors")).items()
        if key.startswith("encoder.")
    }
    missing, unexpected = hf.load_state_dict(state, strict=False)
    missing = [k for k in missing if "position_ids" not in k]
    assert not missing and not unexpected, (missing, unexpected)

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    ids = tokenizer.encode(LONG_TEXT, add_special_tokens=False).ids
    assert 256 < len(ids) < 512, len(ids)
    with torch.no_grad():
        hidden = hf(
            input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)
        ).last_hidden_state[0]

    # mlx-vlm on the same ids, for the record: it clips distances past 128
    # instead of bucketing them, so its rows past position 128 differ.
    from mlx_vlm.models.gliner2_5.config import EncoderConfig
    from mlx_vlm.models.gliner2_5.deberta import DebertaModel

    mlx_model = DebertaModel(EncoderConfig.from_dict(encoder_config))
    mlx_model.load_weights(
        [
            (
                k.replace("encoder.layer.", "encoder.layers.")
                .replace(".attention.self.", ".attention.self_attn.")
                .replace(".LayerNorm.", ".layer_norm."),
                mx.array(v.numpy()),
            )
            for k, v in state.items()
        ],
        strict=True,
    )
    mlx_hidden = mlx_model(mx.array([ids], dtype=mx.int32))[0]
    mx.eval(mlx_hidden)
    rows = [0, 1, 64, 130, 200, len(ids) - 1]
    hidden_np = hidden.numpy()
    mlx_np = np.array(mlx_hidden)
    return {
        "oracle": {
            "torch": torch.__version__,
            "transformers": version_of("transformers"),
            "mlx_vlm": version_of("mlx-vlm"),
            "mlx": mx.__version__,
            "model": "fastino/gliner2.5-small-v1 encoder.* tensors in DebertaV2Model (float32, CPU)",
        },
        "text": LONG_TEXT,
        "input_ids": ids,
        "rows": rows,
        "hidden_rows": {str(r): hidden_np[r].astype(np.float32).tolist() for r in rows},
        "mlx_vlm_max_abs_diff_per_row": {
            str(r): float(np.abs(hidden_np[r] - mlx_np[r]).max()) for r in rows
        },
    }


def dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    model_dir, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "gliner2_5_mlx_vlm.json", gliner_cases(model_dir))
    dump(out / "deberta_v2_layer_fixture.json", deberta_layer_fixture())
    dump(out / "deberta_v2_hf_long.json", hf_long_fixture(model_dir))


if __name__ == "__main__":
    main()
