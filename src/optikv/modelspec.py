"""Model architecture loaded from a Hugging Face config snapshot or a hardcoded YAML."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from optikv.kvbytes import gqa_bytes_per_token, mla_bytes_per_token

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "configs" / "models"


class ContextCapError(ValueError):
    """Raised when a simulation asks for a context past the model's study cap."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    source_status: str
    kv_arch: str
    num_hidden_layers: int
    num_full_attention_layers: int
    num_linear_attention_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    simulation_context_cap: int
    vocab_size: int
    torch_dtype: str
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None
    q_lora_rank: int | None = None
    first_k_dense_replace: int = 0
    moe_intermediate_size: int | None = None
    num_experts_per_tok: int | None = None
    n_shared_experts: int | None = None
    linear_num_key_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_num_value_heads: int | None = None
    linear_value_head_dim: int | None = None
    full_attention_interval: int | None = None
    kv_note: str = ""
    context_cap_note: str = ""

    def kv_bytes_per_token(self, dtype_bytes: int) -> int:
        if self.kv_arch == "mla":
            if self.kv_lora_rank is None or self.qk_rope_head_dim is None:
                raise ValueError(f"{self.name} is MLA but kv_lora_rank or qk_rope_head_dim is missing")
            return mla_bytes_per_token(
                self.num_hidden_layers,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                dtype_bytes,
            )
        return gqa_bytes_per_token(
            self.num_full_attention_layers,
            self.num_key_value_heads,
            self.head_dim,
            dtype_bytes,
        )

    def check_context(self, n_tokens: int) -> None:
        if n_tokens > self.simulation_context_cap:
            raise ContextCapError(
                f"{self.name} context {n_tokens} exceeds simulation cap "
                f"{self.simulation_context_cap}. {self.context_cap_note}"
            )


def _text_config(cfg: dict) -> dict:
    text = cfg.get("text_config")
    if isinstance(text, dict) and "num_hidden_layers" in text:
        return text
    return cfg


def architecture_from_hf(cfg: dict) -> dict:
    """Parse a HF config.json object. Hybrid models need an explicit layer split."""
    text = _text_config(cfg)
    n_layers = int(text["num_hidden_layers"])
    n_heads = int(text["num_attention_heads"])
    n_kv = int(text.get("num_key_value_heads", n_heads))
    hidden = int(text["hidden_size"])
    if text.get("head_dim"):
        head_dim = int(text["head_dim"])
    elif n_heads and hidden % n_heads == 0:
        head_dim = hidden // n_heads
    else:
        head_dim = 0
    layer_types = text.get("layer_types")
    interval = text.get("full_attention_interval")
    has_linear = any(k.startswith("linear_") for k in text)
    kv_arch = "gqa"
    n_full = n_layers
    n_lin = 0
    note = ""
    if isinstance(layer_types, list) and layer_types:
        n_full = sum(1 for item in layer_types if item == "full_attention")
        n_lin = sum(1 for item in layer_types if item != "full_attention")
        if n_full > 0 and any("linear" in str(item) for item in layer_types):
            kv_arch = "hybrid_gqa"
            note = (
                "Per-token KV bytes count full_attention layers listed in layer_types. "
                "Linear-attention layers keep a fixed recurrent state and are omitted "
                "from the per-token term."
            )
    elif interval and has_linear:
        interval = int(interval)
        if interval <= 0 or n_layers % interval != 0:
            raise ValueError("full_attention_interval does not divide num_hidden_layers")
        n_full = n_layers // interval
        n_lin = n_layers - n_full
        kv_arch = "hybrid_gqa"
        note = (
            "Per-token KV bytes count num_hidden_layers / full_attention_interval full-attention "
            "layers. Linear-attention layers keep a fixed recurrent state and are omitted "
            "from the per-token term."
        )
    if "kv_lora_rank" in text and "qk_rope_head_dim" in text:
        kv_arch = "mla"
        n_full = n_layers
        n_lin = 0
        note = (
            "MLA per-token KV uses num_hidden_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes. "
            "num_nextn_predict_layers is not added."
        )
        if not head_dim:
            head_dim = int(text.get("qk_nope_head_dim", 0)) + int(text.get("qk_rope_head_dim", 0))
    if kv_arch != "mla" and head_dim <= 0:
        raise ValueError("head_dim missing and hidden_size is not divisible by num_attention_heads")
    dtype = str(text.get("torch_dtype") or text.get("dtype") or "bfloat16")
    return {
        "kv_arch": kv_arch,
        "num_hidden_layers": n_layers,
        "num_full_attention_layers": n_full,
        "num_linear_attention_layers": n_lin,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "intermediate_size": int(text.get("intermediate_size", 0)),
        "max_position_embeddings": int(text.get("max_position_embeddings", 0)),
        "vocab_size": int(text.get("vocab_size", 0)),
        "torch_dtype": dtype,
        "kv_lora_rank": text.get("kv_lora_rank"),
        "qk_rope_head_dim": text.get("qk_rope_head_dim"),
        "qk_nope_head_dim": text.get("qk_nope_head_dim"),
        "v_head_dim": text.get("v_head_dim"),
        "q_lora_rank": text.get("q_lora_rank"),
        "first_k_dense_replace": int(text.get("first_k_dense_replace") or 0),
        "moe_intermediate_size": text.get("moe_intermediate_size"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "n_shared_experts": text.get("n_shared_experts"),
        "linear_num_key_heads": text.get("linear_num_key_heads"),
        "linear_key_head_dim": text.get("linear_key_head_dim"),
        "linear_num_value_heads": text.get("linear_num_value_heads"),
        "linear_value_head_dim": text.get("linear_value_head_dim"),
        "full_attention_interval": int(interval) if interval else None,
        "kv_note": note,
    }


def _optional_int(value):
    if value is None or value == "":
        return None
    return int(value)


def _spec_from_mapping(name: str, meta: dict, arch: dict) -> ModelSpec:
    return ModelSpec(
        name=name,
        hf_id=str(meta.get("hf_id", name)),
        source_status=str(meta.get("source_status", "downloaded")),
        kv_arch=str(arch["kv_arch"]),
        num_hidden_layers=int(arch["num_hidden_layers"]),
        num_full_attention_layers=int(arch["num_full_attention_layers"]),
        num_linear_attention_layers=int(arch["num_linear_attention_layers"]),
        num_attention_heads=int(arch["num_attention_heads"]),
        num_key_value_heads=int(arch["num_key_value_heads"]),
        head_dim=int(arch["head_dim"]),
        hidden_size=int(arch["hidden_size"]),
        intermediate_size=int(arch["intermediate_size"]),
        max_position_embeddings=int(arch["max_position_embeddings"]),
        simulation_context_cap=int(meta["simulation_context_cap"]),
        vocab_size=int(arch.get("vocab_size", 0)),
        torch_dtype=str(arch.get("torch_dtype", "bfloat16")),
        kv_lora_rank=_optional_int(arch.get("kv_lora_rank")),
        qk_rope_head_dim=_optional_int(arch.get("qk_rope_head_dim")),
        qk_nope_head_dim=_optional_int(arch.get("qk_nope_head_dim")),
        v_head_dim=_optional_int(arch.get("v_head_dim")),
        q_lora_rank=_optional_int(arch.get("q_lora_rank")),
        first_k_dense_replace=int(arch.get("first_k_dense_replace") or 0),
        moe_intermediate_size=_optional_int(arch.get("moe_intermediate_size")),
        num_experts_per_tok=_optional_int(arch.get("num_experts_per_tok")),
        n_shared_experts=_optional_int(arch.get("n_shared_experts")),
        linear_num_key_heads=_optional_int(arch.get("linear_num_key_heads")),
        linear_key_head_dim=_optional_int(arch.get("linear_key_head_dim")),
        linear_num_value_heads=_optional_int(arch.get("linear_num_value_heads")),
        linear_value_head_dim=_optional_int(arch.get("linear_value_head_dim")),
        full_attention_interval=_optional_int(arch.get("full_attention_interval")),
        kv_note=str(arch.get("kv_note") or meta.get("kv_note") or ""),
        context_cap_note=str(meta.get("context_cap_note", "")),
    )


def load_model(name: str, models_dir: Path | None = None) -> ModelSpec:
    directory = Path(models_dir) if models_dir else MODELS_DIR
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no model yaml for {name} at {path}")
    meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    if meta.get("source_status") == "hardcoded_gated":
        arch = dict(meta["architecture"])
        arch.setdefault("kv_note", meta.get("kv_note", ""))
        return _spec_from_mapping(name, meta, arch)
    snapshot = meta.get("snapshot")
    if not snapshot:
        raise FileNotFoundError(f"{name} is marked downloaded but has no snapshot")
    snap_path = (path.parent / snapshot).resolve()
    if not snap_path.is_file():
        raise FileNotFoundError(f"missing config snapshot {snap_path}")
    cfg = json.loads(snap_path.read_text(encoding="utf-8"))
    arch = architecture_from_hf(cfg)
    return _spec_from_mapping(name, meta, arch)


def load_catalog(models_dir: Path | None = None) -> dict[str, ModelSpec]:
    directory = Path(models_dir) if models_dir else MODELS_DIR
    catalog = {}
    for path in sorted(directory.glob("*.yaml")):
        spec = load_model(path.stem, directory)
        catalog[spec.name] = spec
    return catalog
