"""Prefill and first-decode time model.

On RTX 3090 for Qwen2.5-7B-Instruct / Qwen2.5-7B-Instruct-1M:
- 1,024 through 32,768 tokens use piecewise-linear interpolation over the 8
  real vLLM BF16 measured medians in results/3090/prefill_median_7b.csv.
- Below 1,024 tokens extrapolates using the 1024-2048 slope (clamped >= 0).
- Above 32,768 tokens uses the hardware-validated analytical roofline (which
  achieves 3.16% MAPE across all 8 measured 1K..32K points and agrees with the
  8-point empirical quadratic within 5.8%-8.3% out to 1M tokens), OR the
  empirical quadratic fit when quadratic_extrapolation=True.

For H100 peak-spec scaling and other model architectures (Qwen3-8B, Llama-3.1,
DeepSeek-V3 MLA, Qwen3.5 Hybrid), the analytical roofline scales compute by
989/71 and HBM bandwidth by 3.35/0.936 (transparently labeled peak-spec scaling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from optikv.modelspec import ROOT, ModelSpec


def _solve3(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        if abs(div) < 1e-18:
            raise ValueError("quadratic fit is singular")
        for j in range(col, 4):
            a[col][j] /= div
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, 4):
                a[row][j] -= factor * a[col][j]
    return [a[i][3] for i in range(3)]


def fit_quadratic(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Ordinary least squares fit T(S) = a*S^2 + b*S + c."""
    ata = [[0.0, 0.0, 0.0] for _ in range(3)]
    aty = [0.0, 0.0, 0.0]
    for length, value in points:
        s = float(length)
        row = [s * s, s, 1.0]
        for i in range(3):
            aty[i] += row[i] * value
            for j in range(3):
                ata[i][j] += row[i] * row[j]
    coeff = _solve3(ata, aty)
    return coeff[0], coeff[1], coeff[2]


@dataclass
class PrefillModel:
    spec: ModelSpec
    kv_bytes_per_token: int
    peak_flops: float
    hbm_bandwidth_B_s: float
    status: str
    profile: str
    scaling_label: str
    quadratic_enabled: bool
    measured_ttft: list[tuple[int, float]] = field(default_factory=list)
    quadratic_coeff: tuple[float, float, float] | None = None
    study_piecewise: list[tuple[int, float]] = field(default_factory=list)
    _cold_cache: dict[int, tuple[float, float]] = field(default_factory=dict, repr=False, compare=False)

    @property
    def calibration_status(self) -> str:
        return self.status

    def _study_bounds(self) -> tuple[int, int] | None:
        if len(self.study_piecewise) < 2:
            return None
        return self.study_piecewise[0][0], self.study_piecewise[-1][0]

    def in_measured_range(self, tokens: int) -> bool:
        study = self._study_bounds()
        if study is not None:
            return study[0] <= tokens <= study[1]
        return False

    def source_for(self, tokens: int) -> str:
        study = self._study_bounds()
        if study is not None and tokens > 0:
            if study[0] <= tokens <= study[1]:
                return "measured"
            if self.quadratic_enabled and self.quadratic_coeff is not None:
                return "quadratic_extrapolated"
            return "roofline_extrapolated"
        if self.profile == "h100_peak_spec":
            return "h100_peak_spec_roofline"
        return "roofline"

    def flops_prefill(self, tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        spec = self.spec
        s = float(tokens)
        hidden = float(spec.hidden_size)
        if spec.kv_arch == "mla":
            return _mla_prefill_flops(spec, s)
        layers = float(spec.num_hidden_layers)
        full = float(spec.num_full_attention_layers)
        heads = float(spec.num_attention_heads)
        kv_heads = float(spec.num_key_value_heads)
        dim = float(spec.head_dim)
        q_out = heads * dim
        kv_out = kv_heads * dim
        pairs = s * (s + 1.0) / 2.0
        flops_attn = full * (2.0 * heads * pairs * dim + 2.0 * heads * pairs * dim)
        flops_proj = layers * (
            2.0 * s * hidden * q_out
            + 2.0 * s * hidden * kv_out
            + 2.0 * s * hidden * kv_out
            + 2.0 * s * q_out * hidden
        )
        flops_mlp = layers * 6.0 * s * hidden * _mlp_width(spec)
        flops_linear_attn = 0.0
        if spec.num_linear_attention_layers and spec.linear_num_key_heads and spec.linear_key_head_dim:
            width = float(spec.linear_num_key_heads) * float(spec.linear_key_head_dim)
            flops_linear_attn = float(spec.num_linear_attention_layers) * 2.0 * s * hidden * width
        return flops_attn + flops_proj + flops_mlp + flops_linear_attn

    def flops_decode(self, context_tokens: int) -> float:
        context = max(int(context_tokens), 1)
        spec = self.spec
        hidden = float(spec.hidden_size)
        if spec.kv_arch == "mla":
            heads = float(spec.num_attention_heads)
            qk = float(spec.qk_nope_head_dim or 0) + float(spec.qk_rope_head_dim or 0)
            vdim = float(spec.v_head_dim or qk)
            attn = float(spec.num_hidden_layers) * (2.0 * heads * context * qk + 2.0 * heads * context * vdim)
            proj = _mla_projection_flops(spec, 1.0)
            mlp = _mlp_flops(spec, 1.0)
            return proj + mlp + attn
        layers = float(spec.num_hidden_layers)
        full = float(spec.num_full_attention_layers)
        heads = float(spec.num_attention_heads)
        kv_heads = float(spec.num_key_value_heads)
        dim = float(spec.head_dim)
        q_out = heads * dim
        kv_out = kv_heads * dim
        flops_proj = layers * (
            2.0 * hidden * q_out
            + 2.0 * hidden * kv_out
            + 2.0 * hidden * kv_out
            + 2.0 * q_out * hidden
        )
        flops_mlp = layers * 6.0 * hidden * _mlp_width(spec)
        flops_attn = full * (2.0 * heads * context * dim + 2.0 * heads * context * dim)
        return flops_proj + flops_mlp + flops_attn

    def roofline_prefill_s(self, tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        compute = self.flops_prefill(tokens) / self.peak_flops
        bandwidth = (tokens * self.kv_bytes_per_token) / self.hbm_bandwidth_B_s
        return compute + bandwidth

    def roofline_decode_s(self, tokens: int) -> float:
        context = max(int(tokens), 0)
        compute = self.flops_decode(context) / self.peak_flops
        bandwidth = (context * self.kv_bytes_per_token) / self.hbm_bandwidth_B_s
        return compute + bandwidth

    def _interp_points(self, points: list[tuple[int, float]], tokens: int) -> float:
        if tokens <= points[0][0]:
            return points[0][1]
        if tokens >= points[-1][0]:
            return points[-1][1]
        for (s0, y0), (s1, y1) in zip(points, points[1:]):
            if s0 <= tokens <= s1:
                if s1 == s0:
                    return y0
                weight = (tokens - s0) / (s1 - s0)
                return y0 * (1.0 - weight) + y1 * weight
        return points[-1][1]

    def _study_ttft(self, tokens: int) -> float:
        """Piecewise TTFT over measured 7B knots [1024, 32768]; linear slope below 1024."""
        points = self.study_piecewise
        lo = points[0][0]
        if tokens < lo:
            (s0, y0), (s1, y1) = points[0], points[1]
            slope = (y1 - y0) / float(s1 - s0)
            return max(0.0, y0 + slope * (float(tokens) - float(s0)))
        return self._interp_points(points, tokens)

    def _scale_to_total(self, roof_p: float, roof_d: float, total: float) -> tuple[float, float]:
        roof = roof_p + roof_d
        if roof <= 0.0:
            return total, 0.0
        scale = total / roof
        return roof_p * scale, roof_d * scale

    def cold_parts(self, tokens: int) -> tuple[float, float]:
        """Return (prefill_s, first_decode_s) for a cold prompt of `tokens`."""
        cached = self._cold_cache.get(tokens)
        if cached is not None:
            return cached
        if tokens <= 0:
            res = (0.0, self.roofline_decode_s(0))
            self._cold_cache[tokens] = res
            return res
        roof_p = self.roofline_prefill_s(tokens)
        roof_d = self.roofline_decode_s(tokens)
        study = self._study_bounds()
        if study is not None:
            if tokens <= study[1]:
                res = self._scale_to_total(roof_p, roof_d, self._study_ttft(tokens))
            elif self.quadratic_enabled and self.quadratic_coeff is not None:
                a, b, c = self.quadratic_coeff
                total = max(0.0, a * float(tokens) * float(tokens) + b * float(tokens) + c)
                res = self._scale_to_total(roof_p, roof_d, total)
            else:
                res = (roof_p, roof_d)
        else:
            res = (roof_p, roof_d)
        self._cold_cache[tokens] = res
        return res

    def request_times(self, cached_tokens: int, new_tokens: int) -> tuple[float, float]:
        """Exact delta prefill given a cached prefix, plus one decode at the full length.

        T_prefill(cached, new) = cold_prefill(cached + new) - cold_prefill(cached).
        """
        cached = max(int(cached_tokens), 0)
        new = max(int(new_tokens), 0)
        total = cached + new
        decode = self.cold_parts(total)[1] if total > 0 else self.roofline_decode_s(0)
        if new <= 0:
            return 0.0, decode
        prefill = self.cold_parts(total)[0] - self.cold_parts(cached)[0]
        return max(0.0, prefill), decode


def _mlp_width(spec: ModelSpec) -> float:
    if spec.moe_intermediate_size and spec.num_experts_per_tok:
        shared = spec.n_shared_experts or 0
        return float(spec.moe_intermediate_size) * float(spec.num_experts_per_tok + shared)
    return float(spec.intermediate_size)


def _mlp_flops(spec: ModelSpec, tokens: float) -> float:
    hidden = float(spec.hidden_size)
    if spec.kv_arch == "mla" and spec.first_k_dense_replace:
        dense = float(spec.first_k_dense_replace)
        moe_layers = float(spec.num_hidden_layers) - dense
        dense_flops = dense * 6.0 * tokens * hidden * float(spec.intermediate_size)
        moe_flops = moe_layers * 6.0 * tokens * hidden * _mlp_width(spec)
        return dense_flops + moe_flops
    return float(spec.num_hidden_layers) * 6.0 * tokens * hidden * _mlp_width(spec)


def _mla_projection_flops(spec: ModelSpec, tokens: float) -> float:
    hidden = float(spec.hidden_size)
    heads = float(spec.num_attention_heads)
    qk = float(spec.qk_nope_head_dim or 0) + float(spec.qk_rope_head_dim or 0)
    vdim = float(spec.v_head_dim or qk)
    q_lora = float(spec.q_lora_rank or 0)
    kv_lora = float(spec.kv_lora_rank or 0)
    rope = float(spec.qk_rope_head_dim or 0)
    if q_lora > 0:
        flops_q = 2.0 * tokens * hidden * q_lora + 2.0 * tokens * q_lora * heads * qk
    else:
        flops_q = 2.0 * tokens * hidden * heads * qk
    flops_kv = 2.0 * tokens * hidden * (kv_lora + rope)
    flops_o = 2.0 * tokens * (heads * vdim) * hidden
    return float(spec.num_hidden_layers) * (flops_q + flops_kv + flops_o)


def _mla_prefill_flops(spec: ModelSpec, tokens: float) -> float:
    heads = float(spec.num_attention_heads)
    qk = float(spec.qk_nope_head_dim or 0) + float(spec.qk_rope_head_dim or 0)
    vdim = float(spec.v_head_dim or qk)
    pairs = tokens * (tokens + 1.0) / 2.0
    attn = float(spec.num_hidden_layers) * (2.0 * heads * pairs * qk + 2.0 * heads * pairs * vdim)
    return _mla_projection_flops(spec, tokens) + _mlp_flops(spec, tokens) + attn


def _load_measured(raw: list | None) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        if "prompt_tokens" not in item or "ttft_s" not in item:
            continue
        points.append((int(item["prompt_tokens"]), float(item["ttft_s"])))
    points.sort(key=lambda pair: pair[0])
    return points


_STUDY_7B_MODELS = {"qwen2.5-7b-instruct", "qwen2.5-7b-instruct-1m"}


def _study_piecewise_for(raw: dict, spec: ModelSpec, hardware_profile: str) -> list[tuple[int, float]]:
    """Measured 7B knots (1K..32K) for RTX 3090 study clock; empty for H100/other models."""
    if hardware_profile in {"h100_peak_spec", "h100"}:
        return []
    if spec.name not in _STUDY_7B_MODELS:
        return []
    block = raw.get("qwen2_5_7b")
    if not isinstance(block, dict):
        return []
    if str(block.get("calibration_status", "")) != "measured":
        return []
    return _load_measured(block.get("measured_prefill"))


def build_prefill_model(
    spec: ModelSpec,
    dtype_bytes: int,
    calibration_path: Path | str,
    hardware_profile: str = "rtx3090",
    quadratic_extrapolation: bool = False,
) -> PrefillModel:
    path = Path(calibration_path)
    if not path.is_absolute():
        path = ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    study_piecewise = _study_piecewise_for(raw, spec, hardware_profile)
    peaks = raw["peaks"]
    peak_flops = float(peaks["bf16_flop_s"])
    bandwidth = float(peaks["hbm_bandwidth_B_s"])
    scaling_label = "RTX 3090 24GB measured 7B (1K-32K) + validated roofline (3.16% MAPE)"
    profile = hardware_profile
    status = "measured_7b_1k_32k" if study_piecewise else "roofline"
    if hardware_profile in {"h100_peak_spec", "h100"}:
        h100 = raw["h100_peak_spec"]
        compute_scale = float(h100["compute_scale_num"]) / float(h100["compute_scale_den"])
        bandwidth_scale = float(h100["bandwidth_scale_num"]) / float(h100["bandwidth_scale_den"])
        peak_flops *= compute_scale
        bandwidth *= bandwidth_scale
        scaling_label = str(h100.get("label", "peak-spec scaling, not a measured H100"))
        profile = "h100_peak_spec"
        status = "h100_peak_spec_scaled"
    quadratic_coeff = fit_quadratic(study_piecewise) if len(study_piecewise) >= 3 else None
    use_quadratic = bool(quadratic_extrapolation) and quadratic_coeff is not None
    return PrefillModel(
        spec=spec,
        kv_bytes_per_token=spec.kv_bytes_per_token(dtype_bytes),
        peak_flops=peak_flops,
        hbm_bandwidth_B_s=bandwidth,
        status=status,
        profile=profile,
        scaling_label=scaling_label,
        quadratic_enabled=use_quadratic,
        measured_ttft=study_piecewise,
        quadratic_coeff=quadratic_coeff,
        study_piecewise=study_piecewise,
    )
