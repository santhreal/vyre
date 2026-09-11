# vyre-libs::nn SKILL

Neural-network primitives: activation, linear layers, normalization,
attention, MoE, optimizers, and quantization. Every op is a Cat-A
composition over `vyre-ops` primitives and lower-level `vyre-libs::math`
functions.

## Coverage (shipped)

- Activations (`nn-activation`): `relu`, `gelu`, `silu`, `swiglu`,
  `leaky_relu_sq`, `logit_softcap`, `sigmoid_gate`, `skip_gate`,
  `cross_entropy`, `embedding`, residual helpers.
- Linear (`nn-linear`): affine / tiled / fused-activation builders,
  including 4-bit paths where feature-gated.
- Normalization (`nn-norm`): `layer_norm`, `rms_norm`, `gated_rms_norm`,
  `last_dim_l2_norm`, `layerwise_ln_scale`.
- Attention (`nn-attention`): `softmax`, `attention`,
  `flash_attention` / `flash_attention_2`, GQA/MLA helpers, RoPE /
  QK-gain / KV-cache utilities.
- MoE, optimizers, quantization, and inference-graph composition under
  their feature gates (`nn-moe`, `nn-inference`, and activation-gated
  `optim` / `quant` modules).

## Witness sources

- `relu`: identity for non-negative u32 lanes where that contract applies.
- `layer_norm`: PyTorch `torch.nn.LayerNorm` reference with `eps=1e-5`,
  plus edge cases (constant input, zero variance, large variance).
- `softmax`: probabilities summing to 1 within documented `f32` tolerance.
- `attention` / flash variants: scaled-dot-product reference fixtures in
  crate tests.

## Benchmark targets (criterion)

- `softmax` on 4096 F32 elements: ≤ 500 µs sequential; shared-memory
  variants use the backend's checked-in baseline.
- `layer_norm` on 4096 F32 elements: ≤ 500 µs sequential.
- `attention` at seq_len=128, head_dim=64: sequential and dispatch
  baselines live with the criterion targets for this crate.

## Backend parity contract

- F32 ops must be bit-identical across backends on inputs whose
  reduction tree is associativity-safe. For non-associative float
  reductions, document an explicit tolerance ≤ `f32::EPSILON * n`.

## Shape contract

- `softmax(input, output, n)`: both 1-D F32 length `n`.
- `layer_norm(input, output, n, eps)`: both 1-D F32 length `n`.
- `attention(q, k, v, out, s, d)`: all four 2-D F32 shape `[s, d]`.
- All builders route through `check_tensors` for collision, dtype,
  and overflow. No op-specific shape logic lives outside the builder.
