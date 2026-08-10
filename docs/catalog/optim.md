# `optim` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

5 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-libs::optim::adamw_step` | `libs` | `nn` | 0:params:ReadWrite:F32<br>1:grads:ReadOnly:F32<br>2:m:ReadWrite:F32<br>3:v:ReadWrite:F32 | `nn` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::optim::adamw_step |
| `vyre-libs::optim::ema_apply` | `libs` | `nn` | 0:ema:ReadWrite:F32<br>1:theta:ReadOnly:F32 | `nn` | reference=true inputs=true expected=true tolerance=1 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::optim::ema_apply |
| `vyre-libs::optim::muon_update` | `libs` | `nn` | 0:params:ReadOnly:F32<br>1:grads:ReadOnly:F32<br>2:momentum:ReadWrite:F32<br>3:output:ReadWrite:F32 | `nn` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::optim::muon_update |
| `vyre-libs::optim::muoneq_r` | `libs` | `nn` | 0:params:ReadOnly:F32<br>1:grads:ReadOnly:F32<br>2:momentum:ReadWrite:F32<br>3:output:ReadWrite:F32 | `nn` | reference=true inputs=true expected=true tolerance=8 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::optim::muoneq_r |
| `vyre-libs::optim::newton_schulz_5step` | `libs` | `nn` | 0:mat:ReadOnly:F32<br>1:output:ReadWrite:F32 | `nn` | reference=true inputs=true expected=true tolerance=64 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::optim::newton_schulz_5step<br>&nbsp;&nbsp;vyre-primitives::math::newton_schulz_poly5_f32 |
