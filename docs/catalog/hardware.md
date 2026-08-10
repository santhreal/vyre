# `hardware` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

9 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-intrinsics::hardware::bit_reverse_u32` | `intrinsic` | `hardware` | 0:input:ReadOnly:U32<br>1:out:ReadWrite:U32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::bit_reverse_u32 |
| `vyre-intrinsics::hardware::fma_f32` | `intrinsic` | `hardware` | 0:a:ReadOnly:F32<br>1:b:ReadOnly:F32<br>2:c:ReadOnly:F32<br>3:out:ReadWrite:F32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::ternary_f32_map (internal) |
| `vyre-intrinsics::hardware::inverse_sqrt_f32` | `intrinsic` | `hardware` | 0:input:ReadOnly:F32<br>1:out:ReadWrite:F32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::inverse_sqrt_f32 |
| `vyre-intrinsics::hardware::popcount_u32` | `intrinsic` | `hardware` | 0:input:ReadOnly:U32<br>1:out:ReadWrite:U32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::popcount_u32 |
| `vyre-intrinsics::hardware::storage_barrier` | `intrinsic` | `hardware` | 0:input:ReadOnly:U32<br>1:out:ReadWrite:U32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::storage_barrier |
| `vyre-intrinsics::hardware::subgroup_add` | `intrinsic` | `hardware` | 0:values:ReadOnly:U32<br>1:out:ReadWrite:U32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::subgroup_add |
| `vyre-intrinsics::hardware::subgroup_ballot` | `intrinsic` | `hardware` | 0:cond:ReadOnly:U32<br>1:out:ReadWrite:U32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::subgroup_ballot |
| `vyre-intrinsics::hardware::subgroup_shuffle` | `intrinsic` | `hardware` | 0:values:ReadOnly:U32<br>1:lanes:ReadOnly:U32<br>2:out:ReadWrite:U32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::subgroup_shuffle |
| `vyre-intrinsics::hardware::workgroup_barrier` | `intrinsic` | `hardware` | 0:input:ReadOnly:U32<br>1:out:ReadWrite:U32 | `hardware` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-intrinsics::hardware::workgroup_barrier |
