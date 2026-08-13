# `visual` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

10 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-libs::visual::blur` | `libs` | `visual` | 0:input:ReadOnly:U32<br>1:scratch:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::blur<br>&nbsp;&nbsp;vyre-primitives::math::conv1d |
| `vyre-libs::visual::box_shadow` | `libs` | `visual` | 0:out:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::box_shadow |
| `vyre-libs::visual::cell_grid` | `libs` | `visual` | 0:cells:ReadOnly:U32<br>1:out:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::cell_grid<br>&nbsp;&nbsp;vyre-primitives::visual::packed_rgba_map |
| `vyre-libs::visual::composite` | `libs` | `visual` | 0:fg:ReadOnly:U32<br>1:bg:ReadOnly:U32<br>2:out:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::composite<br>&nbsp;&nbsp;vyre-primitives::visual::packed_rgba_map |
| `vyre-libs::visual::downsample` | `libs` | `visual` | 0:input:ReadOnly:U32<br>1:output:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::downsample |
| `vyre-libs::visual::filter_chain` | `libs` | `visual` | 0:pixels:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::filter_chain<br>&nbsp;&nbsp;vyre-primitives::visual::packed_rgba_map |
| `vyre-libs::visual::glass` | `libs` | `visual` | 0:scene:ReadOnly:U32<br>1:scratch:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::glass<br>&nbsp;&nbsp;vyre-libs::visual::blur<br>&nbsp;&nbsp;&nbsp;&nbsp;vyre-primitives::math::conv1d |
| `vyre-libs::visual::gradient` | `libs` | `visual` | 0:output:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::gradient<br>&nbsp;&nbsp;vyre-primitives::visual::packed_rgba_map |
| `vyre-libs::visual::upsample` | `libs` | `visual` | 0:input:ReadOnly:U32<br>1:output:ReadWrite:U32 | `visual` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::visual::upsample |
| `vyre-primitives::visual::packed_rgba_map` | `primitive` | `visual` | 0:in:ReadOnly:U32<br>1:out:ReadWrite:U32 | `visual`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::visual::packed_rgba_map |
