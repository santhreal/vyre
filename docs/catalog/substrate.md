# `substrate` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

2 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-libs::substrate::indexed_map` | `libs` | `substrate` | 0:input:ReadOnly:U32<br>1:out:ReadWrite:U32 | `full` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::substrate::indexed_map<br>&nbsp;&nbsp;vyre-libs::substrate::indexed_map |
| `vyre-libs::substrate::strided_accumulate` | `libs` | `substrate` | 0:values:ReadOnly:U32<br>0:scratch:Workgroup:U32<br>1:out:ReadWrite:U32 | `full` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::substrate::strided_accumulate<br>&nbsp;&nbsp;vyre-libs::substrate::strided_accumulate |
