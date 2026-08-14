# `builder` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

2 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-libs::builder::indexed_map` | `libs` | `builder` | 0:input:ReadOnly:U32<br>1:out:ReadWrite:U32 | `full` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::builder::indexed_map<br>&nbsp;&nbsp;vyre-libs::builder::indexed_map |
| `vyre-libs::builder::strided_accumulate` | `libs` | `builder` | 0:values:ReadOnly:U32<br>0:scratch:Workgroup:U32<br>1:out:ReadWrite:U32 | `full` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::builder::strided_accumulate<br>&nbsp;&nbsp;vyre-libs::builder::strided_accumulate |
