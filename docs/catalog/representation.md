# `representation` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

1 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-libs::representation::unpack_4bit_f32` | `libs` | `representation` | 0:input:ReadOnly:U32<br>1:output:ReadWrite:F32 | `full` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::representation::unpack_4bit_f32 |
