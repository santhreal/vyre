# `logical` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

2 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-libs::logical::nand` | `libs` | `logical` | 0:a:ReadOnly:U32<br>1:b:ReadOnly:U32<br>2:out:ReadWrite:U32 | `logical` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::logical::nand |
| `vyre-libs::logical::nor` | `libs` | `logical` | 0:a:ReadOnly:U32<br>1:b:ReadOnly:U32<br>2:out:ReadWrite:U32 | `logical` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::logical::nor |
