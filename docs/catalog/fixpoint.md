# `fixpoint` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

2 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-primitives::fixpoint::bitset_fixpoint` | `primitive` | `fixpoint` | 0:current:ReadOnly:U32<br>1:next:ReadOnly:U32<br>2:fp_changed:ReadWrite:U32 | `fixpoint`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::fixpoint::bitset_fixpoint |
| `vyre-primitives::fixpoint::persistent_fixpoint` | `primitive` | `fixpoint` | 0:current:ReadWrite:U32<br>1:next:ReadWrite:U32<br>2:changed:ReadWrite:U32 | `fixpoint`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::fixpoint::persistent_fixpoint |
