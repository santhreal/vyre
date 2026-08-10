# `geom` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

2 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-primitives::geom::clifford2_geometric_product` | `primitive` | `geom` | 0:lhs:ReadOnly:U32<br>1:rhs:ReadOnly:U32<br>2:out:ReadWrite:U32 | `geom`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::geom::clifford2_geometric_product |
| `vyre-primitives::geom::tfn_scalar_mix` | `primitive` | `geom` | 0:features:ReadOnly:U32<br>1:weights:ReadOnly:U32<br>2:out:ReadWrite:U32 | `geom`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::geom::tfn_scalar_mix |
