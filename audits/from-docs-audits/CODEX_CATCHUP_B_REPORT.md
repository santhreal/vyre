# CODEX CATCHUP B Report

## Scope

- FIX 3: progressive lowering contract is in core, with wgpu lowering owned by `vyre-wgpu`.
- FIX 4: canonical primitives expose interpreter, storage, serialization, and capability hooks.
- FIX 5: reference graph interpretation runs through generic node `interpret` calls with a 10k random graph parity oracle.

## Verification

- `cargo check -p vyre`: passed.
- `cargo check -p vyre-primitives`: passed.
- `cargo check -p vyre-reference`: passed.
- `cargo check -p vyre-wgpu`: passed.
- `cargo test -p vyre-reference generic_storage_graph_matches_recursive_oracle_for_10k_programs`: passed.
- `cargo check --workspace`: blocked by pre-existing missing source/include paths in `vyre-conform` and `vyre-conform-enforce`, including missing `vyre-conform/src/{golden,harnesses,parity_10m,regression}.rs`, missing `vyre-conform/src/meta/*` modules, missing `vyre-conform/src/runner/backend.rs`, and missing generated defender include `../conform/defenders/backdoor/add_rhs_keyed_arith_defect.rs`.

## Notes

- `vyre-core/src/lower/wgsl/` is absent; backend-specific lowering lives under `vyre-wgpu/src/lowering/`.
- Hot scalar primitives lower to compact `NodeStorage::BinOp` or `NodeStorage::UnOp`; region, hash, and pattern primitives use `NodeStorage::Extern` with stable payload bytes.
- The generic reference graph interpreter rejects missing dependencies and cycles with actionable errors.
