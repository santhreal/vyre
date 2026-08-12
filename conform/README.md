# conform/

Conformance = certifiable parity between every vyre backend and the
CPU reference interpreter.

If WGSL emits `7.0`, SPIR-V emits `7.0`, and `vyre-reference` emits
`7.0` for the same program, the op conforms. If any one disagrees,
even by a single bit, the op is non-conformant and no backend is
allowed to ship.

Two crates own the complete route:

- `vyre-conform-spec`: dependency-light witness, certificate, and replay schemas.
- `vyre-conform`: production artifact execution, independent reference
  comparison, law checking, counterexample minimization, certificate issuance,
  replay, and the thin CLI.

## Invariants

1. **Witness sets are deterministic.** `WitnessSet::enumerate()`
   produces the same sequence in the same order on every run; the
   enumeration is part of the conformance contract.
2. **Law verdicts are structural.** `LawVerdict::Failed` carries the
   counterexample tuple that proved the failure: no hashing, no
   summarisation. A law failure is reproducible byte-for-byte from
   the verdict alone.
3. **Minimization converges.** `CounterexampleMinimizer` halves the
   u32 input on every step and terminates in `O(log n)` calls; it
   never loops and never returns a larger counterexample than the
   input.
4. **No backend ships without a green matrix.** CI blocks publish on
   `vyre-conform`'s matrix returning zero divergences.
5. **No exemptions.** The `UniversalDiffExemption` registry has been
   removed. Tolerance for approximate operations is encoded in the canonical
   `OperationRegistration` (for example ULP budgets for transcendental kernels).
   Every other op must match byte-for-byte or fail the matrix. There is
   no skip path for missing fixtures, capabilities, or known failures.

## Boundaries

This directory owns witness enumeration, certificate and replay schemas,
production-versus-reference proof execution, law checking, minimization, and
certificate output.

Operations remain in their semantic owner crates. Reference evaluation remains
in `vyre-reference`. Concrete target compilation and materialization remain in
driver crates. Benchmark orchestration remains in `vyre-bench`.

## Per-crate READMEs

- `vyre-conform-spec/README.md`: schema and witness contracts.
- `vyre-conform/README.md`: proof engine, replay, and CLI.

## Extension guide: adding a DataType / law / backend to conformance

1. **New DataType witness**: implement `WitnessSet` for the type in
   `vyre-conform-spec`; the enumeration order is part of the public
   contract, so pick it once and document why.
2. **New algebraic law**: add a variant to `LawVerdict` and the
   corresponding proof pass in `LawProver`; add at least three
   counterexample tuples that are known to fail for a broken op, and
   assert the prover finds them.
3. **New backend**: register the backend with `vyre-driver`, then
   add a matrix row in `vyre-conform`'s parity matrix
   fixture. The runner will diff your backend's dispatch against the
   CPU reference automatically.
4. **Tolerance contracts**: for ops whose contracts already permit
   backend-defined drift (for example `softmax` and `attention`), set the ULP
   tolerance in the canonical `OperationRegistration`. All other ops must reach
   byte-identity across every backend.

See `vyre-conform/tests/parity_matrix.rs` for the end-to-end
wiring and `vyre-conform/src/prover.rs` for the verdict
shape.

## Release evidence

Release readiness is proven through the Vyre evidence manifest and generated artifacts under `release/evidence/`. Claims here must map to concrete gate output, benchmark output, conformance output, or documentation proof files before the release requirement can be closed.

Semantic conformance is the release contract: backend parity is accepted only
when the generated conformance evidence proves the same semantic result as the
CPU reference for the claimed op/backend surface.

Concrete evidence anchors:

- `release/evidence/conformance/conformance-matrix.json`
- `release/evidence/conformance/cuda-conformance.json`
- `release/evidence/conformance/wgpu-conformance.json`
- `release/evidence/conformance/release-gate-log.json`
