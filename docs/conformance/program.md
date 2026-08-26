# Conformance

If WGSL emits `7.0`, SPIR-V emits `7.0` and `vyre-reference` emits `7.0`
for the same program, the operation conforms. If any one disagrees by a
single bit, the operation is non-conformant and no backend ships.

Two crates own the route. `vyre-conform-spec` holds the dependency-light
witness, certificate and replay schemas. `vyre-conform` runs production
artifacts, compares against the independent reference, checks laws,
minimizes counterexamples, issues certificates, replays them, and carries
the CLI.

## Five invariants

**Witness sets are deterministic.** `WitnessSet::enumerate()` produces the
same sequence in the same order on every run. The enumeration order is part
of the conformance contract, not an implementation detail.

**Law verdicts are structural.** `LawVerdict::Failed` carries the
counterexample tuple that proved the failure. No hashing and no summary: a
law failure reproduces byte for byte from the verdict alone.

**Minimization converges.** `CounterexampleMinimizer` halves the `u32`
input on every step and terminates in `O(log n)` calls. It never loops and
never returns a counterexample larger than its input.

**No backend ships without a green matrix.** Publish is blocked on
`vyre-conform`'s matrix returning zero divergences.

**No exemptions.** There is no exemption registry. Tolerance for an
approximate operation is encoded in its canonical `OperationRegistration`,
as a ULP budget the operation owns. Every other operation matches byte for
byte or fails the matrix. There is no skip path for a missing fixture, a
missing capability or a known failure.

## Adding to the matrix

**A new `DataType`** needs a `WitnessSet` implementation in
`vyre-conform-spec`. The enumeration order is public contract, so choose it
once and record why.

**A new algebraic law** needs a `LawVerdict` variant and its proof pass in
`LawProver`, plus at least three counterexample tuples known to fail for a
broken operation, with an assertion that the prover finds them.

**A new backend** registers with `vyre-driver` and adds a matrix row in
`vyre-conform`'s parity matrix fixture. The runner then diffs that
backend's dispatch against the reference without further wiring.

**A tolerance contract** is set on the canonical `OperationRegistration`,
for an operation whose contract already permits backend-defined drift. Every
other operation reaches byte identity across every backend.

`conform/vyre-conform/tests/parity_matrix.rs` is the end-to-end wiring.
`conform/vyre-conform/src/prover.rs` is the verdict shape.

## How a production result is obtained

`ProductionSession` holds an `Arc<dyn SemanticExecutor>`, a
`SemanticExecutionPolicy` and the schedule-free `Program` under proof. Every
submission crosses `SemanticExecutor::execute`: the executor compiles the
program, admits its target payload, submits the frozen entry geometry the
admitted artifact carries, and returns the artifact and payload identities
alongside the output bytes. The session states no grid and no workgroup, so a
conformance run and a release run submit the same bytes.

`RegisteredSemanticExecutor` binds a session to one `BackendRegistration`. The
policy takes its target facts from that registration's acquired device, an
external-facts digest, `CompileObjective::MinimizeLatency`, the conformance
search budget and an artifact ceiling. A backend row in the matrix therefore
differs from another only in the device it acquires.

## Boundaries

Conformance owns witness enumeration, the certificate and replay schemas,
production-versus-reference proof execution, law checking, minimization and
certificate output.

It does not own operations, which stay in their semantic owner crates. It
does not own reference evaluation, which stays in `vyre-reference`. It does
not own target compilation or materialization, which stay in driver crates.
It does not own benchmark orchestration, which stays in `vyre-bench`.

## Evidence

Semantic conformance is the release contract. Backend parity is accepted
only when generated conformance evidence proves the same semantic result as
the reference for the claimed operation and backend surface. The anchors:

- `release/evidence/conformance/conformance-matrix.json`
- `release/evidence/conformance/cuda-conformance.json`
- `release/evidence/conformance/wgpu-conformance.json`
- `release/evidence/conformance/release-gate-log.json`
