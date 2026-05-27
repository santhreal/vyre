# Audit: surgec Arbitrary-Compute  -  Release Bar

**Date:** 2026-04-23  
**Scope:** Everything on the arbitrary-compute path of surgec.  
**Goal:** list every obstacle between where the code is today and
"release"  -  the bar where a user can sit down, write a SURGE
file expressing any data-parallel computation over byte buffers,
and have surgec dispatch it correctly + efficiently + reproducibly
across Vulkan / Metal / DX12 / SPIR-V.

---

## Release acceptance bar

An arbitrary-compute pipeline is release when **every** item on
this list is true:

1. **A single command**  -  `surgec run program.surge --input
   a=a.bin --output b=b.bin`  -  compiles, dispatches, and streams
   output bytes to disk without a line of surrounding glue.
2. **Every SURGE Expr / Node variant** lowers to a valid naga
   module on every shipped backend (wgpu, spirv, … future Metal /
   DX12 / PTX).
3. **Byte-equivalent output** between CPU reference and GPU
   dispatch for every integer op. ULP-bounded equivalence for
   every float op, per the transcendental ULP budget.
4. **Cross-machine fingerprint stability**  -  same program on
   different hosts produces the same `PipelineFingerprint` hex so
   the distributed pipeline cache works.
5. **Byte-equivalent output** whether dispatched via
   `scan_gpu_with_context` (scanner shape) or `run_program`
   (arbitrary shape). The run path is a subset of the scan path,
   not a parallel pipeline.
6. **Signed conformance certificate** attests the output for a
   published corpus. Verifier passes both hash-chain and Ed25519
   signature.
7. **Every error carries a `Fix:` hint** and a stable code.
   `surgec run` failures name the problematic Program location.
8. **No silent correctness shortcuts.** Every op either lowers
   correctly or rejects with a named reason. Zero "works by
   accident" paths.
9. **Deterministic compile**  -  same `.surge` input produces the
   same compiled Program bytes across runs (modulo timestamps).
10. **≥1000× vs competitor CPU baseline** on the benchmark
    matrix, per the GATE_CLOSURE.md G4 cell definitions.
11. **First-party docs**  -  a newcomer opens `ARBITRARY_COMPUTE.md`
    + `AUTHORING.md` + `BENCHMARK.md` and can ship their first
    compute kernel end-to-end in ≤ 30 minutes.

## Findings against the bar

### L1  -  CRITICAL | the `surgec run` CLI verb ships the Rust API only, no CLI

**File:** `libs/tools/surgec/src/lib.rs:runtime_paths.rs` (no CLI wire-up).

The Rust-level `surgec::run_program` foundation shipped this
cycle (VISION V4 Rust half, task #231). The CLI verb  -  the *one*
surface that unlocks bar item #1  -  does not exist. A user today
cannot invoke arbitrary compute without writing Rust.

**Fix:** add `Run(args)` to the CLI enum; wire it to `run_program`
via argument parsing for `--input KEY=PATH` + `--output KEY=PATH`
+ `--workgroup-size N`. One file diff in `surgec/src/bin/`.

**Test hint:** `surgec run <crate_dir>/tests/fixtures/gemv.surge --input a=a.bin --input b=b.bin --output c=c.bin`
produces the bytes that `tests/run_arbitrary.rs` already
verifies Rust-side.

---

### L2  -  HIGH | SURGE lang has no dispatch-geometry annotation

A `.surge` file today has no way to declare its intended
`workgroup_size` or `dispatch_count`. `run_program` accepts those
from the caller; the CLI will rely on caller-supplied defaults.
Release means the `.surge` can declare its own geometry so a
downstream author doesn't have to pick numbers.

**Fix:** add `program { workgroup_size = [128, 1, 1]; dispatch = "..." }` preamble to the SURGE grammar (#41 C.1 follow-up), or `#[workgroup_size(128)]` attribute syntax. Lowering reads the annotation into the emitted `Program::wrapped(..., workgroup_size, ...)`.

---

### L3  -  HIGH | SURGE lang has no output-schema declaration

The user hands `run_program` raw byte buffers. Release means the
`.surge` file declares the shape + dtype + count of each output so
the CLI can emit a typed `.json` or `.npy` artifact instead of
bytes. Scanner rules sidestep this by emitting Findings; arbitrary
compute needs it.

**Fix:** add `output { name: "c"; dtype: F32; shape: [32, 32]; count: 1024 }` blocks (C.1 follow-up). The CLI emits typed outputs in multiple formats (raw bytes, JSON, npy) based on `--format`.

---

### L4  -  HIGH | Cross-machine fingerprint stability is claimed but not tested

RUNTIME Finding 1 fixed buffer-declaration-order instability. No
test pins cross-machine stability explicitly.

**Fix:** add `vyre-runtime/tests/fingerprint_cross_host.rs` that
snapshot-tests the fingerprint hex for a curated set of programs.
A bump in the hex column is a legitimate fingerprint contract
change  -  the test forces that bump to be visible and intentional.

---

### L5  -  HIGH | Run-path / scan-path byte-equivalence is untested

Item #5 on the release bar. `run_arbitrary.rs` proves the run
path works; no test proves that a program dispatched via both
paths produces the same bytes.

**Fix:** `tests/run_scan_equivalence.rs`  -  hand-build a program
with a synthetic scan output schema, dispatch via both paths,
assert byte-equivalent outputs (ignoring the scan envelope).

---

### L6  -  HIGH | Deterministic compile unproven

Item #9. SURGE compiles twice today without a hash check between
runs.

**Fix:** `tests/compile_determinism.rs`  -  compile a fixed `.surge`
100× and assert the hash of every emitted `Program.to_wire()` is
identical.

---

### L7  -  MEDIUM | ≥1000× gate coverage is per-cell but documented thresholds are missing

`BENCHMARK.md` describes the matrix. `benches/thresholds.toml`
doesn't exist yet  -  the post-process script that blocks merge on a
missed cell requires it.

**Fix:** land `libs/tools/surgec/benches/thresholds.toml` with
placeholder values derived from a dry-run baseline; populate the
real numbers on the first full certification run (B.5).

---

### L8  -  MEDIUM | Every error code is documented per-crate but the codebase doesn't enforce "Fix:" prefix at compile-time

`santh-error` enforces it via typestate on its builder. Other
crates rely on runtime `debug_assert`. A cargo-xtask lint grepping
every `Display` / `Error::message` implementation is the only
static enforcement today (partial implementation under
`cargo xtask gate1`).

**Fix:** ship the full grep check; fail the gate on any public
error without `Fix:`.

---

### L9  -  MEDIUM | the `AUTHORING.md` 30-minute onboarding claim is unverified

Claim #11 needs a hand-timed walkthrough (a new engineer or a
fresh agent session runs the docs, times the wall clock, lands
their first compute kernel).

**Fix:** schedule the walkthrough as part of the release pre-
flight (RELEASE_ENGINEERING.md step 10). Log the time + any
friction surfaced as follow-up UX findings in `UX_SWEEP.md`.

---

### L10  -  MEDIUM | No first-class examples directory

A user who finds surgec via `cargo search` sees the README but
has no `examples/` directory to read. Every other tool in the
arbitrary-compute adjacency (candle, burn, tensorflow-rs) ships
examples.

**Fix:** `examples/gemv`, `examples/fixpoint`, `examples/sha3`,
`examples/wave-sim`  -  each a self-contained Rust example using
`surgec::run_program` with a bundled `.surge` + a CPU reference
comparison.

## Summary

| ID | Severity | Domain | Tracked |
|---|---|---|---|
| L1 | CRITICAL | CLI verb missing | new task |
| L2 | HIGH | dispatch geometry in SURGE | new task |
| L3 | HIGH | output schema in SURGE | new task |
| L4 | HIGH | cross-machine fingerprint test | new task |
| L5 | HIGH | run/scan byte-equivalence test | new task |
| L6 | HIGH | compile determinism test | new task |
| L7 | MEDIUM | thresholds.toml | under #39 B.5 |
| L8 | MEDIUM | Fix: prefix grep gate | under #32 A.8 follow-up |
| L9 | MEDIUM | 30-min onboarding walkthrough | under RELEASE_ENGINEERING |
| L10 | MEDIUM | examples/ directory | new task |

These are the concrete obstacles between today's arbitrary-compute
story and the release bar. Closing L1 + L2 + L3 + L4 + L5 + L6
is the definition of "done" for the arbitrary-compute pipeline.
