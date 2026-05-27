# Per-op Surface Contract

Closes #28 A.4 per-op surface complete.

Every registered op (Tier 2 intrinsic, Tier 2.5 primitive, Tier 3
dialect) exposes eight properties. The fixture gate now fails on any
registered op whose `OpEntry` lacks either `test_inputs` or
`expected_output`; there is no documented exemption list for missing
fixtures.

## The eight properties

| Property | Purpose | Enforcement |
|---|---|---|
| **witness** | `test_inputs: fn() -> Vec<Vec<Vec<u8>>>`  -  a canonical input corpus. | CI test `every_op_has_test_fixtures_or_is_explicitly_exempt` (tightened by CONFORM M7 to require both test_inputs AND expected_output, not just either). |
| **ref** | `expected_output: fn() -> Vec<Vec<Vec<u8>>>`  -  the CPU reference computed by vyre-reference. | Same CI test. |
| **dispatch** | `build: fn() -> Program`  -  the op body itself, exposed for dispatch. | Every `inventory::submit!` call sets `build`. |
| **emit** | Naga lowering path. Program-level construction lives in `vyre-emit-naga::program`; backend crates project device/runtime facts into that shared emitter instead of carrying local Naga arms. For Tier-3 ops, the chain through Region + intrinsics proves out automatically. | CI runs `cargo test -p vyre-driver-wgpu --tests` (includes `naga_deeper_regressions.rs`). |
| **cert** | Certificate signed by `vyre-conform-runner prove`. | CI mints a fresh cert every release; CONFORM C2 seeds keys with OsRng so a certification run is non-reproducible by design. |
| **parity** | CPU↔GPU parity lens. `compare_output_buffers` checks bytewise or within-ULP. | `cat_a_gpu_differential.rs` + `lens_gpu_parity.rs`. |
| **ULP** | F32 transcendental ops declare a per-op ULP budget; `fp_parity::f32_ulp_tolerance` consults the registry (M5 tracked). | `fp_parity.rs` gate. |
| **fuzz/proptest** | Proptest generators per input type + a gap test checking the "arbitrary input in, never panic" invariant. | Per-crate `proptest!` blocks + nightly structural fuzz. |

## Worked example  -  `vyre-primitives::bitset::and::bitset_and`

- **witness:** `bitset_and_test_inputs()` emits three 256-bit input
  pairs (zero × zero, all × zero, alternating).
- **ref:** `bitset_and_expected` computes the component-wise AND on
  host for each input.
- **dispatch:** `bitset_and()` is the `fn() -> Program` builder.
- **emit:** bitset_and composes Tier 2.5 ops that emit directly;
  the Region chain in `print-composition bitset_and` terminates at
  `vyre-intrinsics::hardware::popcount_u32` via `bitset::popcount`.
- **cert:** included in every `prove` run (F-C2 close-out).
- **parity:** `cat_a_conform::bitset_and_cpu_gpu_agrees` in
  `vyre-driver-wgpu`.
- **ULP:** N/A (integer op).
- **fuzz/proptest:** `bitset::proptest::and_commutative` (landed).

## Current catalog exceptions

The generated catalogs under `docs/catalog/` are the live source of
truth. As of this pass:

- `docs/catalog/parsing.md` shows `c_lexer` and `c_keyword` with both
  witness and expected-output fixtures and byte-identity status.
- `docs/catalog/security.md` still records UniversalDiffExemption rows
  for the security dataflow family. Those rows are not fixture-gate
  entries; they are source contracts that must be removed only
  when the op registrations and convergence lens prove byte-equivalent
  execution through the shim itself.

A generated catalog that loses a witness or expected-output check is a
regression.
