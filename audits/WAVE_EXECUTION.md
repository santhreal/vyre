# WYRE performance inventory  -  execution waves

Companion: [`VYRE_PERFORMANCE_ARCHITECTURE_INVENTORY_2026-04-28.md`](./VYRE_PERFORMANCE_ARCHITECTURE_INVENTORY_2026-04-28.md).

## Phases (exit criteria)

**Phase 1  -  Spec (tests, gates, baselines)**
Executable contracts cover inventory themes; every new contract maps to one or more inventory IDs in file headers. No “ship later” tests: either the check runs in CI/release, or the script documents why it is opt-in (GPU-only).

**Phase 2  -  Implementation**
Code changes close `open` rows; `cargo xtask perf-inventory-wave1` and the wave-1 test targets stay green. Order work by **Highest Leverage** in the inventory (dispatch/readback → clone removal → cache → decoupling → …).

**Phase 3  -  Safety (Santh bar)**
`scripts/check_release_signoff.sh`, public API snapshots, gap/failure tests, cross-backend spot checks, and doc-claim gates. No relaxing Phase 1 budgets without inventory text change.

## Wave 1.1 (Phase 1)  -  in tree

| Artifact | Role |
|----------|------|
| `vyre-driver-wgpu/tests/dispatch_allocation_contract.rs` | P0 #10  -  steady-state CPU heap bounds (direct, compiled, async) |
| `vyre-foundation/tests/optimizer_reference_parity_smoke.rs` | P0 #34 seed  -  optimized vs unoptimized reference agreement |
| `scripts/check_performance_inventory_wave1.sh` | One command: wave-1 tests + signoff that scripts exist |
| `cargo xtask perf-inventory-wave1` | Same, from repo root via xtask |

## Commands

```sh
cd libs/performance/matching/vyre
bash scripts/check_performance_inventory_wave1.sh
# or
cargo xtask perf-inventory-wave1
```

GPU-required tests use the same “fail loud” policy as `tests/dispatch_hot_path.rs` (`WgpuBackend::acquire()`).

## Later waves (Phase 1 continuation)

- **1.2**  -  Pipeline cache: byte budget + LRU contract tests; disk/remote poisoning (P0 #16–23).
- **1.3**  -  `no Program::clone` / pass counters in optimizer hot paths (P0 #28–31) as `xtask` or grep gates with allowlists.
- **1.4**  -  P1 gate scripts for inventory #100–112, integrated into `check_release_signoff.sh` as each lands.

## Phase 2 handoff

When Wave 1.1 is green, Phase 2 agents take the **inventory ID list** in each test header and implement until budgets tighten and `open` rows flip. Phase 3 runs only after Phase 2 PRs for a wave are merged and release is re-run on the result.
