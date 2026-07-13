//! GPU-IR vs CPU-ref parity for `reduce::gather` on an OUT-OF-RANGE index.
//!
//! The gather GPU IR guards the store with `if idx < count` and has NO else
//! branch (`vyre-primitives/src/reduce/indexed_move.rs`), so an out-of-range
//! index SKIPS the store, leaving `dst[lane]` at whatever it held before. The
//! CPU reference writes 0 on an out-of-range index (`src.get(idx).unwrap_or(0)`).
//! If `dst` is reused with residual data, these DIVERGE, the same class as the
//! `bitset_test_bit` unconditional-load parity bug. This pins the intended
//! contract: pre-fill `dst` with a sentinel, feed an out-of-range index, and
//! assert the GPU-IR result (via `reference_eval`) equals `cpu_ref`: the
//! skipped lane must read 0, not the sentinel.
#![cfg(all(feature = "reduce", feature = "cpu-parity"))]

use vyre_primitives::reduce::gather::{cpu_ref as gather_cpu_ref, gather as gather_fn};
use vyre_primitives::reduce::scatter::scatter as scatter_fn;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

#[test]
fn gather_out_of_range_index_matches_cpu_ref_with_reused_dst() {
    let count = 3u32;
    let src = [10u32, 20, 30];
    // lane 1 indexes 5 (>= count == out of range); lanes 0 and 2 are in range.
    let indices = [0u32, 5, 1];
    // dst reused with a nonzero sentinel, the exact condition under which a
    // skip-on-OOB GPU diverges from a write-0 CPU ref.
    let sentinel = 0xDEAD_BEEFu32;
    let dst_init = [sentinel; 3];

    let program = gather_fn("src", "indices", "dst", count);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&src)),
            Value::from(pack(&indices)),
            Value::from(pack(&dst_init)),
        ],
    )
    .expect("gather reference evaluation must succeed");

    let gpu_ir = unpack(&outputs[0].to_bytes());
    let cpu = gather_cpu_ref(&src, &indices); // expected [10, 0, 20]

    assert_eq!(
        gpu_ir, cpu,
        "gather GPU-IR must match cpu_ref on an out-of-range index: the skipped OOB lane \
         must read 0, not the reused-dst sentinel {sentinel:#x}. GPU={gpu_ir:?} cpu={cpu:?}"
    );
}

/// Structural lock for the fix. With the explicit `else { dst[lane]=0 }` in place, the
/// behavioral test above reads 0 at the OOB lane regardless of whether the runtime
/// happens to zero-init output buffers, so it documents the contract but cannot by
/// itself catch a regression that drops the else branch (if the runtime DID zero-init,
/// the behavioral test would still see 0). This test locks the fix independently of any
/// runtime zeroing assumption: gather emits an explicit OOB `store 0` that scatter (which
/// correctly skips on OOB) does not, so gather's program has strictly more nodes.
/// Removing gather's else branch collapses that delta and fails here.
#[test]
fn gather_emits_explicit_oob_store_scatter_does_not() {
    let count = 8u32;
    let gather_nodes = gather_fn("src", "indices", "dst", count).stats().node_count;
    let scatter_nodes = scatter_fn("src", "indices", "dst", count)
        .stats()
        .node_count;
    assert!(
        gather_nodes > scatter_nodes,
        "gather must emit strictly more nodes than scatter (the explicit OOB `store 0` that \
         scatter correctly omits): gather={gather_nodes} scatter={scatter_nodes}. If not greater, \
         the gather else branch was dropped and out-of-range lanes are again left to dst's prior \
         contents (the bitset_test_bit divergence class)."
    );
}
