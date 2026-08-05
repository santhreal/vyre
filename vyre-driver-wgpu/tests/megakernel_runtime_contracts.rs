//! Runtime-owned megakernel lifecycle exercised through the WGPU backend.
//!
//! Covers valid and invalid launch geometry, exact protocol boundaries,
//! malformed ring rejection before backend submission, and the prohibition on
//! silent host fallback.

use std::sync::Arc;
use vyre_runtime::megakernel::{protocol, Megakernel};
use vyre_runtime::PipelineError;

use vyre_driver_wgpu::WgpuBackend;


fn assert_no_cpu_fallback_wording(err: &PipelineError) {
    let msg = err.to_string().to_lowercase();
    assert!(!msg.contains("cpu"), "error must never mention CPU: {msg}");
    assert!(
        !msg.contains("fallback"),
        "error must never mention fallback: {msg}"
    );
    assert!(
        !msg.contains("software"),
        "error must never imply software emulation: {msg}"
    );
}

fn require_backend() -> WgpuBackend {
    WgpuBackend::new().expect(
        "Fix: WGPU backend required for megakernel dispatcher sizing contracts; missing GPU is a configuration bug.",
    )
}

// ---------------------------------------------------------------------------
// 1. Worker group sizing invariants
// ---------------------------------------------------------------------------

#[test]
fn bootstrap_sharded_rejects_zero_slot_count() {
    let result = Megakernel::bootstrap_sharded(Arc::new(require_backend()), 0, 256, vec![]);
    let Err(err) = result else {
        panic!("zero slot count must fail")
    };
    assert!(matches!(err, PipelineError::QueueFull { .. }));
    assert_no_cpu_fallback_wording(&err);
}

#[test]
fn bootstrap_sharded_rejects_zero_workgroup_size() {
    let result = Megakernel::bootstrap_sharded(Arc::new(require_backend()), 256, 0, vec![]);
    let Err(err) = result else {
        panic!("zero workgroup size must fail")
    };
    assert!(matches!(err, PipelineError::QueueFull { .. }));
    assert_no_cpu_fallback_wording(&err);
}

#[test]
fn bootstrap_sharded_rejects_non_multiple_slot_count() {
    let result = Megakernel::bootstrap_sharded(Arc::new(require_backend()), 257, 256, vec![]);
    let Err(err) = result else {
        panic!("non-multiple slot count must fail")
    };
    assert!(matches!(err, PipelineError::QueueFull { .. }));
}

#[test]
fn worker_groups_computed_from_bootstrap_geometry() {
    assert_eq!(
        Megakernel::worker_groups_for_geometry(512, 64).expect("valid geometry must divide"),
        8,
        "worker_groups must be slot_count / workgroup_size_x"
    );
}

// ---------------------------------------------------------------------------
// 2. Publish/batch capacity checks
// ---------------------------------------------------------------------------

#[test]
fn batch_publish_empty_items_consumes_only_fence_slot() {
    let mut ring = Megakernel::encode_empty_ring(4).unwrap();
    let items: &[(u32, &[u32])] = &[];
    let consumed = Megakernel::batch_publish(&mut ring, 0, 0, items, 0xABCD).unwrap();
    assert_eq!(
        consumed, 1,
        "empty batch must consume exactly the fence slot"
    );
    let fence_op = u32::from_le_bytes(ring[4..8].try_into().unwrap());
    let fence_tag = u32::from_le_bytes(ring[20..24].try_into().unwrap());
    assert_eq!(fence_op, protocol::opcode::BATCH_FENCE);
    assert_eq!(fence_tag, 0xABCD);
}

#[test]
fn batch_publish_empty_items_fence_item_count_is_zero() {
    let mut ring = Megakernel::encode_empty_ring(2).unwrap();
    let items: &[(u32, &[u32])] = &[];
    Megakernel::batch_publish(&mut ring, 0, 0, items, 0).unwrap();
    let item_count = u32::from_le_bytes(
        ring[(protocol::ARG0_WORD as usize) * 4..(protocol::ARG0_WORD as usize) * 4 + 4]
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        item_count, 0,
        "fence item_count must be zero for empty batch"
    );
}

#[test]
fn malformed_ring_protocol_is_rejected_before_backend_submission() {
    let mut malformed_ring = vec![0_u8; (protocol::SLOT_WORDS as usize * 4) - 1];
    let error = Megakernel::publish_slot(
        &mut malformed_ring,
        0,
        0,
        protocol::opcode::NOP,
        &[],
    )
    .expect_err("one-byte-short ring record must be rejected");

    assert!(matches!(error, PipelineError::QueueFull { .. }));
    let message = error.to_string();
    assert!(
        message.contains("Fix:") && message.to_ascii_lowercase().contains("slot"),
        "malformed protocol error must identify the slot shape and remediation: {message}"
    );
}

// ---------------------------------------------------------------------------
// 3. No silent CPU fallback assumptions
// ---------------------------------------------------------------------------

#[test]
fn bootstrap_errors_contain_no_cpu_fallback_wording() {
    let backend = require_backend();
    for (slots, wg_size, desc) in [(0, 256, "zero slots"), (256, 0, "zero workgroup")] {
        let result =
            Megakernel::bootstrap_sharded(Arc::new(backend.clone()), slots, wg_size, vec![]);
        let Err(err) = result else {
            panic!("{desc} must fail")
        };
        assert_no_cpu_fallback_wording(&err);
    }
}
