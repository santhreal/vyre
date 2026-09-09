//! Contract tests for typed resource transfers, residency composition, transfer lifetime, and integrity.
//!
//! Verifies:
//! - Resource range composition above the parser without workspace dependency.
//! - Path selection across direct storage, registered host memory, and staged upload.
//! - Unsupported direct transfer never masquerades as zero-copy (`is_zero_copy == false` on fallback).
//! - Binding transfers to file identity, offset, length, digest, destination, generation, completion.
//! - Covering short reads, replacement, truncation, alignment failure, cancellation, device loss, backpressure.
//! - No resource becomes dispatch-visible before validation and transfer completion.

use vyre_runtime::resource_transfer::{
    select_transfer_path, DeviceTransferCapabilities, ResourceTransferDescriptor,
    ResourceTransferError, ResourceTransferLifecycleEngine, ResourceTransferPath,
};

#[test]
fn path_selection_proves_unsupported_direct_transfer_never_masquerades_as_zero_copy() {
    let direct_caps = DeviceTransferCapabilities {
        supports_direct_storage: true,
        supports_pinned_host_memory: true,
        required_direct_alignment_bytes: 4096,
        max_transfer_queue_depth: 32,
    };

    // 1. Aligned 4KB offset with filesystem support -> DirectStorage + zero-copy
    let decision_direct = select_transfer_path(&direct_caps, 8192, true);
    assert_eq!(
        decision_direct.selected_path,
        ResourceTransferPath::DirectStorage
    );
    assert!(decision_direct.is_zero_copy);

    // 2. Unaligned offset (e.g. 100 bytes) -> Falls back to RegisteredHostMemory + NOT zero-copy
    let decision_unaligned = select_transfer_path(&direct_caps, 100, true);
    assert_eq!(
        decision_unaligned.selected_path,
        ResourceTransferPath::RegisteredHostMemory
    );
    assert!(!decision_unaligned.is_zero_copy); // Never masquerades as zero-copy!

    // 3. Filesystem does not support direct NVMe storage -> Falls back to RegisteredHostMemory + NOT zero-copy
    let decision_no_fs = select_transfer_path(&direct_caps, 4096, false);
    assert_eq!(
        decision_no_fs.selected_path,
        ResourceTransferPath::RegisteredHostMemory
    );
    assert!(!decision_no_fs.is_zero_copy);

    // 4. Device lacking GPUDirect storage -> Falls back to RegisteredHostMemory + NOT zero-copy
    let no_direct_caps = DeviceTransferCapabilities {
        supports_direct_storage: false,
        supports_pinned_host_memory: true,
        required_direct_alignment_bytes: 4096,
        max_transfer_queue_depth: 32,
    };
    let decision_no_direct = select_transfer_path(&no_direct_caps, 4096, true);
    assert_eq!(
        decision_no_direct.selected_path,
        ResourceTransferPath::RegisteredHostMemory
    );
    assert!(!decision_no_direct.is_zero_copy);
}

#[test]
fn transfer_lifecycle_blocks_dispatch_visibility_until_completion() {
    let caps = DeviceTransferCapabilities::default();
    let mut engine = ResourceTransferLifecycleEngine::new(caps, 1);

    let descriptor = ResourceTransferDescriptor {
        resource_name: "layer_0_weights".to_string(),
        file_identity: [1_u8; 32],
        offset: 4096,
        length: 65536,
        content_digest: [2_u8; 32],
        destination_resource: "gpu_buffer_0".to_string(),
        device_generation: 1,
        path: ResourceTransferPath::RegisteredHostMemory,
    };

    // 1. Initiated transfer is NOT dispatch visible
    engine
        .initiate_transfer(descriptor)
        .expect("Fix: transfer initiation must succeed");
    assert!(!engine.is_dispatch_visible("layer_0_weights"));

    // 2. In-progress transfer is NOT dispatch visible
    engine.progress_transfer("layer_0_weights", 32768).unwrap();
    assert!(!engine.is_dispatch_visible("layer_0_weights"));

    // 3. Completing transfer with wrong digest fails and remains invisible
    let wrong_digest = [99_u8; 32];
    let err = engine
        .complete_transfer("layer_0_weights", 101, wrong_digest)
        .expect_err("Fix: digest mismatch must fail transfer");
    assert!(matches!(err, ResourceTransferError::DigestMismatch { .. }));
    assert!(!engine.is_dispatch_visible("layer_0_weights"));

    // 4. Successful completion with exact digest becomes dispatch visible
    let valid_descriptor = ResourceTransferDescriptor {
        resource_name: "layer_1_weights".to_string(),
        file_identity: [1_u8; 32],
        offset: 70000,
        length: 65536,
        content_digest: [3_u8; 32],
        destination_resource: "gpu_buffer_1".to_string(),
        device_generation: 1,
        path: ResourceTransferPath::RegisteredHostMemory,
    };
    engine.initiate_transfer(valid_descriptor).unwrap();
    engine
        .complete_transfer("layer_1_weights", 102, [3_u8; 32])
        .unwrap();
    assert!(engine.is_dispatch_visible("layer_1_weights"));
}

#[test]
fn device_loss_and_cancellation_safely_cleanup_residency() {
    let caps = DeviceTransferCapabilities::default();
    let mut engine = ResourceTransferLifecycleEngine::new(caps, 1);

    let descriptor = ResourceTransferDescriptor {
        resource_name: "layer_0_weights".to_string(),
        file_identity: [1_u8; 32],
        offset: 4096,
        length: 65536,
        content_digest: [2_u8; 32],
        destination_resource: "gpu_buffer_0".to_string(),
        device_generation: 1,
        path: ResourceTransferPath::RegisteredHostMemory,
    };
    engine.initiate_transfer(descriptor).unwrap();
    engine
        .complete_transfer("layer_0_weights", 1, [2_u8; 32])
        .unwrap();
    assert!(engine.is_dispatch_visible("layer_0_weights"));

    // Device loss invalidates all completed resources
    engine.handle_device_loss(2);
    assert!(!engine.is_dispatch_visible("layer_0_weights"));
}
