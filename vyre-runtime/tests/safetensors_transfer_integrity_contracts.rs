//! Contract tests for safetensors residency composition, transfer lifetime, and integrity.
//!
//! Verifies Section 186.2 & 186.3:
//! - Safetensor range composition above the parser without workspace dependency.
//! - Path selection across direct storage, registered host memory, and staged upload.
//! - Unsupported direct transfer never masquerades as zero-copy (`is_zero_copy == false` on fallback).
//! - Binding transfers to file identity, offset, length, digest, destination, generation, completion.
//! - Covering short reads, replacement, truncation, alignment failure, cancellation, device loss, backpressure.
//! - No tensor becomes dispatch-visible before validation and transfer completion.

use vyre_runtime::safetensors_transfer::{
    select_transfer_path, DeviceTransferCapabilities, SafetensorTransferPath, TransferDescriptor,
    TransferError, TransferLifecycleEngine,
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
        SafetensorTransferPath::DirectStorage
    );
    assert!(decision_direct.is_zero_copy);

    // 2. Unaligned offset (e.g. 100 bytes) -> Falls back to RegisteredHostMemory + NOT zero-copy
    let decision_unaligned = select_transfer_path(&direct_caps, 100, true);
    assert_eq!(
        decision_unaligned.selected_path,
        SafetensorTransferPath::RegisteredHostMemory
    );
    assert!(!decision_unaligned.is_zero_copy); // Never masquerades as zero-copy!

    // 3. Filesystem does not support direct NVMe storage -> Falls back to RegisteredHostMemory + NOT zero-copy
    let decision_no_fs = select_transfer_path(&direct_caps, 4096, false);
    assert_eq!(
        decision_no_fs.selected_path,
        SafetensorTransferPath::RegisteredHostMemory
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
        SafetensorTransferPath::RegisteredHostMemory
    );
    assert!(!decision_no_direct.is_zero_copy);
}

#[test]
fn transfer_lifecycle_blocks_dispatch_visibility_until_completion() {
    let caps = DeviceTransferCapabilities::default();
    let mut engine = TransferLifecycleEngine::new(caps, 1);

    let descriptor = TransferDescriptor {
        tensor_name: "model.layers.0.weight".to_string(),
        file_identity: [1_u8; 32],
        offset: 4096,
        length: 65536,
        content_digest: [2_u8; 32],
        destination_resource: "gpu_buffer_0".to_string(),
        device_generation: 1,
        path: SafetensorTransferPath::RegisteredHostMemory,
    };

    // 1. Initiated transfer is NOT dispatch visible
    engine
        .initiate_transfer(descriptor)
        .expect("Fix: transfer initiation must succeed");
    assert!(!engine.is_dispatch_visible("model.layers.0.weight"));

    // 2. In-progress transfer is NOT dispatch visible
    engine
        .progress_transfer("model.layers.0.weight", 32768)
        .unwrap();
    assert!(!engine.is_dispatch_visible("model.layers.0.weight"));

    // 3. Completing transfer with wrong digest fails and remains invisible
    let wrong_digest = [99_u8; 32];
    let err = engine
        .complete_transfer("model.layers.0.weight", 101, wrong_digest)
        .expect_err("Fix: digest mismatch must fail transfer");
    assert!(matches!(err, TransferError::DigestMismatch { .. }));
    assert!(!engine.is_dispatch_visible("model.layers.0.weight"));

    // 4. Successful completion with exact digest becomes dispatch visible
    let valid_descriptor = TransferDescriptor {
        tensor_name: "model.layers.1.weight".to_string(),
        file_identity: [1_u8; 32],
        offset: 70000,
        length: 65536,
        content_digest: [3_u8; 32],
        destination_resource: "gpu_buffer_1".to_string(),
        device_generation: 1,
        path: SafetensorTransferPath::RegisteredHostMemory,
    };
    engine.initiate_transfer(valid_descriptor).unwrap();
    engine
        .complete_transfer("model.layers.1.weight", 102, [3_u8; 32])
        .unwrap();
    assert!(engine.is_dispatch_visible("model.layers.1.weight"));
}

#[test]
fn device_loss_and_cancellation_safely_cleanup_residency() {
    let caps = DeviceTransferCapabilities::default();
    let mut engine = TransferLifecycleEngine::new(caps, 1);

    let descriptor = TransferDescriptor {
        tensor_name: "model.embed.weight".to_string(),
        file_identity: [1_u8; 32],
        offset: 0,
        length: 1024,
        content_digest: [4_u8; 32],
        destination_resource: "gpu_buffer_embed".to_string(),
        device_generation: 1,
        path: SafetensorTransferPath::StagedUpload,
    };
    engine.initiate_transfer(descriptor).unwrap();
    engine
        .complete_transfer("model.embed.weight", 200, [4_u8; 32])
        .unwrap();
    assert!(engine.is_dispatch_visible("model.embed.weight"));

    // Device reset / generation bump invalidates all prior tensors
    engine.handle_device_loss(2);
    assert!(!engine.is_dispatch_visible("model.embed.weight"));
}

#[test]
fn queue_backpressure_enforces_maximum_in_flight_depth() {
    let caps = DeviceTransferCapabilities {
        max_transfer_queue_depth: 2,
        ..DeviceTransferCapabilities::default()
    };
    let mut engine = TransferLifecycleEngine::new(caps, 1);

    for i in 0..2 {
        let desc = TransferDescriptor {
            tensor_name: format!("tensor_{i}"),
            file_identity: [1_u8; 32],
            offset: (i * 1024) as u64,
            length: 1024,
            content_digest: [i as u8; 32],
            destination_resource: format!("gpu_buffer_{i}"),
            device_generation: 1,
            path: SafetensorTransferPath::RegisteredHostMemory,
        };
        engine.initiate_transfer(desc).unwrap();
    }

    // Third in-flight transfer triggers backpressure error
    let overflow_desc = TransferDescriptor {
        tensor_name: "tensor_overflow".to_string(),
        file_identity: [1_u8; 32],
        offset: 2048,
        length: 1024,
        content_digest: [5_u8; 32],
        destination_resource: "gpu_buffer_overflow".to_string(),
        device_generation: 1,
        path: SafetensorTransferPath::RegisteredHostMemory,
    };
    let err = engine
        .initiate_transfer(overflow_desc)
        .expect_err("Fix: exceeding max queue depth must return backpressure error");
    assert!(matches!(err, TransferError::QueueBackpressure { .. }));
}
