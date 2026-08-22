//! Adapter/device limits honesty: every reported limit must originate from
//! the live wgpu adapter or device, never from a hardcoded constant.
//!
//! Guarantees:
//! - `max_workgroup_size` is the adapter's workgroup limit, never wider, and the
//!   width it reports is a block the device runs
//! - `max_compute_workgroups_per_dimension` matches the live device limits
//! - `max_compute_invocations_per_workgroup` is the live device limit, never
//!   more, and agrees with the profile a compile validates against
//! - `max_storage_buffer_bytes` matches the live device limits
//! - `device_limits()` is the actual `wgpu::Limits` of the created device

#![cfg(feature = "device-tests")]

mod harness;
use harness::{selected_adapter, shared_live_backend as live_backend};

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::{DispatchConfig, VyreBackend};

// ------------------------------------------------------------------
// 1. max_workgroup_size never exceeds the adapter, and the width it reports
//    is a block the device runs
// ------------------------------------------------------------------

/// One word per invocation across a single workgroup of `width` invocations.
/// Every lane stores its own index, so the returned bytes account for the whole
/// declared block rather than a prefix of it.
fn full_block_program(width: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(width)
            .with_output_byte_range(0..(width as usize * 4))],
        [width, 1, 1],
        vec![Node::store("out", Expr::gid_x(), Expr::gid_x())],
    )
}

/// WHY: closes the class "a reported workgroup limit is wider than what the
/// backend can run". The report is the adapter limit clamped to the dialect
/// envelope, so equality with the raw adapter number is not the contract: a
/// backend advertising the raw limit hands a composition a block its own dialect
/// rejects. Both edges hold instead: no axis exceeds the adapter, and the width
/// the backend publishes dispatches on the device and runs every invocation.
///
/// What it does not catch: a report narrower than the envelope for no reason,
/// which is legal and costs only occupancy.
#[test]
fn max_workgroup_size_is_the_adapter_limit_the_backend_can_run() {
    let backend = live_backend();
    let adapter = selected_adapter(&backend);
    let limits = adapter.limits();
    let info = backend.adapter_info();

    let reported = backend.max_workgroup_size();
    let adapter_limits = [
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroup_size_z,
    ];

    for (axis, (got, limit)) in reported.iter().zip(adapter_limits).enumerate() {
        assert!(
            *got >= 1 && *got <= limit,
            "Fix: max_workgroup_size axis {axis} reports {got}, outside the 1..={limit} \
             the adapter admits. Adapter: {}",
            info.name
        );
    }

    let width = reported[0];
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&full_block_program(width), &[], &config)
        .expect("Fix: the workgroup width the backend reports must dispatch.");
    let expected: Vec<u8> = (0..width).flat_map(u32::to_le_bytes).collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0], expected,
        "Fix: one workgroup of the reported width must run every invocation. Adapter: {}",
        info.name
    );
}

// ------------------------------------------------------------------
// 2. max_compute_workgroups_per_dimension must come from device limits
// ------------------------------------------------------------------

#[test]
fn max_compute_workgroups_per_dimension_matches_device_limits() {
    let backend = live_backend();
    let device_limits = backend.device_limits();
    let info = backend.adapter_info();

    let reported = backend.max_compute_workgroups_per_dimension();
    let expected = device_limits.max_compute_workgroups_per_dimension;

    assert_eq!(
        reported, expected,
        "Fix: max_compute_workgroups_per_dimension must match live device limits. \
         Got {}, expected {}. Adapter: {}",
        reported, expected, info.name
    );
}

// ------------------------------------------------------------------
// 3. max_compute_invocations_per_workgroup never exceeds the device, and
//    every report of it agrees
// ------------------------------------------------------------------

/// WHY: closes the class "the invocation budget a caller reads is not the budget
/// a compile validates against". The number is the device limit clamped to the
/// dialect envelope, so three edges are the contract: it never exceeds the live
/// limit, the 1D block the backend advertises fits inside it, and the device
/// profile a target compile checks carries the same number as the accessor a
/// caller reads. The adapter-probed and device-probed paths clamp separately,
/// and a disagreement between them refuses a program at compile time for a
/// width the backend published.
///
/// What it does not catch: a clamp wrong in the same way in both paths.
#[test]
fn max_compute_invocations_per_workgroup_is_the_device_limit_the_backend_can_run() {
    let backend = live_backend();
    let device_limits = backend.device_limits();
    let info = backend.adapter_info();

    let reported = backend.max_compute_invocations_per_workgroup();
    let live = device_limits.max_compute_invocations_per_workgroup;

    assert!(
        reported >= 1 && reported <= live,
        "Fix: max_compute_invocations_per_workgroup reports {reported}, outside the \
         1..={live} the device admits. Adapter: {}",
        info.name
    );

    let block = backend.max_workgroup_size();
    assert!(
        block[0] <= reported,
        "Fix: the backend advertises a {}-wide 1D block and only {reported} invocations \
         per workgroup, so the block it publishes cannot be declared. Adapter: {}",
        block[0],
        info.name
    );

    assert_eq!(
        backend.device_profile().max_invocations_per_workgroup,
        reported,
        "Fix: the device profile a compile validates against must carry the invocation \
         budget the backend reports. Adapter: {}",
        info.name
    );
}

// ------------------------------------------------------------------
// 4. max_storage_buffer_bytes must come from device limits
// ------------------------------------------------------------------

#[test]
fn max_storage_buffer_bytes_matches_device_limits() {
    let backend = live_backend();
    let device_limits = backend.device_limits();
    let info = backend.adapter_info();

    let reported = backend.max_storage_buffer_bytes();
    let expected = u64::from(device_limits.max_storage_buffer_binding_size);

    assert_eq!(
        reported, expected,
        "Fix: max_storage_buffer_bytes must match live device limits. \
         Got {}, expected {}. Adapter: {}",
        reported, expected, info.name
    );
}

// ------------------------------------------------------------------
// 5. device_limits() must be the actual wgpu device limits object
// ------------------------------------------------------------------

#[test]
fn device_limits_is_actual_wgpu_limits() {
    let backend = live_backend();
    let device_limits = backend.device_limits();
    let adapter = selected_adapter(&backend);
    let adapter_limits = adapter.limits();
    let info = backend.adapter_info();

    // The device limits should be at least as high as the adapter limits
    // for the fields we request explicitly at device creation.
    assert!(
        device_limits.max_compute_workgroup_size_x >= adapter_limits.max_compute_workgroup_size_x,
        "Fix: device max_compute_workgroup_size_x must not be lower than adapter limit. \
         Got {}, adapter has {}. Adapter: {}",
        device_limits.max_compute_workgroup_size_x,
        adapter_limits.max_compute_workgroup_size_x,
        info.name
    );

    assert!(
        device_limits.max_compute_workgroup_size_y >= adapter_limits.max_compute_workgroup_size_y,
        "Fix: device max_compute_workgroup_size_y must not be lower than adapter limit. \
         Got {}, adapter has {}. Adapter: {}",
        device_limits.max_compute_workgroup_size_y,
        adapter_limits.max_compute_workgroup_size_y,
        info.name
    );

    assert!(
        device_limits.max_compute_workgroup_size_z >= adapter_limits.max_compute_workgroup_size_z,
        "Fix: device max_compute_workgroup_size_z must not be lower than adapter limit. \
         Got {}, adapter has {}. Adapter: {}",
        device_limits.max_compute_workgroup_size_z,
        adapter_limits.max_compute_workgroup_size_z,
        info.name
    );
}
