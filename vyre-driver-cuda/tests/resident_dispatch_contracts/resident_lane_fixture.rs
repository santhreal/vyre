//! Shared setup for the CUDA resident dispatch contract family.
//!
//! Every contract here runs the same shapes: a four-lane u32 resident buffer
//! seeded `[1, 2, 3, 4]`, an elementwise add or multiply over one wrapped
//! workgroup, and a readback expectation that depends on whether the borrowed
//! host-buffer fallback is opted in. Each contract file carried its own copy,
//! so the lane width, the seed and the fallback rule each had four owners. A
//! change made in three of them would have left the fourth asserting a stale
//! expectation while still reporting green.

use super::*;

/// Lanes in every resident buffer this family allocates.
pub(super) const LANES: u32 = 4;

/// Byte length of a `LANES`-wide u32 resident buffer.
pub(super) const LANE_BYTES: usize = LANES as usize * 4;

/// Input lanes every contract uploads before dispatching.
pub(super) const SEED: [u32; LANES as usize] = [1, 2, 3, 4];

/// `SEED` in device byte order.
pub(super) fn seed_bytes() -> Vec<u8> {
    u32_bytes(&SEED)
}

/// One wrapped-workgroup program writing `value` into every `dst` lane.
fn lane_program(src: &str, dst: &str, value: Expr) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read(src, 0, DataType::U32).with_count(LANES),
            BufferDecl::output(dst, 1, DataType::U32).with_count(LANES),
        ],
        [1, 1, 1],
        vec![Node::store(dst, Expr::gid_x(), value)],
    )
}

/// `dst[i] = src[i] + addend`.
pub(super) fn add_program(src: &str, dst: &str, addend: u32) -> Program {
    lane_program(
        src,
        dst,
        Expr::add(Expr::load(src, Expr::gid_x()), Expr::u32(addend)),
    )
}

/// `dst[i] = src[i] * factor`.
pub(super) fn mul_program(src: &str, dst: &str, factor: u32) -> Program {
    lane_program(
        src,
        dst,
        Expr::mul(Expr::load(src, Expr::gid_x()), Expr::u32(factor)),
    )
}

/// `dst[i] = src[i]`.
pub(super) fn copy_program(src: &str, dst: &str) -> Program {
    lane_program(src, dst, Expr::load(src, Expr::gid_x()))
}

/// Whether the borrowed host-buffer fallback is both compiled in and opted into.
///
/// The fallback reads whole resident windows back through host buffers, so the
/// compact readback byte counts the native path asserts do not hold on a run
/// that enables it.
pub(super) fn borrowed_fallback_active() -> bool {
    if std::env::var_os("VYRE_CUDA_RESIDENT_BORROWED_FALLBACK").is_none() {
        return false;
    }
    #[cfg(debug_assertions)]
    {
        true
    }
    #[cfg(not(debug_assertions))]
    {
        std::env::var("VYRE_CUDA_ALLOW_BORROWED_FALLBACK")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
            .unwrap_or(false)
    }
}

/// Readback bytes to expect: the native compact count, or the whole-window
/// count the borrowed fallback transfers when it is opted in.
pub(super) fn expected_readback_bytes(native_resident: u64, fallback_resident: u64) -> u64 {
    if borrowed_fallback_active() {
        fallback_resident
    } else {
        native_resident
    }
}

/// Acquire the CUDA backend a contract runs on.
pub(super) fn acquire() -> CudaBackend {
    CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host.")
}

/// Acquire the object-safe registration wrapper the trait-seam contracts use.
pub(super) fn acquire_registration() -> CudaBackendRegistration {
    CudaBackendRegistration::new(acquire())
}

/// Allocate one `LANE_BYTES` resident buffer on the concrete backend.
pub(super) fn handle_lane(backend: &CudaBackend, role: &str) -> CudaResidentBuffer {
    backend
        .allocate_resident(LANE_BYTES)
        .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} allocation failed: {error}"))
}

/// Allocate one resident buffer and upload `seed` into it.
pub(super) fn seeded_handle_lane(
    backend: &CudaBackend,
    role: &str,
    seed: &[u32],
) -> CudaResidentBuffer {
    let handle = handle_lane(backend, role);
    backend
        .upload_resident(handle, &u32_bytes(seed))
        .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} upload failed: {error}"));
    handle
}

/// Free resident buffers allocated through [`handle_lane`].
pub(super) fn free_handle_lanes(backend: &CudaBackend, lanes: &[(CudaResidentBuffer, &str)]) {
    for (handle, role) in lanes {
        backend
            .free_resident(*handle)
            .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} free failed: {error}"));
    }
}

/// Read back `handle` as u32 lanes.
pub(super) fn download_lanes(
    backend: &CudaBackend,
    handle: CudaResidentBuffer,
    role: &str,
) -> Vec<u32> {
    let bytes = backend
        .download_resident(handle)
        .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} download failed: {error}"));
    bytes_u32(&bytes)
}

/// Allocate one `LANE_BYTES` resident resource through the object-safe seam.
pub(super) fn resource_lane(backend: &CudaBackendRegistration, role: &str) -> Resource {
    VyreBackend::allocate_resident(backend, LANE_BYTES)
        .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} allocation failed: {error}"))
}

/// Allocate one resident resource and upload [`SEED`] into it.
pub(super) fn seeded_resource_lane(backend: &CudaBackendRegistration, role: &str) -> Resource {
    let resource = resource_lane(backend, role);
    VyreBackend::upload_resident(backend, &resource, &seed_bytes())
        .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} upload failed: {error}"));
    resource
}

/// Free resident resources allocated through [`resource_lane`].
pub(super) fn free_resource_lanes(backend: &CudaBackendRegistration, lanes: Vec<(Resource, &str)>) {
    for (resource, role) in lanes {
        VyreBackend::free_resident(backend, resource)
            .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} free failed: {error}"));
    }
}

/// Allocate one `LANE_BYTES` resident buffer through the optimizer dispatcher seam.
pub(super) fn dispatcher_lane(dispatcher: &CudaProgramDispatcher<'_>, role: &str) -> u64 {
    dispatcher
        .alloc_resident(LANE_BYTES)
        .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} allocation failed: {error}"))
}

/// Free resident buffers allocated through [`dispatcher_lane`].
pub(super) fn free_dispatcher_lanes(dispatcher: &CudaProgramDispatcher<'_>, lanes: &[(u64, &str)]) {
    for (handle, role) in lanes {
        dispatcher
            .free_resident(*handle)
            .unwrap_or_else(|error| panic!("Fix: CUDA resident {role} free failed: {error}"));
    }
}

/// One optimizer-sequence step with the dispatcher's resolved launch geometry.
pub(super) fn dispatcher_step<'a>(
    program: &'a Program,
    handle_ids: &'a [u64],
) -> ResidentDispatchStep<'a> {
    ResidentDispatchStep {
        program,
        handle_ids,
        grid_override: None,
    }
}

/// One sequence step with the backend's resolved default launch geometry.
pub(super) fn step<'a>(
    program: &'a Program,
    resources: &'a [Resource],
) -> vyre_driver::ResidentDispatchStep<'a> {
    vyre_driver::ResidentDispatchStep {
        program,
        resources,
        grid_override: None,
        workgroup_override: None,
    }
}

/// One compact readback range over a resident resource.
pub(super) fn read_range(
    resource: &Resource,
    byte_offset: usize,
    byte_len: usize,
) -> vyre_driver::ResidentReadRange<'_> {
    vyre_driver::ResidentReadRange {
        resource,
        byte_offset,
        byte_len,
    }
}

/// Run `program` once over a seeded input lane and one output lane, returning
/// the output lanes and the borrowed-fallback dispatch count for that dispatch.
pub(super) fn dispatch_resident_lanes(program: &Program, seed: &[u32]) -> (Vec<u32>, u64) {
    let backend = acquire();
    let input = seeded_handle_lane(&backend, "input", seed);
    let output = handle_lane(&backend, "output");

    backend.reset_telemetry();
    backend
        .dispatch_resident(program, &[input, output], &DispatchConfig::default())
        .expect("Fix: CUDA resident dispatch must execute the scalar trainer-safe subset.");

    let lanes = download_lanes(&backend, output, "output");
    let borrowed_fallback_dispatches = backend
        .telemetry_snapshot()
        .resident_borrowed_fallback_dispatches;
    free_handle_lanes(&backend, &[(input, "input"), (output, "output")]);
    (lanes, borrowed_fallback_dispatches)
}

/// Assert the native path fused every requested range into one compact D2H copy
/// and never escaped to the borrowed host-buffer fallback.
///
/// Skipped when the borrowed fallback is opted in, because that path reads whole
/// resident windows and the compact counts below do not describe it.
pub(super) fn assert_native_compact_readback(
    telemetry: &CudaTelemetrySnapshot,
    expected_bytes: u64,
    what: &str,
) {
    if borrowed_fallback_active() {
        return;
    }
    assert_eq!(
        telemetry.readback_bytes, expected_bytes,
        "Fix: native CUDA sequence readback must {what} into one {expected_bytes}-byte D2H interval."
    );
    assert_eq!(
        telemetry.device_readback_operations, 1,
        "Fix: native CUDA sequence readback must issue one D2H operation when it {what}."
    );
    assert_eq!(
        telemetry.resident_borrowed_fallback_dispatches, 0,
        "Fix: native CUDA resident sequence readback must not touch the borrowed fallback path."
    );
}
