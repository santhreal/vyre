use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};
use std::time::Duration;
use vyre_driver::BackendError;

type Result<T, E = BackendError> = std::result::Result<T, E>;

use super::features::{enabled_features_for_adapter, subgroup_smoke_compiles, EnabledFeatures};
use super::reserve_probe_vec;
use super::selector::gpu_candidate_score;

/// Excludes concurrent Vulkan loader startup across the process.
///
/// A `wgpu::Instance` starts the Vulkan loader, and loader startup is not
/// reentrant. While one thread is inside `vkCreateInstance` negotiating an ICD,
/// the loader dispatch table is half written, and a second thread entering
/// `vkEnumerateInstanceExtensionProperties` at that moment calls through a null
/// function pointer and the process dies with SIGSEGV.
static LOADER_STARTUP: Mutex<()> = Mutex::new(());

/// A resolve on a created device already waited for its submission, so the
/// mapping callback is a driver hand-off rather than device work. This bounds a
/// driver that never delivers it, so device acquisition cannot hang on the
/// capability probe.
const TIMESTAMP_PROBE_READBACK_TIMEOUT: Duration = Duration::from_secs(10);

/// Take the loader lock, or end the process.
///
/// Poison here is the exact state the lock excludes. A thread that panicked
/// inside `vkCreateInstance` left the loader dispatch table half written,
/// `PoisonError::into_inner` hands that table to the next caller, and what the
/// caller gets is the SIGSEGV the comment above describes, in an ICD frame that
/// names no vyre code. Loader state is also not this process's to rebuild.
fn loader_startup() -> MutexGuard<'static, ()> {
    match vyre_driver::lock_policy::govern_mutex(
        &LOADER_STARTUP,
        "the wgpu device factory",
        "the graphics loader dispatch table",
        vyre_driver::lock_policy::RecoveryClass::ProcessFatal,
    ) {
        Ok(guard) => guard,
        Err(_) => unreachable!(),
    }
}

/// Backends this driver dispatches compute through.
///
/// Every backend here runs a compute pipeline. `Backends::all()` additionally
/// carries GL, which reaches the same physical device a Vulkan or Metal adapter
/// already reports, adds no compute capability this driver lowers to, and costs
/// an EGL context in every instance that enables it.
///
/// That context is not free to create and destroy. The NVIDIA EGL runtime
/// registers a thread-local whose destructor takes an EGL-internal lock at
/// thread exit, and `vkDestroyDevice` joins driver threads that run that
/// destructor. Two threads tearing down instances at once close a cycle:
/// one waits inside `vkDestroyDevice` for a thread parked in the EGL
/// destructor, that lock is held by the thread destroying the other instance,
/// and that thread waits on the Vulkan loader lock the first one holds. No
/// deadline breaks it, because a thread already inside a thread-exit destructor
/// cannot be cancelled. Not enabling GL is what removes the cycle: this driver
/// never wanted the adapter.
pub(crate) const COMPUTE_BACKENDS: wgpu::Backends = wgpu::Backends::VULKAN
    .union(wgpu::Backends::METAL)
    .union(wgpu::Backends::DX12)
    .union(wgpu::Backends::BROWSER_WEBGPU);

/// One wgpu instance for the whole process.
///
/// Creating an instance starts the Vulkan loader, which dlopens every installed
/// ICD, and destroying it unloads them again. glibc reserves a fixed static TLS
/// surplus for dlopened libraries and does not return the NVIDIA driver's
/// initial-exec block when the ICD unloads, so the tenth loader cycle in one
/// process fails to load `libGLX_nvidia.so.0` with "cannot allocate memory in
/// static TLS block" and every enumeration after it reports the software
/// rasterizer alone. A conformance run measured that exactly: 9 of 349
/// operations dispatched and the remaining 340 were refused for want of a real
/// adapter on a host holding an idle RTX 3080 Ti.
///
/// One instance for the process removes the cycle. An adapter and a device are
/// still acquired per backend, so two backends still hold two physical devices
/// and device-loss recovery still replaces one device without disturbing the
/// other.
static INSTANCE: LazyLock<wgpu::Instance> = LazyLock::new(|| {
    let _startup = loader_startup();
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: COMPUTE_BACKENDS,
        ..Default::default()
    })
});

/// The process-wide wgpu instance every adapter and device comes from.
pub(crate) fn shared_instance() -> &'static wgpu::Instance {
    &INSTANCE
}

pub(crate) fn poll_device_once(
    device: &wgpu::Device,
) -> std::result::Result<wgpu::PollStatus, vyre_driver::BackendError> {
    device.poll(wgpu::PollType::Poll).map_err(|error| {
        vyre_driver::BackendError::new(format!(
            "wgpu device poll failed: {error}. Fix: inspect device loss and driver health before reusing this backend."
        ))
    })
}

pub(crate) fn poll_device_wait_for(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
) -> std::result::Result<wgpu::PollStatus, vyre_driver::BackendError> {
    device
        .poll(wgpu::PollType::wait_for(submission))
        .map_err(|error| {
            vyre_driver::BackendError::new(format!(
                "wgpu device wait-for-submission poll failed: {error}. Fix: inspect device loss, driver health, and submission lifetime before reusing this backend."
            ))
        })
}

struct CachedRuntime {
    device_queue: Arc<(wgpu::Device, wgpu::Queue)>,
    adapter_info: wgpu::AdapterInfo,
    #[cfg(all(test, feature = "device-tests"))]
    enabled_features: EnabledFeatures,
}

static CACHED_RUNTIME: LazyLock<Result<CachedRuntime>> = LazyLock::new(|| {
    #[cfg(all(test, feature = "device-tests"))]
    let ((device, queue), adapter_info, enabled_features) = init_device()?;
    #[cfg(not(all(test, feature = "device-tests")))]
    let ((device, queue), adapter_info, _enabled_features) = init_device()?;
    Ok(CachedRuntime {
        device_queue: Arc::new((device, queue)),
        adapter_info,
        #[cfg(all(test, feature = "device-tests"))]
        enabled_features,
    })
});

fn cached_runtime() -> &'static Result<CachedRuntime> {
    &CACHED_RUNTIME
}

/// Acquire the singleton device/queue pair.
///
/// ⚠ **Test / convenience helper  -  not the production path.**
///
/// Production backends construct their own `wgpu::Device` via
/// [`WgpuBackend::acquire`](crate::WgpuBackend::acquire), which routes
/// through [`init_device`] and returns a fresh device per call. Using
/// `cached_device()` from production code forces every consumer to
/// share one process-wide GPU handle, which prevents:
///
/// - running two backends against two different physical GPUs;
/// - using a dedicated discrete GPU while a test fixture is holding
///   the integrated GPU singleton;
/// - recovering from device loss (recovery swaps the backend's local
///   device; the singleton's `LazyLock` cannot be replaced in-place).
///
/// The singleton survives because a handful of test fixtures want one
/// shared GPU handle across all tests to amortize init cost. Consumers
/// that actually need a GPU runtime should construct a `WgpuBackend`
/// instead.
///
/// # Errors
///
/// Returns an error if the GPU adapter or device cannot be initialized.
#[inline]
pub fn cached_device() -> Result<Arc<(wgpu::Device, wgpu::Queue)>> {
    cached_runtime()
        .as_ref()
        .map(|runtime| Arc::clone(&runtime.device_queue))
        .map_err(Clone::clone)
}

/// Acquire adapter info for the singleton runtime device.
///
/// # Errors
///
/// Returns an error if the GPU adapter or device cannot be initialized.
#[inline]
pub fn cached_adapter_info() -> Result<&'static wgpu::AdapterInfo> {
    cached_runtime()
        .as_ref()
        .map(|runtime| &runtime.adapter_info)
        .map_err(Clone::clone)
}

/// Acquire the enabled feature snapshot for the singleton runtime device.
#[cfg(all(test, feature = "device-tests"))]
pub(crate) fn cached_enabled_features() -> Result<&'static EnabledFeatures> {
    cached_runtime()
        .as_ref()
        .map(|runtime| &runtime.enabled_features)
        .map_err(Clone::clone)
}

/// Return true when the device is the singleton cached device.
///
/// Asking the question initializes the singleton, because the cell holds its
/// own initializer. Every caller is a device test that already acquired it, so
/// the answer is the comparison and not a probe, and on a host with no adapter
/// the initializer errors and the answer is false.
#[cfg(all(test, feature = "device-tests"))]
#[inline]
pub(crate) fn is_cached_device(device: &wgpu::Device) -> bool {
    cached_runtime()
        .as_ref()
        .ok()
        .map(|runtime| &runtime.device_queue.0 == device)
        .unwrap_or(false)
}

/// Initialize a new GPU device and queue.
///
/// # Errors
///
/// Returns an actionable GPU error if no compatible adapter is available, if
/// the selected adapter is CPU-backed, or if device creation fails.
#[inline]
pub fn init_device() -> Result<(
    (wgpu::Device, wgpu::Queue),
    wgpu::AdapterInfo,
    EnabledFeatures,
)> {
    let gpu = wait_for_gpu(acquire_gpu())?;
    Ok(gpu)
}

/// Asynchronously initialize a new GPU device and queue.
///
/// # Errors
///
/// Returns an actionable GPU error if no compatible adapter is available, if
/// the selected adapter is CPU-backed, or if device creation fails.
#[inline]
pub async fn acquire_gpu() -> Result<(
    (wgpu::Device, wgpu::Queue),
    wgpu::AdapterInfo,
    EnabledFeatures,
)> {
    if let Some(index) = super::selector::adapter_index_from_env()? {
        return super::selector::acquire_gpu_for_adapter(index).await;
    }

    let instance = shared_instance();
    let adapters = instance.enumerate_adapters(COMPUTE_BACKENDS);
    let mut candidates = Vec::new();
    reserve_probe_vec(
        &mut candidates,
        adapters.len(),
        "GPU acquisition candidates",
    )?;
    candidates.extend(adapters.iter().filter_map(|adapter| {
        let info = adapter.get_info();
        crate::capabilities::is_real_gpu(&info).then(|| {
            let score = gpu_candidate_score(&info, adapter.features(), &adapter.limits());
            (adapter, info, score)
        })
    }));
    candidates.sort_by(|left, right| right.2.cmp(&left.2));

    let mut failures = Vec::new();
    reserve_probe_vec(&mut failures, candidates.len(), "GPU acquisition failures")?;
    for (adapter, info, _) in candidates {
        match request_device_for_adapter(adapter, "vyre device").await {
            Ok(device) => return Ok(device),
            Err(error) => failures.push(format!("{} ({:?}): {error}", info.name, info.device_type)),
        }
    }

    let mut probed = Vec::new();
    reserve_probe_vec(&mut probed, adapters.len(), "GPU acquisition probe report")?;
    probed.extend(adapters.iter().map(|adapter| {
        let info = adapter.get_info();
        format!(
            "{} ({:?}, backend={:?})",
            info.name, info.device_type, info.backend
        )
    }));
    Err(BackendError::new(format!(
        "no real GPU adapter could create a wgpu device. Probed adapters: [{}]. Device failures: [{}]. Fix: expose a discrete, integrated, or virtual GPU through a wgpu-supported driver before running vyre.",
        probed.join(", "),
        failures.join("; ")
    )))
}

pub(super) async fn request_device_for_adapter(
    adapter: &wgpu::Adapter,
    label: &'static str,
) -> Result<(
    (wgpu::Device, wgpu::Queue),
    wgpu::AdapterInfo,
    EnabledFeatures,
)> {
    let adapter_info = adapter.get_info();
    if !crate::capabilities::is_real_gpu(&adapter_info) {
        return Err(BackendError::new(format!(
            "wgpu adapter `{}` reports device type {:?}, which is not a real GPU execution target. Fix: select a discrete, integrated, or virtual GPU adapter; CPU/software adapters are not production dispatch backends.",
            adapter_info.name, adapter_info.device_type
        )));
    }
    // Opt into every feature the adapter advertises that we know how to
    // lower against. Each feature is additive: enabling it unlocks the
    // corresponding VyreBackend capability report (see
    // `WgpuBackend::supports_subgroup_ops`, `supports_f16`, etc.) and
    // costs nothing at runtime if no lowering emits the corresponding
    // intrinsic. Features we do NOT lower against (e.g. mesh shaders,
    // ray tracing) are deliberately omitted  -  enabling them would be a
    // LAW 9 evasion (claiming support that the lowering path does not
    // deliver).
    let adapter_features = adapter.features();
    let adapter_limits = adapter.limits();
    let (mut features, mut enabled) =
        enabled_features_for_adapter(adapter_features, &adapter_limits, adapter_info.backend);

    let mut device_queue = request_device_with(
        adapter,
        label,
        features,
        &enabled,
        &adapter_limits,
        &adapter_info,
    )
    .await?;

    // An adapter may advertise both timestamp features and still resolve a zero
    // end timestamp, which reaches a caller as a delta underflow on its first
    // timed dispatch instead of as a missing capability. Resolve once here, in
    // the layout every timed dispatch records, and drop the capability when the
    // pair is not monotonic. The device is recreated without the features so
    // `device.features()` agrees with what the resolve proved.
    let resolve_verdict = if enabled.timestamp_query && enabled.timestamp_query_inside_encoders {
        timestamp_resolve_is_monotonic(&device_queue.0, &device_queue.1)
    } else {
        Ok(())
    };
    if let Err(reason) = resolve_verdict {
        tracing::warn!(
            target: "vyre.wgpu.timestamps",
            adapter = %adapter_info.name,
            %reason,
            "adapter advertises timestamp queries but its resolve produced no monotonic pair; reporting no timestamp capability"
        );
        features.remove(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        );
        enabled.timestamp_query = false;
        enabled.timestamp_query_inside_encoders = false;
        drop(device_queue);
        device_queue = request_device_with(
            adapter,
            label,
            features,
            &enabled,
            &adapter_limits,
            &adapter_info,
        )
        .await?;
    }

    let device_limits = device_queue.0.limits();
    enabled.max_workgroup_size = [
        device_limits.max_compute_workgroup_size_x,
        device_limits.max_compute_workgroup_size_y,
        device_limits.max_compute_workgroup_size_z,
    ];
    enabled.max_storage_buffer_binding_size =
        u64::from(device_limits.max_storage_buffer_binding_size);
    enabled.max_subgroup_size = device_limits.max_subgroup_size;
    enabled.min_subgroup_size = device_limits.min_subgroup_size;

    if enabled.subgroup {
        subgroup_smoke_compiles(&device_queue.0).map_err(|error| BackendError::new(format!(
            "adapter `{}` advertises SUBGROUP but rejects the subgroup compute-pipeline smoke test: {error}. Fix: repair the wgpu feature negotiation or GPU driver; do not silently report subgroup support as disabled on a subgroup-capable adapter.",
            adapter_info.name
        )))?;
    }

    Ok((device_queue, adapter_info, enabled))
}

async fn request_device_with(
    adapter: &wgpu::Adapter,
    label: &'static str,
    features: wgpu::Features,
    enabled: &EnabledFeatures,
    adapter_limits: &wgpu::Limits,
    adapter_info: &wgpu::AdapterInfo,
) -> Result<(wgpu::Device, wgpu::Queue)> {
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some(label),
                required_features: features,
                required_limits: wgpu::Limits {
                    max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
                    max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y,
                    max_compute_workgroup_size_z: adapter_limits.max_compute_workgroup_size_z,
                    max_compute_invocations_per_workgroup: adapter_limits
                        .max_compute_invocations_per_workgroup,
                    max_compute_workgroups_per_dimension: adapter_limits
                        .max_compute_workgroups_per_dimension,
                    max_compute_workgroup_storage_size: adapter_limits
                        .max_compute_workgroup_storage_size,
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    // Modern adapters expose multi-GiB per-buffer caps; the
                    // wgpu spec floor is 256 MiB which is too small for
                    // batch-amortized scanners (`MAX_BATCH × num_rules
                    // × 65 536 × 4` packed-output buffer scales beyond that
                    // when MAX_BATCH grows past ~50). Take whatever the
                    // adapter reports  -  falls back to the spec floor on
                    // adapters that don't expose more.
                    max_buffer_size: adapter_limits.max_buffer_size,
                    min_subgroup_size: if enabled.subgroup {
                        adapter_limits.min_subgroup_size
                    } else {
                        0
                    },
                    max_subgroup_size: if enabled.subgroup {
                        adapter_limits.max_subgroup_size
                    } else {
                        0
                    },
                    max_storage_buffers_per_shader_stage:
                        adapter_limits.max_storage_buffers_per_shader_stage,
                    max_push_constant_size: if enabled.push_constants {
                        adapter_limits.max_push_constant_size
                    } else {
                        0
                    },
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        )
        .await
        .map_err(|error| BackendError::new(format!("failed to acquire device for adapter `{}`: {error}. Fix: check requested wgpu limits/features against the adapter and update the GPU driver if limits are unexpectedly low.", adapter_info.name)))
}

/// Resolve the timed-dispatch query layout once on a created device and report
/// whether it produced usable timestamps.
///
/// A Metal adapter advertises `TIMESTAMP_QUERY` and
/// `TIMESTAMP_QUERY_INSIDE_ENCODERS`, admits the capability check, and resolves
/// an end-of-pass timestamp of zero. Without this resolve the first timed
/// dispatch reports a delta underflow, which reads as a broken dispatch rather
/// than as an adapter that cannot be timed.
fn timestamp_resolve_is_monotonic(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> std::result::Result<(), String> {
    use crate::engine::record_and_readback::timestamp::{
        timestamp_ticks, TIMESTAMP_QUERY_COUNT, TIMESTAMP_READBACK_BYTES,
    };

    const WGSL: &str = r"
@compute @workgroup_size(1)
fn main() {
}
";

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("vyre timestamp resolve probe"),
        ty: wgpu::QueryType::Timestamp,
        count: TIMESTAMP_QUERY_COUNT,
    });
    let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vyre timestamp resolve probe resolve"),
        size: TIMESTAMP_READBACK_BYTES,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vyre timestamp resolve probe readback"),
        size: TIMESTAMP_READBACK_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vyre timestamp resolve probe"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vyre timestamp resolve probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vyre timestamp resolve probe"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vyre timestamp resolve probe"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        pass.set_pipeline(&pipeline);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.write_timestamp(&query_set, 2);
    encoder.write_timestamp(&query_set, 3);
    encoder.resolve_query_set(&query_set, 0..TIMESTAMP_QUERY_COUNT, &resolve_buffer, 0);
    encoder.copy_buffer_to_buffer(
        &resolve_buffer,
        0,
        &readback_buffer,
        0,
        TIMESTAMP_READBACK_BYTES,
    );
    let submission = queue.submit(std::iter::once(encoder.finish()));

    let _ = poll_device_wait_for(device, submission).map_err(|error| error.to_string())?;
    // After the submission wait, so encoder-recording validation has surfaced
    // and the scope resolves without a second blocking poll.
    match pop_error_scope_now(device) {
        Ok(None) => {}
        Ok(Some(error)) => return Err(format!("validation error: {error}")),
        // The resolved ticks are the verdict. An error scope that has not
        // resolved is not evidence against the device, and withdrawing a
        // working capability over it would cost every timed dispatch.
        Err(reason) => tracing::debug!(
            target: "vyre.wgpu.timestamps",
            %reason,
            "timestamp probe error scope did not resolve"
        ),
    }

    let (sender, receiver) = crossbeam_channel::bounded(1);
    readback_buffer
        .slice(0..TIMESTAMP_READBACK_BYTES)
        .map_async(wgpu::MapMode::Read, move |result| {
            drop(sender.send(result));
        });
    let _ = device
        .poll(wgpu::PollType::Wait)
        .map_err(|error| format!("device poll before timestamp probe readback failed: {error}"))?;
    match receiver.recv_timeout(TIMESTAMP_PROBE_READBACK_TIMEOUT) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(format!("probe readback mapping failed: {error:?}")),
        Err(error) => return Err(format!("probe readback did not complete: {error}")),
    }

    let ticks = {
        let mapped = readback_buffer
            .slice(0..TIMESTAMP_READBACK_BYTES)
            .get_mapped_range();
        if mapped.len() < TIMESTAMP_READBACK_BYTES as usize {
            let len = mapped.len();
            return Err(format!(
                "probe readback returned {len} bytes, expected {TIMESTAMP_READBACK_BYTES}"
            ));
        }
        timestamp_ticks(&mapped)
    };
    readback_buffer.unmap();

    timestamp_ticks_are_usable(&ticks)
}

/// Pairs a timed dispatch subtracts: the compute pass, the encoder bracket, and
/// the whole submission. A resolve is usable only if every one of them advances.
const TIMESTAMP_MONOTONIC_PAIRS: [(usize, usize); 3] = [(1, 0), (3, 2), (3, 0)];

/// A resolve is usable when each recorded pair advances and the whole set is
/// not zero. An all-zero resolve advertises a device time of zero for every
/// dispatch, which is not a measurement.
fn timestamp_ticks_are_usable(ticks: &[u64]) -> std::result::Result<(), String> {
    if ticks.iter().all(|tick| *tick == 0) {
        return Err("resolve returned only zero timestamps".to_string());
    }
    for (end, start) in TIMESTAMP_MONOTONIC_PAIRS {
        let (Some(&end_tick), Some(&start_tick)) = (ticks.get(end), ticks.get(start)) else {
            return Err(format!(
                "resolve returned {} timestamps, too few to compare query {end} against {start}",
                ticks.len()
            ));
        };
        if end_tick < start_tick {
            return Err(format!(
                "query {end} resolved {end_tick}, earlier than query {start} at {start_tick}"
            ));
        }
    }
    Ok(())
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}

    fn wake_by_ref(self: &Arc<Self>) {}
}

pub(crate) fn pop_error_scope_now(
    device: &wgpu::Device,
) -> std::result::Result<Option<wgpu::Error>, &'static str> {
    device
        .poll(wgpu::PollType::Poll)
        .map_err(|_| "wgpu device poll failed before error-scope pop")?;
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(device.pop_error_scope());
    match Future::poll(Pin::as_mut(&mut future), &mut context) {
        Poll::Ready(error) => Ok(error),
        Poll::Pending => Err(
            "wgpu error scope did not resolve after a nonblocking device poll. Fix: inspect the backend event loop; validation must not require a hot-path host wait.",
        ),
    }
}

pub(super) fn wait_for_gpu<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    loop {
        match Pin::as_mut(&mut future).poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

// Inline: the suite drives the crate-private `cached_enabled_features`,
// `is_cached_device`, which an integration test does not compile. The two
// singleton tests open a real device, so hardware admits them and the rest of
// this module is host arithmetic that runs in every lane.
#[cfg(test)]
mod tests {
    use super::*;

    /// Every pair the timed-dispatch layout compares, driven from the layout
    /// itself rather than a hand-listed set, so a query count change turns this
    /// red instead of leaving a pair unchecked.
    #[test]
    fn a_pair_that_runs_backwards_is_not_a_usable_resolve() {
        let count = crate::engine::record_and_readback::timestamp::TIMESTAMP_QUERY_COUNT as usize;
        let highest = TIMESTAMP_MONOTONIC_PAIRS
            .iter()
            .flat_map(|(end, start)| [*end, *start])
            .max()
            .expect("Fix: a timed dispatch must compare at least one pair");
        assert!(
            highest < count,
            "Fix: pair {highest} is compared but the resolve records only {count} queries"
        );
        let baseline: Vec<u64> = (0..count).map(|index| 1_000 + index as u64).collect();
        timestamp_ticks_are_usable(&baseline)
            .expect("Fix: an increasing resolve is the usable case");

        for (end, start) in TIMESTAMP_MONOTONIC_PAIRS {
            let mut ticks = baseline.clone();
            ticks[end] = baseline[start] - 1;
            let reason = timestamp_ticks_are_usable(&ticks)
                .expect_err("Fix: a pair whose end precedes its start is not a usable resolve");
            assert!(
                reason.contains(&format!("query {end} resolved")),
                "Fix: the reason must name the query that ran backwards, got {reason}"
            );
        }
    }

    #[test]
    fn a_zero_end_timestamp_is_reported_as_no_capability() {
        // The measured Metal resolve: a beginning-of-pass tick and a
        // zero end-of-pass tick, which reached a caller as a delta underflow.
        let reason = timestamp_ticks_are_usable(&[1_177_748_395_222_791, 0, 1, 2])
            .expect_err("Fix: a zero end-of-pass tick is not a usable resolve");
        assert!(
            reason.contains("query 1 resolved 0"),
            "Fix: the reason must name the zero tick, got {reason}"
        );
    }

    #[test]
    fn an_all_zero_resolve_is_not_a_measurement() {
        let reason = timestamp_ticks_are_usable(&[0, 0, 0, 0])
            .expect_err("Fix: an all-zero resolve reports a zero device time for every dispatch");
        assert!(
            reason.contains("only zero timestamps"),
            "Fix: the reason must say the resolve was all zero, got {reason}"
        );
    }

    #[test]
    fn a_short_resolve_names_the_pair_it_cannot_compare() {
        let highest = TIMESTAMP_MONOTONIC_PAIRS
            .iter()
            .flat_map(|(end, start)| [*end, *start])
            .max()
            .expect("Fix: a timed dispatch must compare at least one pair");
        let (end, start) = TIMESTAMP_MONOTONIC_PAIRS
            .into_iter()
            .find(|(end, start)| *end >= highest || *start >= highest)
            .expect("Fix: the highest compared index belongs to some pair");
        let ticks = vec![1u64; highest];
        let reason = timestamp_ticks_are_usable(&ticks)
            .expect_err("Fix: too few timestamps cannot prove a monotonic layout");
        assert!(
            reason.contains(&format!("too few to compare query {end} against {start}")),
            "Fix: the reason must name the pair it could not read, got {reason}"
        );
    }

    /// The cached-device helper now returns a stable singleton.
    #[cfg(feature = "device-tests")]
    #[test]
    fn cached_device_is_singleton() {
        let first = cached_device().expect("Fix: GPU must be available for runtime tests");
        let second = cached_device().expect("Fix: GPU must be available for runtime tests");
        assert!(
            Arc::ptr_eq(&first, &second),
            "cached_device must return the same Arc after singleton initialization"
        );
        assert!(
            is_cached_device(&first.0),
            "legacy shared APIs must still recognize cached_device-created devices"
        );
    }

    #[cfg(feature = "device-tests")]
    #[test]
    fn cached_adapter_info_uses_cached_runtime() {
        let info = cached_adapter_info().expect("Fix: cached adapter info must share GPU init");
        let enabled =
            cached_enabled_features().expect("Fix: cached runtime must retain capability snapshot");
        let device_queue = cached_device().expect("Fix: GPU must be available for runtime tests");
        assert!(
            !info.name.is_empty(),
            "cached adapter info must come from the initialized runtime adapter"
        );
        assert!(
            enabled.max_workgroup_size.iter().all(|axis| *axis > 0),
            "cached runtime must retain nonzero device workgroup limits for capability reporting"
        );
        assert!(
            is_cached_device(&device_queue.0),
            "cached adapter info must not replace the cached device with a second init"
        );
    }

    /// Every backend wgpu publishes is either dispatched through or excluded
    /// with a reason.
    ///
    /// `COMPUTE_BACKENDS` is a hand-written mask, so a wgpu upgrade that adds a
    /// backend would silently leave it out of both the instance and every
    /// adapter probe. The member set is read from `Backends::all()` rather than
    /// restated, so a new member belongs to neither list and turns this red
    /// until someone records which one it joins.
    #[test]
    fn every_published_backend_is_dispatched_or_excluded_with_a_reason() {
        const EXCLUDED: [(wgpu::Backends, &str); 2] = [
            (
                wgpu::Backends::GL,
                "reaches a device Vulkan or Metal already reports, lowers no compute this driver \
                 emits, and costs an EGL context whose thread-exit teardown deadlocks against \
                 concurrent device destruction",
            ),
            (
                wgpu::Backends::NOOP,
                "executes nothing, so a program dispatched on it proves nothing",
            ),
        ];

        let excluded = EXCLUDED
            .iter()
            .fold(wgpu::Backends::empty(), |mask, (backend, _)| {
                mask | *backend
            });
        assert!(
            !COMPUTE_BACKENDS.intersects(excluded),
            "Fix: {:?} is both dispatched through and excluded.",
            COMPUTE_BACKENDS & excluded
        );
        assert_eq!(
            COMPUTE_BACKENDS | excluded,
            wgpu::Backends::all(),
            "Fix: {:?} is a backend wgpu publishes that this driver neither dispatches through nor \
             excludes with a reason. Add it to COMPUTE_BACKENDS, or to EXCLUDED naming why a \
             compute program must not run there.",
            wgpu::Backends::all() - (COMPUTE_BACKENDS | excluded)
        );
        for (backend, reason) in EXCLUDED {
            assert!(
                !reason.trim().is_empty(),
                "Fix: {backend:?} is excluded with no reason recorded."
            );
        }
    }
}
