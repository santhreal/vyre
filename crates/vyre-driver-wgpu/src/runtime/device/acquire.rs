use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};
use vyre_runtime::BackendError;

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

/// One device per adapter, for the process lifetime.
///
/// `wgpu::Device` is a reference-counted handle and dropping the last one
/// calls `vkDestroyDevice`. On the NVIDIA driver that call joins a driver
/// worker, and a worker thread exiting runs the `libnvidia-eglcore`
/// thread-local destructor, which takes a lock inside `libnvidia-glsi`
/// through a priority-inheritance futex. A second thread creating or
/// destroying a device at that moment holds that lock, so the two wait on each
/// other, neither wait carries a deadline, and the process hangs until it is
/// killed. Captured under gdb on an RTX 3080 Ti with driver 580.173.02 from a
/// four-thread create-and-release loop that reproduced it on 6 of 8 runs.
///
/// Handing every caller of one adapter the same device removes the call: no
/// caller's drop is ever the last reference, so `vkDestroyDevice` is
/// unreachable and the cycle cannot close. The bound matters as much as the
/// exclusion. Holding one live device per created backend instead reaches the
/// driver's limit, and the request after it fails with `Parent device is
/// lost`, which turns a suite that hangs into a suite that cannot get a
/// device.
///
/// Device loss replaces the entry rather than dropping it. The lost handle
/// moves to [`RETIRED_DEVICES`], because destroying it is the call this
/// excludes.
static ADAPTER_DEVICES: Mutex<Vec<AdapterDevice>> = Mutex::new(Vec::new());

/// Devices replaced by device-loss recovery, held so none is ever destroyed.
///
/// One entry per loss event on one adapter, so this grows only when the GPU
/// faults.
static RETIRED_DEVICES: Mutex<Vec<wgpu::Device>> = Mutex::new(Vec::new());

/// The device this process holds for one adapter. See [`ADAPTER_DEVICES`].
struct AdapterDevice {
    info: wgpu::AdapterInfo,
    device_queue: (wgpu::Device, wgpu::Queue),
    enabled: EnabledFeatures,
}

/// Whether a caller takes the device this process already holds for an adapter.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DeviceRequest {
    /// Take the held device. Every caller of one adapter shares it.
    Held,
    /// Replace it, because the device it holds reported loss.
    Replace,
}

/// Recover a poisoned registry lock rather than propagating it.
///
/// The guarded value is a list a panic cannot leave inconsistent, and refusing
/// the lock would leave the caller to create an unshared device, which is the
/// teardown this registry excludes.
fn lock_registry<T>(registry: &'static Mutex<T>) -> MutexGuard<'static, T> {
    registry.lock().unwrap_or_else(PoisonError::into_inner)
}

/// How many live devices this process holds for the adapter `info` describes.
#[cfg(all(test, feature = "device-tests"))]
pub(crate) fn held_devices_for(info: &wgpu::AdapterInfo) -> usize {
    lock_registry(&ADAPTER_DEVICES)
        .iter()
        .filter(|entry| entry.info == *info)
        .count()
}

/// Acquisition reports where it stalled rather than parking forever.
///
/// Adapter enumeration, `request_device`, and the capability probe all run
/// under this bound. It is generous because a cold ICD load on a loaded host
/// is slow, and it exists so a driver that never wakes the future surfaces as
/// an error naming the stage instead of a run that never ends.
const ACQUISITION_TIMEOUT: Duration = Duration::from_secs(120);

/// Take the loader lock, or end the process.
///
/// Poison here is the exact state the lock excludes. A thread that panicked
/// inside `vkCreateInstance` left the loader dispatch table half written,
/// `PoisonError::into_inner` hands that table to the next caller, and what the
/// caller gets is the SIGSEGV the comment above describes, in an ICD frame that
/// names no vyre code. Loader state is also not this process's to rebuild.
fn loader_startup() -> MutexGuard<'static, ()> {
    match vyre_runtime::lock_policy::govern_mutex(
        &LOADER_STARTUP,
        "the wgpu device factory",
        "the graphics loader dispatch table",
        vyre_runtime::lock_policy::RecoveryClass::ProcessFatal,
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
/// cannot be cancelled. Not enabling GL drops one EGL context this driver never
/// wanted, which narrows the window without closing it: a Vulkan adapter on the
/// NVIDIA driver loads `libnvidia-eglcore` regardless, and the same destructor
/// appears in the device-teardown deadlock [`ADAPTER_DEVICES`] excludes.
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
/// One instance for the process removes the cycle. Each adapter still has its
/// own device, so backends on two physical GPUs hold two devices and
/// device-loss recovery on one adapter leaves the other's device in place.
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
) -> std::result::Result<wgpu::PollStatus, vyre_runtime::BackendError> {
    device.poll(wgpu::PollType::Poll).map_err(|error| {
        vyre_runtime::BackendError::new(format!(
            "wgpu device poll failed: {error}. Fix: inspect device loss and driver health before reusing this backend."
        ))
    })
}

pub(crate) fn poll_device_wait_for(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
) -> std::result::Result<wgpu::PollStatus, vyre_runtime::BackendError> {
    device
        .poll(wgpu::PollType::wait_for(submission))
        .map_err(|error| {
            vyre_runtime::BackendError::new(format!(
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
/// Production backends acquire their device via
/// [`WgpuBackend::acquire`](crate::WgpuBackend::acquire), which routes
/// through [`init_device`] and takes the one device the process holds for
/// the selected adapter. `cached_device()` pins the first adapter
/// [`init_device`] selected for the process lifetime, which prevents:
///
/// - running two backends against two different physical GPUs;
/// - using a dedicated discrete GPU while a test fixture is holding
///   the integrated GPU singleton;
/// - recovering from device loss (recovery replaces the backend's held
///   device; the singleton's `LazyLock` cannot be replaced in-place).
///
/// The singleton survives because a handful of test fixtures want one
/// shared GPU handle across all tests to amortize init cost. Consumers
/// that need a GPU runtime construct a `WgpuBackend` instead.
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

/// Acquire a device on the best real adapter with every timestamp feature
/// withheld.
///
/// An adapter that advertises no timestamp pair, and one whose resolve proves
/// no monotonic pair, both reach dispatch as a device whose `features()` carry
/// neither timestamp bit. Withholding the bits here reproduces that device on
/// a host whose adapter does advertise them, which is what lets one GPU host
/// exercise both branches of every timed dispatch path.
///
/// The device is created once and held for the process, like every device in
/// [`ADAPTER_DEVICES`], so no test's release reaches `vkDestroyDevice`.
///
/// # Errors
///
/// Returns an error when no real adapter is available or device creation
/// fails.
#[cfg(all(test, feature = "device-tests"))]
pub(crate) fn init_device_without_timestamps() -> Result<(
    (wgpu::Device, wgpu::Queue),
    wgpu::AdapterInfo,
    EnabledFeatures,
)> {
    type Held = Result<(
        (wgpu::Device, wgpu::Queue),
        wgpu::AdapterInfo,
        EnabledFeatures,
    )>;
    static HELD: std::sync::OnceLock<Held> = std::sync::OnceLock::new();
    HELD.get_or_init(create_device_without_timestamps).clone()
}

#[cfg(all(test, feature = "device-tests"))]
fn create_device_without_timestamps() -> Result<(
    (wgpu::Device, wgpu::Queue),
    wgpu::AdapterInfo,
    EnabledFeatures,
)> {
    let instance = shared_instance();
    let adapters = instance.enumerate_adapters(COMPUTE_BACKENDS);
    let mut best: Option<(&wgpu::Adapter, wgpu::AdapterInfo, u128)> = None;
    for adapter in adapters.iter() {
        let info = adapter.get_info();
        if !crate::capabilities::is_real_gpu(&info) {
            continue;
        }
        let score = gpu_candidate_score(&info, adapter.features(), &adapter.limits());
        if best.as_ref().is_none_or(|(_, _, seen)| score > *seen) {
            best = Some((adapter, info, score));
        }
    }
    let (adapter, adapter_info, _) = best.ok_or_else(|| {
        BackendError::new(
            "no real GPU adapter is available for a timestamp-free device. Fix: run this device test on a host with a discrete, integrated, or virtual GPU adapter."
                .to_string(),
        )
    })?;

    let adapter_limits = adapter.limits();
    let (mut features, mut enabled) =
        enabled_features_for_adapter(adapter.features(), &adapter_limits, adapter_info.backend);
    features
        .remove(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    enabled.timestamp_query = false;
    enabled.timestamp_query_inside_encoders = false;

    let device_queue = request_device_with(
        adapter,
        "vyre timestamp-free device",
        features,
        &enabled,
        &adapter_limits,
        &adapter_info,
    )?;

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

    Ok((device_queue, adapter_info, enabled))
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

/// Take the device and queue this process holds for the best real adapter,
/// creating them on first use.
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
    acquire_gpu_blocking()
}

/// Async variant of [`init_device`]. Acquisition runs synchronously.
///
/// # Errors
///
/// Returns an actionable GPU error if no compatible adapter is available, if
/// the selected adapter is CPU-backed, or if device creation fails.
pub async fn acquire_gpu() -> Result<(
    (wgpu::Device, wgpu::Queue),
    wgpu::AdapterInfo,
    EnabledFeatures,
)> {
    acquire_gpu_blocking()
}

/// Select the best adapter and take the device this process holds for it.
///
/// The whole chain is synchronous. Only the `request_device` future itself
/// needs driving, and [`wait_for_gpu`] does that at the one place it happens,
/// so no caller nests one park loop inside another.
fn acquire_gpu_blocking() -> Result<(
    (wgpu::Device, wgpu::Queue),
    wgpu::AdapterInfo,
    EnabledFeatures,
)> {
    if let Some(index) = super::selector::adapter_index_from_env()? {
        return super::selector::init_device_for_adapter(index);
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
        match request_device_for_adapter(adapter, "vyre device", DeviceRequest::Held) {
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

/// Take the device this process holds for `adapter`, creating it if needed.
///
/// `request` decides whether the held device is taken or replaced. Every
/// caller of one adapter receives the same device, which is what makes
/// `vkDestroyDevice` unreachable; see [`ADAPTER_DEVICES`].
pub(super) fn request_device_for_adapter(
    adapter: &wgpu::Adapter,
    label: &'static str,
    request: DeviceRequest,
) -> Result<(
    (wgpu::Device, wgpu::Queue),
    wgpu::AdapterInfo,
    EnabledFeatures,
)> {
    let mut devices = lock_registry(&ADAPTER_DEVICES);
    let adapter_info = adapter.get_info();
    let held = devices.iter().position(|entry| entry.info == adapter_info);
    match (request, held) {
        (DeviceRequest::Held, Some(index)) => {
            let entry = &devices[index];
            return Ok((
                entry.device_queue.clone(),
                entry.info.clone(),
                entry.enabled,
            ));
        }
        // The replaced device is never dropped to zero references, so
        // recovery cannot race a create on another thread.
        (DeviceRequest::Replace, Some(index)) => {
            let replaced = devices.swap_remove(index);
            lock_registry(&RETIRED_DEVICES).push(replaced.device_queue.0);
        }
        (_, None) => {}
    }
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
    )?;

    // An adapter may advertise both timestamp features and still resolve a zero
    // end timestamp, which reaches a caller as a delta underflow on its first
    // timed dispatch instead of as a missing capability. Resolve once here, in
    // the layout every timed dispatch records, and drop the capability when the
    // pair is not monotonic. The device is recreated without the features so
    // `device.features()` agrees with what the resolve proved.
    let resolve_verdict = if enabled.timestamp_query && enabled.timestamp_query_inside_encoders {
        super::timestamp_probe::timestamp_resolve_is_monotonic(&device_queue.0, &device_queue.1)
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
        // Releases this function's handle. The device was never registered, so
        // this is the one place the driver destroys one, and it happens under
        // the registry lock where no other thread can be creating a device.
        drop(device_queue);
        device_queue = request_device_with(
            adapter,
            label,
            features,
            &enabled,
            &adapter_limits,
            &adapter_info,
        )?;
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

    devices.push(AdapterDevice {
        info: adapter_info.clone(),
        device_queue: device_queue.clone(),
        enabled,
    });
    Ok((device_queue, adapter_info, enabled))
}

fn request_device_with(
    adapter: &wgpu::Adapter,
    label: &'static str,
    features: wgpu::Features,
    enabled: &EnabledFeatures,
    adapter_limits: &wgpu::Limits,
    adapter_info: &wgpu::AdapterInfo,
) -> Result<(wgpu::Device, wgpu::Queue)> {
    wait_for_gpu(
        "device request",
        adapter.request_device(
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
        ),
    )?
    .map_err(|error| BackendError::new(format!("failed to acquire device for adapter `{}`: {error}. Fix: check requested wgpu limits/features against the adapter and update the GPU driver if limits are unexpectedly low.", adapter_info.name)))
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

/// Drive `future` to completion on this thread, under [`ACQUISITION_TIMEOUT`].
///
/// `stage` names the step being waited on and appears in the timeout error, so
/// a stalled acquisition states where it stopped instead of parking with no
/// output.
///
/// # Errors
///
/// Returns `BackendError` when the future makes no progress before the
/// deadline.
pub(super) fn wait_for_gpu<T>(stage: &str, future: impl Future<Output = T>) -> Result<T> {
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    let deadline = Instant::now() + ACQUISITION_TIMEOUT;
    loop {
        if let Poll::Ready(value) = Pin::as_mut(&mut future).poll(&mut context) {
            return Ok(value);
        }
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return Err(BackendError::new(format!(
                "GPU acquisition stalled in {stage} for {}s without progress. Fix: check the GPU driver and the adapter it is enumerating; a device request that never resolves is a driver fault, not a vyre one.",
                ACQUISITION_TIMEOUT.as_secs()
            )));
        };
        thread::park_timeout(remaining);
    }
}

// Inline: the suite drives the crate-private `cached_enabled_features`,
// `is_cached_device` and the held-device registry, which an integration test
// does not compile. The device tests open a real device, so hardware admits
// them, and the backend-mask test is host arithmetic that runs in every lane.
#[cfg(test)]
mod tests {
    use super::*;

    /// Creating and releasing devices from several threads at once terminates.
    ///
    /// This is the shape that deadlocked: threads each acquired a device and
    /// released it, one reached `vkDestroyDevice` while another was inside an
    /// NVIDIA thread-exit destructor, and neither wait carried a deadline. The
    /// bound is asserted by a receive deadline on the main thread rather than
    /// by joining the workers, because a wedged worker never returns from the
    /// driver and a join would hang with it.
    ///
    /// Termination alone proves nothing, since threads that do not interleave
    /// also terminate. Every worker must receive the same device, which is
    /// what makes a release unable to reach `vkDestroyDevice`, and the adapter
    /// must hold one device. The count is per adapter because other tests in
    /// this binary acquire devices on the host's other adapters concurrently.
    #[cfg(feature = "device-tests")]
    #[test]
    fn concurrent_device_creation_and_release_terminates() {
        const THREADS: usize = 4;
        const ROUNDS: usize = 3;
        const BOUND: Duration = Duration::from_secs(300);

        let (report, finished) = std::sync::mpsc::channel();
        for worker in 0..THREADS {
            let report = report.clone();
            thread::spawn(move || {
                let mut last = None;
                for round in 0..ROUNDS {
                    match init_device() {
                        Ok(((device, queue), info, _)) => {
                            last = Some((device, info));
                            drop(queue);
                        }
                        Err(error) => {
                            drop(report.send(Err(format!(
                                "worker {worker} round {round} could not acquire a device: {error}"
                            ))));
                            return;
                        }
                    }
                }
                drop(report.send(Ok(last)));
            });
        }
        drop(report);

        let deadline = Instant::now() + BOUND;
        let mut devices = Vec::new();
        while devices.len() < THREADS {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or_default();
            match finished.recv_timeout(remaining) {
                Ok(Ok(device)) => devices.push(
                    device.expect("Fix: a worker that reported success ran at least one round"),
                ),
                Ok(Err(reason)) => panic!("Fix: {reason}"),
                Err(_) => panic!(
                    "Fix: {} of {THREADS} threads finished within {}s. Concurrent device creation and release must not deadlock; a thread stuck in vkDestroyDevice means a device reached zero references.",
                    devices.len(),
                    BOUND.as_secs()
                ),
            }
        }

        let (first, info) = &devices[0];
        assert!(
            devices.iter().all(|(device, _)| device == first),
            "Fix: {THREADS} threads on one adapter received different devices. Every caller of an adapter must take the held device, or a release is the last reference and calls vkDestroyDevice."
        );
        let held = held_devices_for(info);
        assert_eq!(
            held, 1,
            "Fix: {THREADS} threads over {ROUNDS} rounds left {held} devices held for `{}`. One adapter must hold one device; holding one per acquisition exhausts the driver.",
            info.name
        );
    }

    /// The adapter probe takes the held device instead of creating its own.
    ///
    /// The probe runs as a diagnostic while other threads hold devices, so a
    /// probe device that is created and dropped is a `vkDestroyDevice` on the
    /// reporting path, which is the teardown the deadlock needs. Every real
    /// adapter the probe opened therefore holds exactly one device afterwards,
    /// on a host with one adapter or several, and the adapter `init_device`
    /// selected still hands out the device it held before the probe.
    #[cfg(feature = "device-tests")]
    #[test]
    fn the_adapter_probe_takes_the_held_device() {
        let ((device, _queue), selected, _) =
            init_device().expect("Fix: a device-test host must acquire a device before probing");
        let report = super::super::adapter_probe_report();
        assert!(
            !report.probed.is_empty(),
            "Fix: a device-test host must expose at least one adapter to probe"
        );
        let adapters = shared_instance().enumerate_adapters(COMPUTE_BACKENDS);
        for info in adapters.iter().map(wgpu::Adapter::get_info) {
            let failed = report
                .missing
                .iter()
                .any(|reason| reason.contains(&format!("device request failed on {}", info.name)));
            if !crate::capabilities::is_real_gpu(&info) || failed {
                continue;
            }
            assert_eq!(
                held_devices_for(&info),
                1,
                "Fix: after the adapter probe `{}` holds {} devices. The probe must take the held device, creating it only when none is held. Probe report: probed {:?}, missing {:?}.",
                info.name,
                held_devices_for(&info),
                report.probed,
                report.missing
            );
        }
        let ((after, _queue), ..) =
            init_device().expect("Fix: acquisition after the probe must still succeed");
        assert!(
            after == device,
            "Fix: the adapter probe replaced the device held for `{}`.",
            selected.name
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
