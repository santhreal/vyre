//! Contracts for the process-wide shared wgpu backend and the one wgpu instance
//! every acquisition in this crate uses.

#![cfg(feature = "device-tests")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};

mod harness;
use harness::selected_adapter;
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;

#[test]
fn shared_backend_reuses_single_backend_instance() {
    let first = WgpuBackend::shared().expect("Fix: shared backend requires a configured GPU");
    let second = WgpuBackend::shared().expect("Fix: shared backend must reuse the configured GPU");

    assert!(
        Arc::ptr_eq(&first, &second),
        "Fix: WgpuBackend::shared must return the same Arc-backed backend so scan paths reuse arenas and pipeline caches"
    );
    assert_eq!(first.id(), "wgpu");
}

/// The reported workgroup ceiling is what this backend can launch.
///
/// The earlier form of this contract demanded the report equal the raw device
/// limits, which on a device advertising `max_compute_workgroup_size_x = 1024`
/// asserted a capability the WGSL dialect refuses: a module declaring
/// `@workgroup_size(1024)` exceeds the 256 invocations per workgroup the
/// dialect admits, so every dispatch at the asserted width failed to compile.
/// A report is therefore checked against what a launch does. The ceiling never
/// exceeds the device, a launch at the ceiling runs, and the raw device limit
/// above it is refused rather than accepted and then failed inside the driver.
#[test]
fn reported_workgroup_ceiling_is_what_the_backend_can_launch() {
    let shared = WgpuBackend::shared().expect("Fix: shared backend requires a configured GPU");
    let concrete = WgpuBackend::new().expect("Fix: concrete backend requires a configured GPU");
    let adapter = selected_adapter(&shared);
    let adapter_limits = adapter.limits();
    let adapter_has_subgroup = adapter.features().contains(wgpu::Features::SUBGROUP)
        && adapter_limits.min_subgroup_size > 0;

    assert_eq!(
        shared.supports_subgroup_ops(),
        concrete.supports_subgroup_ops(),
        "Fix: shared backend must not stale or fabricate subgroup capability reporting"
    );
    assert_eq!(
        shared.supports_indirect_dispatch(),
        concrete.supports_indirect_dispatch(),
        "Fix: shared backend must not stale or fabricate indirect-dispatch capability reporting"
    );
    assert_eq!(
        shared.max_workgroup_size(),
        concrete.max_workgroup_size(),
        "Fix: shared backend must expose the same workgroup limits as the live concrete backend"
    );
    if adapter_has_subgroup {
        assert!(
            shared.supports_subgroup_ops(),
            "Fix: shared backend must not report subgroup_ops=false when the live adapter advertises SUBGROUP and min_subgroup_size={}.",
            adapter_limits.min_subgroup_size
        );
    }
    let reported = shared.max_workgroup_size();
    let device = [
        shared.device_limits().max_compute_workgroup_size_x,
        shared.device_limits().max_compute_workgroup_size_y,
        shared.device_limits().max_compute_workgroup_size_z,
    ];
    for axis in 0..3 {
        assert!(
            reported[axis] <= device[axis],
            "Fix: reported workgroup ceiling axis {axis} is {} but the selected device admits {}; a backend must not report an extent the device cannot run",
            reported[axis],
            device[axis]
        );
        assert!(
            reported[axis] > 0,
            "Fix: reported workgroup ceiling axis {axis} is zero, so no launch is expressible"
        );
    }

    let words = reported[0];
    let at_ceiling = harness::add_one_program_at_width(words, reported[0]);
    let outputs = shared
        .dispatch_borrowed(
            &at_ceiling,
            &[&harness::add_one_input(words)],
            &vyre_driver::DispatchConfig::default(),
        )
        .expect("Fix: a launch at the reported workgroup ceiling must reach the device");
    assert_eq!(
        outputs[0],
        harness::add_one_expected(words),
        "Fix: a launch at the reported workgroup ceiling must compute its result"
    );

    if device[0] > reported[0] {
        let above = harness::add_one_program_at_width(words, device[0]);
        let refused = shared.dispatch_borrowed(
            &above,
            &[&harness::add_one_input(words)],
            &vyre_driver::DispatchConfig::default(),
        );
        assert!(
            refused.is_err(),
            "Fix: the raw device width {} is above the dialect ceiling {} and must be refused, not reported as admissible",
            device[0],
            reported[0]
        );
    }
}

/// Acquiring and releasing a backend from many threads at once must finish on
/// every thread.
///
/// A `wgpu::Instance` owns the Vulkan loader instance and loader startup is not
/// reentrant, so two threads creating an instance together left one of them
/// calling through a null dispatch pointer inside
/// `vkEnumerateInstanceExtensionProperties`. That defect does not surface as a
/// failed assertion: the process dies with SIGSEGV and takes the harness with
/// it, so this test fails by disappearing. The barrier releases every thread
/// into acquisition at once, which is the state that faulted.
///
/// Release is the other half, and it is the half that hung. While an instance
/// enabled GL, teardown closed a cycle across the Vulkan loader lock, an EGL
/// lock, and a thread-exit destructor `vkDestroyDevice` was joining, so a
/// conformance run that acquired one backend per worker stopped for hours with
/// no output. Each thread therefore drops its backend and reports afterwards,
/// and the report is collected under a deadline: a thread stuck in teardown
/// cannot be joined, so a regression has to fail as an expired wait rather than
/// as a suite that never returns.
#[test]
fn concurrent_backend_acquisition_and_release_finishes_on_every_thread() {
    let threads = std::thread::available_parallelism().map_or(8, |count| count.get().max(8));
    let barrier = Arc::new(Barrier::new(threads));
    let (report, reports) = std::sync::mpsc::channel();
    for _ in 0..threads {
        let barrier = Arc::clone(&barrier);
        let report = report.clone();
        std::thread::spawn(move || {
            barrier.wait();
            let acquired = WgpuBackend::acquire().map(|backend| {
                let name = backend.adapter_info().name.clone();
                drop(backend);
                name
            });
            let _ = report.send(acquired);
        });
    }
    drop(report);

    let deadline = std::time::Duration::from_secs(120);
    let mut adapters = BTreeSet::new();
    for index in 0..threads {
        match reports.recv_timeout(deadline) {
            Ok(Ok(name)) => {
                adapters.insert(name);
            }
            Ok(Err(error)) => {
                panic!("Fix: concurrent wgpu acquisition failed on one of {threads} threads: {error}")
            }
            Err(error) => panic!(
                "Fix: {index} of {threads} concurrent acquisitions reported within {deadline:?} ({error}). A thread that acquired a backend and did not report has not finished releasing it; instance and device teardown must not block on another thread's teardown."
            ),
        }
    }
    assert_eq!(
        adapters.len(),
        1,
        "Fix: {threads} concurrent acquisitions selected {adapters:?}; adapter selection must not depend on which thread got there first."
    );
}

/// One module constructs the wgpu instance.
///
/// The loader race above is unrepresentable only while a single owner holds the
/// instance, and nothing in the type system stops a second `Instance::default()`
/// from being added somewhere else in the crate. The construction sites are
/// therefore read out of the crate source at run time, so a new one turns this
/// red instead of reintroducing a segmentation fault under concurrency.
#[test]
fn only_the_device_acquisition_module_constructs_a_wgpu_instance() {
    const OWNER: &str = "runtime/device/acquire.rs";
    let root =
        vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")).join("src");
    let mut sources = Vec::new();
    collect_rust_sources(&root, &mut sources);
    assert!(
        sources.len() > 1,
        "Fix: the crate source walk found {} files under {}; the ownership check needs the real source tree.",
        sources.len(),
        root.display()
    );

    let mut sites = BTreeSet::new();
    for source in &sources {
        let text = std::fs::read_to_string(source).unwrap_or_else(|error| {
            panic!(
                "Fix: could not read {} for the instance-ownership check: {error}",
                source.display()
            )
        });
        for (index, line) in text.lines().enumerate() {
            let code = line.split_once("//").map_or(line, |(before, _)| before);
            if code.contains("Instance::default") || code.contains("Instance::new(") {
                let relative = source.strip_prefix(&root).unwrap_or(source);
                sites.insert(format!("{}:{}", relative.display(), index + 1));
            }
        }
    }

    let foreign = sites
        .iter()
        .filter(|site| !site.starts_with(OWNER))
        .cloned()
        .collect::<BTreeSet<_>>();
    assert!(
        foreign.is_empty(),
        "Fix: {foreign:?} construct a wgpu instance outside {OWNER}. The Vulkan loader cannot start up twice at once; call runtime::device::acquire::new_instance instead."
    );
    assert_eq!(
        sites.len(),
        1,
        "Fix: expected exactly one wgpu instance construction site in this crate, found {sites:?}."
    );
}

fn collect_rust_sources(directory: &Path, into: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(directory).unwrap_or_else(|error| {
        panic!(
            "Fix: could not walk {} for the instance-ownership check: {error}",
            directory.display()
        )
    });
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: could not read an entry of {} for the instance-ownership check: {error}",
                    directory.display()
                )
            })
            .path();
        if path.is_dir() {
            collect_rust_sources(&path, into);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            into.push(path);
        }
    }
}
