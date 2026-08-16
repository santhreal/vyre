//! Contracts for the process-wide shared wgpu backend and the one wgpu instance
//! every acquisition in this crate uses.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};

use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;

fn selected_adapter(backend: &WgpuBackend) -> wgpu::Adapter {
    vyre_driver_wgpu::runtime::adapter_for_info(backend.adapter_info()).expect(
        "Fix: selected wgpu backend adapter must remain enumerable for live capability probing",
    )
}

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

#[test]
fn shared_backend_reports_same_capabilities_as_concrete_backend() {
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
    assert_eq!(
        shared.max_workgroup_size(),
        [
            shared.device_limits().max_compute_workgroup_size_x,
            shared.device_limits().max_compute_workgroup_size_y,
            shared.device_limits().max_compute_workgroup_size_z,
        ],
        "Fix: shared backend max_workgroup_size must come from the live selected device limits"
    );
}

/// Acquiring a backend from many threads at once must succeed on every thread.
///
/// A `wgpu::Instance` owns the Vulkan loader instance and loader startup is not
/// reentrant, so two threads creating an instance together left one of them
/// calling through a null dispatch pointer inside
/// `vkEnumerateInstanceExtensionProperties`. That defect does not surface as a
/// failed assertion: the process dies with SIGSEGV and takes the harness with
/// it, so this test fails by disappearing. The barrier releases every thread
/// into acquisition at once, which is the state that faulted.
#[test]
fn concurrent_backend_acquisition_succeeds_on_every_thread() {
    let threads = std::thread::available_parallelism().map_or(8, |count| count.get().max(8));
    let barrier = Arc::new(Barrier::new(threads));
    let acquisitions = (0..threads)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                WgpuBackend::acquire().map(|backend| backend.adapter_info().name.clone())
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|handle| {
            handle
                .join()
                .expect("Fix: a concurrent wgpu acquisition thread must not panic")
        })
        .collect::<Vec<_>>();

    let adapters = acquisitions
        .iter()
        .map(|acquisition| match acquisition {
            Ok(name) => name.clone(),
            Err(error) => panic!(
                "Fix: concurrent wgpu acquisition failed on one of {threads} threads: {error}"
            ),
        })
        .collect::<BTreeSet<_>>();
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
    let root = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")).join("src");
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
