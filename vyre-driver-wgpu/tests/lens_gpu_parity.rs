//! GPU-side reference parity over the canonical library operation view.

#![cfg(feature = "device-tests")]

use vyre_conform::lens::backend_parity;
use vyre_conform::lens::outcome::LensOutcome;
fn backend() -> &'static vyre_driver::BackendRegistration {
    vyre_driver::backend_registration(vyre_driver_wgpu::WGPU_BACKEND_ID)
        .expect("Fix: WGPU driver crate must register its authenticated artifact target.")
}

#[test]
fn cpu_vs_backend_lens_every_eligible_op() {
    let be = backend();
    let mut failures: Vec<String> = Vec::new();
    let mut passed = 0usize;
    for entry in vyre_libs::operation_catalog::fixture_entries() {
        match backend_parity::run(&entry, be) {
            LensOutcome::Pass { cases } => {
                passed += 1;
                println!("  pass {} ({cases} cases)", entry.id);
            }
            LensOutcome::Fail { case_index, detail } => {
                failures.push(format!("{} case {case_index}: {detail}", entry.id));
            }
        }
    }
    println!(
        "cpu_vs_backend: {passed} passed, 0 coverage gaps, {} failed",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "cpu_vs_backend lens failures:\n  - {}",
        failures.join("\n  - ")
    );
}

/// WHY: every security operation that needs a whole-grid fence must reach
/// WGPU as sequential dispatches and preserve the reference bytes. Deriving
/// the set from the live registry and IR makes a newly fenced security
/// operation fail here until the split path proves it too.
#[test]
fn security_grid_sync_lens_matches_reference_for_every_registered_operation() {
    let be = backend();
    let entries = vyre_libs::operation_catalog::fixture_entries()
        .filter(|entry| entry.id.starts_with("vyre-libs::security::"))
        .filter(|entry| {
            entry
                .program()
                .is_some_and(|program| vyre_driver::grid_sync::contains_grid_sync(&program))
        })
        .collect::<Vec<_>>();
    assert!(
        !entries.is_empty(),
        "Fix: the live security registry must contain grid-sync coverage"
    );
    for entry in entries {
        assert!(
            matches!(
                backend_parity::run(&entry, be),
                LensOutcome::Pass { cases } if cases > 0
            ),
            "Fix: `{}` must split GridSync into WGPU dispatch boundaries and match the reference",
            entry.id
        );
    }
}
