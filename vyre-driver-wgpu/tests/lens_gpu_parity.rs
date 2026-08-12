//! GPU-side lens parity: exercise the three conform lenses against the
//! wgpu backend on a live device. Complements the CPU-only
//! `lens_parity.rs` by adding the `cpu_vs_backend` and `fixpoint`
//! coverage the CPU-only harness can't exercise.

use vyre_conform::lens::{self, LensOutcome};
fn backend() -> &'static vyre_driver::BackendRegistration {
    vyre_driver::backend::backend_registration(vyre_driver_wgpu::WGPU_BACKEND_ID)
        .expect("Fix: WGPU driver crate must register its authenticated artifact target.")
}

#[test]
fn cpu_vs_backend_lens_every_eligible_op() {
    let be = backend();
    let mut failures: Vec<String> = Vec::new();
    let mut passed = 0usize;
    for entry in vyre_libs::fixture_catalog::all_entries() {
        match lens::cpu_vs_backend(&entry, be) {
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

#[test]
fn fixpoint_lens_every_registered_contract() {
    let be = backend();
    let mut failures: Vec<String> = Vec::new();
    for entry in vyre_libs::fixture_catalog::all_entries() {
        if vyre_libs::fixture_catalog::fixpoint_contract(entry.id).is_none() {
            continue;
        }
        match lens::fixpoint(&entry, be) {
            LensOutcome::Pass { cases } => {
                println!("  pass {} ({cases} cases)", entry.id);
            }
            LensOutcome::Fail { case_index, detail } => {
                failures.push(format!("{} case {case_index}: {detail}", entry.id));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "fixpoint lens failures:\n  - {}",
        failures.join("\n  - ")
    );
}
