//! Does every backend the registry serves state what it does with each float
//! lowering mode?
//!
//! WHY: a `DispatchConfig` carries a `FloatLoweringMode`, and a backend that
//! does not lower the requested mode answered with the mode it does lower. That
//! returns contracted arithmetic under a request for one rounding per
//! operation: a wrong answer rather than a slow one, and one the caller cannot
//! see. The registry wrapper now refuses it, which only helps for a backend
//! whose decision somebody recorded.
//!
//! Both halves of the closure are derived at run time. The backend set comes
//! from the live registry and the linked driver crates, and the mode set from
//! `FloatLoweringMode::EVERY`, so a driver crate added without a decision and a
//! mode added without one both turn this red instead of inheriting whatever the
//! trait default happens to say.
//!
//! Whether a shipped backend answers what its row claims needs a device, so
//! that half is `float_lowering_device_decisions.rs` behind `device-tests`.
//! The decision's presence is checked here, in every lane.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre_driver::DispatchConfig;
use vyre_foundation::fp_parity::FloatLoweringMode;
use vyre_registry_link::backend::{linked_backend_sources, live_backend_registry, DECLARED_SOURCES};

use vyre_foundation::ir::UnOp;
use vyre_test_support::strict_float_programs::f32_multiply_add_program;

use crate::float_lowering::{backends_needing_a_decision, f32_bytes, ledger_path, read_ledger};
/// Every backend states a decision for every mode.
#[test]
fn every_backend_records_a_decision_for_every_float_lowering_mode() {
    let ledger = read_ledger();
    let modes: BTreeSet<String> = FloatLoweringMode::EVERY
        .iter()
        .map(|mode| mode.cache_label().to_string())
        .collect();

    let mut findings = Vec::new();
    for backend in backends_needing_a_decision() {
        let Some(decision) = ledger.get(&backend) else {
            findings.push(format!(
                "  backend `{backend}` has no row: state whether it lowers each of {modes:?}"
            ));
            continue;
        };
        let recorded: BTreeSet<String> = decision.lowers.keys().cloned().collect();
        for missing in modes.difference(&recorded) {
            findings.push(format!(
                "  backend `{backend}` has no decision for mode `{missing}`"
            ));
        }
        for unknown in recorded.difference(&modes) {
            findings.push(format!(
                "  backend `{backend}` records mode `{unknown}`, which no FloatLoweringMode names"
            ));
        }
        assert!(
            !decision.reason.trim().is_empty(),
            "Fix: backend `{backend}` states an empty reason for its float lowering decisions"
        );
    }

    if linked_backend_sources().len() == DECLARED_SOURCES.len() {
        let known = backends_needing_a_decision();
        for recorded in ledger.keys() {
            if !known.contains(recorded) {
                findings.push(format!(
                    "  row `{recorded}` names no backend this build links: delete it or link its driver"
                ));
            }
        }
    }

    assert!(
        findings.is_empty(),
        "Fix: {} does not close over the backends and modes this build carries:\n{}",
        ledger_path().display(),
        findings.join("\n")
    );
}

/// A backend that cannot honor strict IEEE lowering refuses compilation and
/// cache-key construction rather than producing a separate cache entry for
/// unhonored contracted code.
#[test]
fn unsupported_backend_refuses_strict_ieee_compilation_and_cache_key_generation() {
    let program = f32_multiply_add_program(4, Some(UnOp::Sin));
    let mut config = DispatchConfig::default();
    config.float_lowering = FloatLoweringMode::StrictIeee;

    #[cfg(feature = "cuda")]
    {
        let result = vyre_driver_cuda::codegen::program_to_ptx(&program, &config);
        assert!(
            result.is_err(),
            "Fix: CUDA PTX codegen must reject strict IEEE lowering rather than emitting contracted PTX"
        );
        let error = result.unwrap_err();
        assert!(
            error.contains("strict-ieee") && error.contains("Sin") && error.contains("Fix:"),
            "Fix: CUDA PTX codegen refusal must name the mode, operation, and corrective action: {error}"
        );
    }
}

/// For every registered backend, strict-IEEE lowering either produces the
/// expanded form or refuses with a diagnostic naming the operation.
///
/// Closure: dynamically enumerates `live_backend_registry()`.
/// Adding a backend without honoring strict IEEE or refusing with an actionable
/// diagnostic naming the operation turns this red.
#[test]
fn every_registered_backend_strict_ieee_lowering_honored_or_refused_naming_operation() {
    let program_with_sin = f32_multiply_add_program(4, Some(UnOp::Sin));
    let program_with_exp2 = f32_multiply_add_program(4, Some(UnOp::Exp2));
    let inputs = vec![
        f32_bytes(&[0.5, 1.25, -2.5, 3.75]),
        f32_bytes(&[1.000_244_2, 0.5, 2.0, -1.5]),
        f32_bytes(&[-1.0, 0.25, 0.5, -0.125]),
    ];
    let mut config = DispatchConfig::default();
    config.float_lowering = FloatLoweringMode::StrictIeee;

    let registry = live_backend_registry().expect("Fix: backend registry must be readable");
    let mut findings = Vec::new();

    for registration in registry {
        let backend = match registration.acquire() {
            Ok(backend) => backend,
            Err(_) => continue,
        };

        // Case 1: Program with expandable operation (Sin)
        let honors_strict = backend.honors_float_lowering(FloatLoweringMode::StrictIeee);
        match backend.dispatch(&program_with_sin, &inputs, &config) {
            Ok(_) => {
                if !honors_strict {
                    findings.push(format!(
                        "  backend `{}` returned Ok for strict-ieee with Sin, but honors_float_lowering returned false",
                        registration.id
                    ));
                }
            }
            Err(error) => {
                if honors_strict {
                    findings.push(format!(
                        "  backend `{}` failed strict-ieee dispatch with expandable Sin: {error}",
                        registration.id
                    ));
                } else {
                    let message = error.to_string();
                    let names_mode = message.contains(FloatLoweringMode::StrictIeee.cache_label())
                        || message.contains("strict IEEE");
                    let names_op = message.contains("Sin");
                    let names_backend = message.contains(registration.id);
                    let has_fix = message.contains("Fix:");
                    if !names_mode || !names_op || !names_backend || !has_fix {
                        findings.push(format!(
                            "  backend `{}` refused unhonored strict-ieee without naming mode, operation Sin, backend, and Fix: in error: {message}",
                            registration.id
                        ));
                    }
                }
            }
        }

        // Case 2: Program with unexpandable approximable operation (Exp2)
        // Every backend must refuse this under StrictIeee with a diagnostic naming "Exp2".
        match backend.dispatch(&program_with_exp2, &inputs, &config) {
            Ok(_) => {
                findings.push(format!(
                    "  backend `{}` silently accepted unexpandable operation Exp2 under strict-ieee mode",
                    registration.id
                ));
            }
            Err(error) => {
                let message = error.to_string();
                let names_mode = message.contains(FloatLoweringMode::StrictIeee.cache_label())
                    || message.contains("strict IEEE");
                let names_op = message.contains("Exp2");
                let has_fix = message.contains("Fix:");
                if !names_mode || !names_op || !has_fix {
                    findings.push(format!(
                        "  backend `{}` refused unexpandable Exp2 under strict-ieee without naming mode, Exp2, and Fix: in error: {message}",
                        registration.id
                    ));
                }
            }
        }
    }

    assert!(
        findings.is_empty(),
        "Fix: every registered backend must either honor strict-IEEE lowering with expanded form or refuse naming the operation:\n{}",
        findings.join("\n")
    );
}
