//! Does a shipped backend answer what the float lowering ledger claims?
//!
//! WHY: a row in `vyre-driver/float-lowering-decisions.toml` is a claim about
//! an implementation, and a claim nothing compares against the implementation
//! is a comment. Every case here constructs a real backend, so the suite is
//! admitted only where a device is.
//!
//! The backend set is the live registry rather than a list, so a driver added
//! to this build takes one of the two outcomes here rather than neither.
//!
//! Every case here opens a vendor device context on the thread it runs on. Run
//! the harness with `--test-threads=1`: the vendor userspace driver spins
//! forever inside its own thread-local destructor when several test threads in
//! one process each build and tear down a context, which reports as a job that
//! never ends rather than as a failure.

#![cfg(feature = "device-tests")]
#![forbid(unsafe_code)]

use vyre_driver::DispatchConfig;
use vyre_foundation::fp_parity::FloatLoweringMode;
use vyre_foundation::ir::UnOp;
use vyre_registry_link::backend::live_backend_registry;
use vyre_test_support::strict_float_programs::f32_multiply_add_program;

use crate::float_lowering::{
    f32_bytes, first_divergence, honored_or_refused_by_name, ledger_path, read_ledger, Lowering,
};

/// A backend that constructs here answers what its row records.
///
/// The row is a claim about the shipped implementation, so it is compared
/// against `honors_float_lowering` on a real backend wherever the device this
/// host carries admits one. A backend whose factory fails states why, and the
/// list is part of the failure message rather than a silent pass.
#[test]
fn a_backend_that_constructs_here_answers_its_recorded_decision() {
    let ledger = read_ledger();
    let mut verified = 0usize;
    let mut unconstructed = Vec::new();
    let mut disagreements = Vec::new();

    for registration in live_backend_registry().expect("the backend registry must be readable") {
        let decision = ledger.get(registration.id).unwrap_or_else(|| {
            panic!(
                "Fix: backend `{}` is registered with no row in {}",
                registration.id,
                ledger_path().display()
            )
        });
        let backend = match (registration.factory)() {
            Ok(backend) => backend,
            Err(error) => {
                unconstructed.push(format!("  {}: {error}", registration.id));
                continue;
            }
        };
        verified += 1;
        for mode in FloatLoweringMode::EVERY {
            let recorded = decision
                .lowers
                .get(mode.cache_label())
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "Fix: backend `{}` has no decision for mode `{}`",
                        registration.id,
                        mode.cache_label()
                    )
                });
            let answered = backend.honors_float_lowering(*mode);
            if !recorded.admits(answered) {
                disagreements.push(format!(
                    "  {} answers {answered} for mode `{}`, recorded as {recorded:?}",
                    registration.id,
                    mode.cache_label()
                ));
            }
            // A measured decision is taken once and remembered. One that is
            // re-taken per call would put a GPU round trip inside an admission
            // check, and a decision that changed between two calls on one
            // adapter would admit a dispatch the previous call refused.
            if recorded == Lowering::Measured && backend.honors_float_lowering(*mode) != answered {
                disagreements.push(format!(
                    "  {} answered two different things for mode `{}` on one adapter",
                    registration.id,
                    mode.cache_label()
                ));
            }
        }
    }

    assert!(
        disagreements.is_empty(),
        "Fix: a backend disagrees with {}:\n{}",
        ledger_path().display(),
        disagreements.join("\n")
    );
    assert!(
        verified > 0,
        "Fix: no registered backend constructed on this host, so no recorded decision was \
         checked against its implementation:\n{}",
        unconstructed.join("\n")
    );
    for line in &unconstructed {
        assert!(
            line.contains("Fix:"),
            "Fix: a backend that cannot be constructed must state the remediation: {line}"
        );
    }
}

/// No backend accepts an operation with no exact f32 expansion under the
/// strict mode, whatever it claims about the mode itself.
///
/// `Sin` has an exact expansion, so honoring the mode is a legitimate answer
/// for it and the pair test above accepts either outcome. `Exp2` has none, so
/// there is only one legal outcome and a backend that answers it under a
/// strict request returned approximate arithmetic to a caller who asked for
/// one rounding per operation. That is a wrong answer the caller cannot see,
/// which is why it is separated from the mode-level contract.
///
/// The backend set is the live registry, so a driver added to this build takes
/// the refusal or turns this red.
#[test]
fn no_backend_accepts_an_operation_with_no_exact_expansion_under_strict_lowering() {
    let program = f32_multiply_add_program(4, Some(UnOp::Exp2));
    let inputs = vec![
        f32_bytes(&[0.5, 1.25, -2.5, 3.75]),
        f32_bytes(&[1.000_244_2, 0.5, 2.0, -1.5]),
        f32_bytes(&[-1.0, 0.25, 0.5, -0.125]),
    ];
    let mut config = DispatchConfig::default();
    config.float_lowering = FloatLoweringMode::StrictIeee;

    let mut findings = Vec::new();
    for registration in live_backend_registry().expect("the backend registry must be readable") {
        let Ok(backend) = (registration.factory)() else {
            continue;
        };
        let named_operation = backend
            .honors_float_lowering(FloatLoweringMode::StrictIeee)
            .then_some("Exp2");
        findings.extend(honored_or_refused_by_name(
            registration.id,
            false,
            FloatLoweringMode::StrictIeee,
            named_operation,
            &backend.dispatch(&program, &inputs, &config),
        ));
    }

    assert!(
        findings.is_empty(),
        "Fix: an operation with no exact f32 expansion must be refused by name under strict \
         lowering:\n{}",
        findings.join("\n")
    );
}

/// A strict dispatch either matches the oracle bit for bit or is refused.
///
/// This is the whole contract of the mode: a backend that cannot deny its
/// target the contraction and the approximate transcendental has to say so, and
/// the refusal has to name both the mode and the backend so the caller knows
/// which of the two to change. A backend that answers with contracted
/// arithmetic under a strict request is the defect, and it passes every test
/// that only checks the mode was accepted.
///
/// The backend set is the live registry, so a driver added to this build takes
/// one of the two outcomes here rather than neither.
#[test]
fn a_strict_dispatch_matches_the_oracle_or_is_refused_by_name() {
    // `sin` makes the witness carry both halves of the strict mode: an
    // approximable transcendental to expand and a multiply-add not to contract.
    let program = f32_multiply_add_program(4, Some(UnOp::Sin));
    let inputs = vec![
        f32_bytes(&[0.5, 1.25, -2.5, 3.75]),
        f32_bytes(&[1.000_244_2, 0.5, 2.0, -1.5]),
        f32_bytes(&[-1.0, 0.25, 0.5, -0.125]),
    ];
    let mut config = DispatchConfig::default();
    config.float_lowering = FloatLoweringMode::StrictIeee;

    let expanded = vyre_foundation::fp_expansion::expand_strict_transcendentals(&program)
        .expect("Fix: strict float expansion must succeed for strict IEEE reference evaluation");
    let eval_program = expanded.as_ref().unwrap_or(&program);
    let expected = vec![vyre_test_support::hardware_oracle::run_eval_single(
        eval_program,
        inputs.clone(),
    )];
    let registry = live_backend_registry().expect("the backend registry must be readable");

    let mut findings = Vec::new();
    let mut judged = Vec::new();
    for registration in registry {
        let backend = match registration.acquire() {
            Ok(backend) => backend,
            Err(error) => {
                judged.push(format!("  {}: no device here ({error})", registration.id));
                continue;
            }
        };
        match backend.dispatch(&program, &inputs, &config) {
            Ok(outputs) => {
                if outputs == expected {
                    judged.push(format!(
                        "  {}: bit-identical to the oracle",
                        registration.id
                    ));
                } else {
                    findings.push(format!(
                        "  {} accepted the strict mode and answered different bits than the \
                         oracle:{}",
                        registration.id,
                        first_divergence(&expected, &outputs)
                    ));
                }
            }
            Err(error) => {
                let message = error.to_string();
                let names_mode = message.contains(FloatLoweringMode::StrictIeee.cache_label());
                let names_backend = message.contains(registration.id);
                if names_mode && names_backend {
                    judged.push(format!("  {}: refused by name", registration.id));
                } else {
                    findings.push(format!(
                        "  {} failed a strict dispatch without naming both the mode and the \
                         backend: {message}",
                        registration.id
                    ));
                }
            }
        }
    }

    assert!(
        findings.is_empty(),
        "Fix: a strict dispatch must match the oracle or be refused by name:\n{}\njudged:\n{}",
        findings.join("\n"),
        judged.join("\n")
    );
    eprintln!("strict lowering per backend:\n{}", judged.join("\n"));
}

/// For every (backend, mode) pair in the runtime registry, the backend either
/// honors the mode or refuses it by name with remediation. No pair is silently
/// accepted and ignored.
#[test]
fn every_backend_and_float_lowering_mode_pair_is_honored_or_refused_with_remediation() {
    let program = f32_multiply_add_program(4, Some(UnOp::Sin));
    let inputs = vec![
        f32_bytes(&[0.5, 1.25, -2.5, 3.75]),
        f32_bytes(&[1.000_244_2, 0.5, 2.0, -1.5]),
        f32_bytes(&[-1.0, 0.25, 0.5, -0.125]),
    ];
    let registry = live_backend_registry().expect("the backend registry must be readable");
    let mut findings = Vec::new();

    for registration in registry {
        let Ok(backend) = registration.acquire() else {
            continue;
        };
        for mode in FloatLoweringMode::EVERY {
            let mut config = DispatchConfig::default();
            config.float_lowering = *mode;
            // Only a mode that blocks contraction can refuse over an
            // operation, so only then is naming one part of the contract.
            let named_operation = mode.blocks_contraction().then_some("Sin");
            findings.extend(honored_or_refused_by_name(
                registration.id,
                backend.honors_float_lowering(*mode),
                *mode,
                named_operation,
                &backend.dispatch(&program, &inputs, &config),
            ));
        }
    }

    assert!(
        findings.is_empty(),
        "Fix: every (backend, mode) pair must be honored or refused with actionable error naming mode and backend:\n{}",
        findings.join("\n")
    );
}
