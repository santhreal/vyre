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
//! What it does not catch: a backend whose device is absent here cannot be
//! constructed, so its recorded decision is checked against the shipped
//! implementation only on a host that has the device. The decision's presence
//! is checked everywhere.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use vyre_driver::DispatchConfig;
use vyre_foundation::fp_parity::FloatLoweringMode;
use vyre_foundation::ir::UnOp;
use vyre_registry_link::backend::{
    linked_backend_sources, live_backend_registry, DECLARED_SOURCES,
};
use vyre_test_support::strict_float_programs::f32_multiply_add_program;

/// What a row claims a backend does with one mode.
///
/// A backend whose answer is a property of its emitter is `Always` or `Never`
/// and the row states which. A backend whose answer is a property of the device
/// in front of it cannot be pinned to either: the wgpu driver hands the platform
/// a contraction-free module and the platform compiles it again, so whether the
/// strict mode survives is measured on the adapter. `Measured` records that the
/// decision is taken at run time, which is a different claim from "either
/// answer is acceptable": the strict-dispatch contract below still holds the
/// backend to matching the oracle or refusing by name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lowering {
    Always,
    Never,
    Measured,
}

impl Lowering {
    /// Whether `answered` is admissible under this claim.
    fn admits(self, answered: bool) -> bool {
        match self {
            Self::Always => answered,
            Self::Never => !answered,
            Self::Measured => true,
        }
    }
}

/// One backend's row: which modes it lowers, and why.
struct Decision {
    lowers: BTreeMap<String, Lowering>,
    reason: String,
}

fn ledger_path() -> PathBuf {
    structure_gate::workspace_root().join("vyre-driver/float-lowering-decisions.toml")
}

fn read_ledger() -> BTreeMap<String, Decision> {
    let path = ledger_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", path.display()));
    let table: toml::Value = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: cannot parse {}: {error}", path.display()));
    let rows = table
        .as_table()
        .unwrap_or_else(|| panic!("Fix: {} must be a table of backend rows", path.display()));
    rows.iter()
        .map(|(backend, row)| {
            let row = row.as_table().unwrap_or_else(|| {
                panic!(
                    "Fix: {} row `{backend}` must be a table of mode decisions",
                    path.display()
                )
            });
            let reason = row
                .get("reason")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| {
                    panic!(
                        "Fix: {} row `{backend}` must state a `reason` for its decisions",
                        path.display()
                    )
                })
                .to_string();
            let lowers = row
                .iter()
                .filter(|(key, _)| key.as_str() != "reason")
                .map(|(mode, value)| {
                    let lowers = match value {
                        toml::Value::Boolean(true) => Lowering::Always,
                        toml::Value::Boolean(false) => Lowering::Never,
                        toml::Value::String(word) if word == "measured" => Lowering::Measured,
                        other => panic!(
                            "Fix: {} row `{backend}` key `{mode}` must be `true`, `false`, or the \
                             string \"measured\", not {other}",
                            path.display()
                        ),
                    };
                    (mode.clone(), lowers)
                })
                .collect();
            (backend.clone(), Decision { lowers, reason })
        })
        .collect()
}

/// Every backend id this build can be asked about: registered here, or owned by
/// a driver crate this build links whose registration is compiled out.
fn backends_needing_a_decision() -> BTreeSet<String> {
    let mut ids: BTreeSet<String> = linked_backend_sources()
        .iter()
        .map(|source| source.backend_id.to_string())
        .collect();
    for registration in live_backend_registry().expect("the backend registry must be readable") {
        ids.insert(registration.id.to_string());
    }
    ids
}

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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The first lane two answers disagree on, as bits and as values.
///
/// A strict mismatch is a rounding question, and "different bits" does not say
/// whether one rounding was lost to a contraction, a subnormal was flushed, or
/// an approximate instruction survived the expansion. The offending lane and
/// its two words separate those without another run.
fn first_divergence(expected: &[Vec<u8>], actual: &[Vec<u8>]) -> String {
    if expected.len() != actual.len() {
        return format!(
            " oracle produced {} buffer(s) and the backend {}",
            expected.len(),
            actual.len()
        );
    }
    for (index, (oracle, device)) in expected.iter().zip(actual).enumerate() {
        if oracle.len() != device.len() {
            return format!(
                " buffer {index}: oracle {} byte(s), backend {} byte(s)",
                oracle.len(),
                device.len()
            );
        }
        for (lane, (left, right)) in oracle
            .chunks_exact(4)
            .zip(device.chunks_exact(4))
            .enumerate()
        {
            if left == right {
                continue;
            }
            let oracle_bits = u32::from_le_bytes([left[0], left[1], left[2], left[3]]);
            let device_bits = u32::from_le_bytes([right[0], right[1], right[2], right[3]]);
            return format!(
                " buffer {index} lane {lane}: oracle 0x{oracle_bits:08x} ({}) backend \
                 0x{device_bits:08x} ({})",
                f32::from_bits(oracle_bits),
                f32::from_bits(device_bits)
            );
        }
    }
    String::from(" no differing lane, so the buffers differ in trailing bytes")
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

    let registry = live_backend_registry().expect("the backend registry must be readable");
    let oracle = registry
        .iter()
        .find(|registration| registration.reference_oracle)
        .expect("Fix: the strict contract is stated against the reference oracle, which this build must link");
    let expected = oracle
        .acquire()
        .expect("Fix: the reference oracle must construct on every host")
        .dispatch(&program, &inputs, &config)
        .expect("Fix: the reference oracle lowers the strict mode and must answer this program");

    let mut findings = Vec::new();
    let mut judged = Vec::new();
    for registration in registry {
        if registration.reference_oracle {
            continue;
        }
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
        let backend = match registration.acquire() {
            Ok(backend) => backend,
            Err(_) => continue,
        };
        for mode in FloatLoweringMode::EVERY {
            let mut config = DispatchConfig::default();
            config.float_lowering = *mode;
            let honors = backend.honors_float_lowering(*mode);
            match backend.dispatch(&program, &inputs, &config) {
                Ok(_) => {
                    if !honors {
                        findings.push(format!(
                            "  backend `{}` returned Ok for mode `{}` but honors_float_lowering returned false",
                            registration.id,
                            mode.cache_label()
                        ));
                    }
                }
                Err(error) => {
                    if honors {
                        findings.push(format!(
                            "  backend `{}` failed dispatch for honored mode `{}`: {error}",
                            registration.id,
                            mode.cache_label()
                        ));
                    } else {
                        let message = error.to_string();
                        let names_mode = message.contains(mode.cache_label());
                        let names_backend = message.contains(registration.id);
                        let has_fix = message.contains("Fix:");
                        if !names_mode || !names_backend || !has_fix {
                            findings.push(format!(
                                "  backend `{}` refused unhonored mode `{}` without naming mode, backend, and Fix: in error: {message}",
                                registration.id,
                                mode.cache_label()
                            ));
                        }
                    }
                }
            }
        }
    }

    assert!(
        findings.is_empty(),
        "Fix: every (backend, mode) pair must be honored or refused with actionable error naming mode and backend:\n{}",
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
            error.contains("strict-ieee") && error.contains("Fix:"),
            "Fix: CUDA PTX codegen refusal must name the mode and corrective action: {error}"
        );
    }
}
