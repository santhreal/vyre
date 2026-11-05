//! WHY: registry membership, populated documents and signed labels do not establish
//! measured qualification. These contracts reject missing cells, changed source,
//! authentication failures, violated invariants, threshold failures and overlapping
//! corpora. Signatures authenticate reviewer statements; these tests do not establish
//! that a reviewer performed an independent experiment or selected adequate thresholds.

use super::*;
use ed25519_dalek::{Signer, SigningKey};

const SOURCE: &str = "git:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:dirty=false";

struct Fixture {
    root: tempfile::TempDir,
    policy: Policy,
    keys: [SigningKey; 2],
    records: Vec<Record>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn artifact(root: &Path, path: &str, bytes: &[u8]) -> RawArtifact {
    std::fs::write(root.join(path), bytes).unwrap();
    RawArtifact {
        path: path.into(),
        sha256: hex(&Sha256::digest(bytes)),
    }
}

fn step(root: &Path, path: &str, role: &str) -> ReproductionStep {
    ReproductionStep {
        command: vec!["qualification-fixture".into(), role.into()],
        exit_code: 0,
        raw_record: artifact(root, path, role.as_bytes()),
    }
}

fn fixture() -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let keys = [
        SigningKey::from_bytes(&[7; 32]),
        SigningKey::from_bytes(&[11; 32]),
    ];
    let policy = Policy {
        schema_version: VERSION,
        reviewers: keys
            .iter()
            .enumerate()
            .map(|(index, key)| {
                (
                    format!("reviewer-{index}"),
                    hex(key.verifying_key().as_bytes()),
                )
            })
            .collect(),
        training_inputs: [hex(&Sha256::digest(b"training corpus"))]
            .into_iter()
            .collect(),
        cells: ENGINEERING_AXES
            .iter()
            .map(|axis| CellPolicy {
                package: "fixture".into(),
                axis: (*axis).into(),
                targets: ["host".into()].into_iter().collect(),
                invariants: ["counterexamples".into()].into_iter().collect(),
                metrics: vec![MetricPolicy {
                    id: "duration".into(),
                    unit: "ns".into(),
                    lower: 1.0,
                    upper: 10.0,
                    minimum_samples: 3,
                }],
            })
            .collect(),
    };
    let policy_text = toml::to_string(&policy).unwrap();
    let records = (0..2)
        .map(|index| Record {
            schema_version: VERSION,
            policy_sha256: hex(&Sha256::digest(policy_text.as_bytes())),
            source_fingerprint: SOURCE.into(),
            host_identity: format!("host-{index}"),
            clean_checkout: true,
            frozen_unix: 1,
            disclosed_unix: 2,
            completed_unix: 3,
            holdouts: vec![artifact(
                root.path(),
                &format!("holdout-{index}"),
                format!("unseen input {index}").as_bytes(),
            )],
            reproductions: vec![Reproduction {
                target: "host".into(),
                published_archive_build: step(
                    root.path(),
                    &format!("archive-{index}"),
                    "archive-build",
                ),
                downstream_compilation: step(
                    root.path(),
                    &format!("application-{index}"),
                    "downstream-compilation",
                ),
                authenticated_execution: step(
                    root.path(),
                    &format!("execution-{index}"),
                    "authenticated-execution",
                ),
                failure_replay: step(root.path(), &format!("replay-{index}"), "failure-replay"),
                regenerated_documents: step(
                    root.path(),
                    &format!("documents-{index}"),
                    "document-regeneration",
                ),
            }],
            cells: policy
                .cells
                .iter()
                .map(|cell| CellMeasurement {
                    package: cell.package.clone(),
                    axis: cell.axis.clone(),
                    target: "host".into(),
                    invariant_violations: [("counterexamples".into(), 0)].into_iter().collect(),
                    metrics: vec![Measurement {
                        id: "duration".into(),
                        unit: "ns".into(),
                        minimum: 1.0,
                        maximum: 10.0,
                        samples: 3,
                    }],
                })
                .collect(),
        })
        .collect();
    Fixture {
        root,
        policy,
        keys,
        records,
    }
}

impl Fixture {
    fn signed(&self) -> Vec<SignedRecord> {
        assert_eq!(self.records.len(), self.keys.len());
        self.records
            .iter()
            .zip(&self.keys)
            .enumerate()
            .map(|(index, (record, key))| {
                let payload = serde_json::to_string(record).unwrap();
                SignedRecord {
                    reviewer: format!("reviewer-{index}"),
                    signature: hex(&key.sign(payload.as_bytes()).to_bytes()),
                    payload,
                }
            })
            .collect()
    }

    fn validate(&self) -> Result<(), String> {
        validate_policy(&self.policy, &["fixture".into()])?;
        validate_records(
            self.root.path(),
            &self.policy,
            &toml::to_string(&self.policy).unwrap(),
            &self.signed(),
            |source| {
                if source == SOURCE {
                    Ok(())
                } else {
                    Err("source differs from frozen source".into())
                }
            },
        )
    }
}

#[test]
fn complete_authenticated_boundary_measurements_pass() {
    fixture().validate().unwrap();
}

#[test]
fn no_policy_and_no_records_cannot_qualify() {
    let fixture = fixture();
    let assessment = assess(fixture.root.path(), &["fixture".into()]);
    assert!(assessment.passed.is_empty());
    assert!(assessment.reason.unwrap().contains(POLICY_PATH));
    assert!(validate_records(
        fixture.root.path(),
        &fixture.policy,
        &toml::to_string(&fixture.policy).unwrap(),
        &[],
        |_| Ok(())
    )
    .is_err());
}

#[test]
fn every_live_package_and_axis_requires_an_explicit_policy() {
    let mut fixture = fixture();
    assert!(validate_policy(&fixture.policy, &[]).is_err());
    assert!(validate_policy(&fixture.policy, &["fixture".into(), "new-package".into()]).is_err());
    for index in 0..fixture.policy.cells.len() {
        let removed = fixture.policy.cells.remove(index);
        assert!(
            validate_policy(&fixture.policy, &["fixture".into()]).is_err(),
            "{}",
            removed.axis
        );
        fixture.policy.cells.insert(index, removed);
    }
    let duplicate: CellPolicy =
        serde_json::from_str(&serde_json::to_string(&fixture.policy.cells[0]).unwrap()).unwrap();
    fixture.policy.cells.push(duplicate);
    assert!(validate_policy(&fixture.policy, &["fixture".into()]).is_err());
}

#[test]
fn every_axis_rejects_missing_failed_and_undermeasured_cells() {
    for index in 0..ENGINEERING_AXES.len() {
        for variant in 0..9 {
            let mut fixture = fixture();
            let measured = &mut fixture.records[0].cells[index];
            match variant {
                0 => {
                    measured
                        .invariant_violations
                        .insert("counterexamples".into(), 1);
                }
                1 => measured.invariant_violations.clear(),
                2 => measured.metrics.clear(),
                3 => measured.metrics[0].minimum = 0.0,
                4 => measured.metrics[0].maximum = 11.0,
                5 => measured.metrics[0].samples = 2,
                6 => measured.metrics[0].unit = "seconds".into(),
                7 => measured.metrics[0].id = "unregistered".into(),
                8 => {
                    fixture.records[0].cells.remove(index);
                }
                _ => unreachable!(),
            }
            assert!(
                fixture.validate().is_err(),
                "axis={} variant={variant}",
                ENGINEERING_AXES[index]
            );
        }
    }
}

#[test]
fn nonfinite_and_reversed_measurements_never_pass() {
    let mut fixture = fixture();
    for (minimum, maximum) in [
        (f64::NAN, 10.0),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 10.0),
        (5.0, 4.0),
    ] {
        fixture.records[0].cells[0].metrics[0].minimum = minimum;
        fixture.records[0].cells[0].metrics[0].maximum = maximum;
        assert!(validate_measurements(&fixture.policy, &fixture.records[0].cells).is_err());
    }
}

#[test]
fn stale_source_schema_policy_and_disclosure_order_are_rejected() {
    for variant in 0..8 {
        let mut fixture = fixture();
        let record = &mut fixture.records[0];
        match variant {
            0 => record.schema_version += 1,
            1 => record.policy_sha256 = "00".repeat(32),
            2 => {
                record.source_fingerprint =
                    "git:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:dirty=false".into()
            }
            3 => record.clean_checkout = false,
            4 => {
                record.source_fingerprint =
                    "git:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:dirty=true".into()
            }
            5 => record.frozen_unix = record.disclosed_unix,
            6 => record.completed_unix = record.disclosed_unix - 1,
            7 => record.holdouts.clear(),
            _ => unreachable!(),
        }
        assert!(fixture.validate().is_err(), "variant={variant}");
    }
}

#[test]
fn signatures_require_trusted_distinct_reviewers_and_hosts() {
    for variant in 0..5 {
        let mut fixture = fixture();
        if variant == 4 {
            fixture.records[1].host_identity = fixture.records[0].host_identity.clone();
        }
        let mut signed = fixture.signed();
        match variant {
            0 => signed[0].signature = "00".repeat(64),
            1 => signed[0].payload.push(' '),
            2 => signed[0].reviewer = "untrusted".into(),
            3 => {
                signed.remove(1);
            }
            4 => {}
            _ => unreachable!(),
        }
        assert!(
            validate_records(
                fixture.root.path(),
                &fixture.policy,
                &toml::to_string(&fixture.policy).unwrap(),
                &signed,
                |_| Ok(())
            )
            .is_err(),
            "variant={variant}"
        );
    }
    let mut fixture = fixture();
    fixture.policy.reviewers.insert(
        "reviewer-1".into(),
        fixture.policy.reviewers["reviewer-0"].to_uppercase(),
    );
    assert!(validate_policy(&fixture.policy, &["fixture".into()]).is_err());
}

#[test]
fn holdouts_must_be_disjoint_and_raw_records_must_match() {
    for variant in 0..5 {
        let mut fixture = fixture();
        match variant {
            0 => {
                fixture.records[0].holdouts[0] =
                    artifact(fixture.root.path(), "leaked", b"training corpus");
            }
            1 => {
                let duplicate: RawArtifact = serde_json::from_str(
                    &serde_json::to_string(&fixture.records[0].holdouts[0]).unwrap(),
                )
                .unwrap();
                fixture.records[0].holdouts.push(duplicate);
            }
            2 => fixture.records[0].holdouts[0].sha256 = "00".repeat(32),
            3 => fixture.records[0].holdouts[0].path = "../outside".into(),
            4 => {
                std::fs::write(
                    fixture
                        .root
                        .path()
                        .join(&fixture.records[0].holdouts[0].path),
                    b"changed",
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        assert!(fixture.validate().is_err(), "variant={variant}");
    }
}

#[test]
fn every_reproduction_role_and_target_is_required() {
    for role in 0..5 {
        let mut fixture = fixture();
        let reproduction = &mut fixture.records[0].reproductions[0];
        let artifact = match role {
            0 => &mut reproduction.published_archive_build,
            1 => &mut reproduction.downstream_compilation,
            2 => &mut reproduction.authenticated_execution,
            3 => &mut reproduction.failure_replay,
            4 => &mut reproduction.regenerated_documents,
            _ => unreachable!(),
        };
        artifact.raw_record.sha256 = "00".repeat(32);
        assert!(fixture.validate().is_err(), "role={role}");
    }
    let mut fixture = fixture();
    fixture.records[0].reproductions.clear();
    assert!(fixture.validate().is_err());
}

#[test]
fn raw_record_budget_is_aggregate_and_cannot_wrap() {
    let mut remaining = 10;
    charge_raw_bytes(&mut remaining, 4).unwrap();
    charge_raw_bytes(&mut remaining, 6).unwrap();
    assert_eq!(remaining, 0);
    assert!(charge_raw_bytes(&mut remaining, 1).is_err());
    assert_eq!(remaining, 0);
    assert!(charge_raw_bytes(&mut remaining, u64::MAX).is_err());
}

#[test]
fn invalid_hex_and_unknown_fields_are_rejected() {
    for value in ["", "0", "zz", "é", "0000"] {
        assert!(decode_hex::<1>(value).is_err());
    }
    assert_eq!(decode_hex::<2>("01aF").unwrap(), [1, 175]);
    let fixture = fixture();
    let mut value = serde_json::to_value(&fixture.records[0]).unwrap();
    value["unverified_shortcut"] = serde_json::json!(true);
    assert!(serde_json::from_value::<Record>(value).is_err());
}

#[test]
fn duplicate_invariant_keys_cannot_hide_a_violation() {
    let text = r#"{"package":"fixture","axis":"semantic_correctness","target":"host","invariant_violations":{"counterexamples":1,"counterexamples":0},"metrics":[]}"#;
    assert!(serde_json::from_str::<CellMeasurement>(text).is_err());
}

#[test]
fn failed_or_unrecorded_commands_cannot_qualify_any_reproduction_role() {
    for role in 0..5 {
        for variant in 0..3 {
            let mut fixture = fixture();
            let reproduction = &mut fixture.records[0].reproductions[0];
            let step = match role {
                0 => &mut reproduction.published_archive_build,
                1 => &mut reproduction.downstream_compilation,
                2 => &mut reproduction.authenticated_execution,
                3 => &mut reproduction.failure_replay,
                4 => &mut reproduction.regenerated_documents,
                _ => unreachable!(),
            };
            match variant {
                0 => step.exit_code = 1,
                1 => step.command.clear(),
                2 => step.command.push(String::new()),
                _ => unreachable!(),
            }
            assert!(fixture.validate().is_err(), "role={role} variant={variant}");
        }
    }
}

#[test]
fn every_policy_cell_requires_finite_thresholds_and_zero_tolerance_invariants() {
    for variant in 0..9 {
        let mut fixture = fixture();
        let cell = &mut fixture.policy.cells[0];
        match variant {
            0 => cell.targets.clear(),
            1 => cell.invariants.clear(),
            2 => cell.metrics.clear(),
            3 => cell.metrics[0].lower = f64::NAN,
            4 => cell.metrics[0].upper = f64::INFINITY,
            5 => cell.metrics[0].lower = 11.0,
            6 => cell.metrics[0].minimum_samples = 0,
            7 => cell.metrics[0].unit.clear(),
            8 => cell.metrics[0].id.clear(),
            _ => unreachable!(),
        }
        assert!(
            validate_policy(&fixture.policy, &["fixture".into()]).is_err(),
            "variant={variant}"
        );
    }
}
