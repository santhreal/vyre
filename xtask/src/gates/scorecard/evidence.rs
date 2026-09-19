//! Assessment of independently signed measurements against a registered policy.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Take};
use std::path::{Component, Path};

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::ENGINEERING_AXES;

pub(super) const POLICY_PATH: &str = "release/engineering-qualification.toml";
pub(super) const RECORDS_PATH: &str = "release/evidence/engineering/qualification.json";
const VERSION: u32 = 1;
const MAX_DOCUMENT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_RAW_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    schema_version: u32,
    reviewers: BTreeMap<String, String>,
    training_inputs: BTreeSet<String>,
    cells: Vec<CellPolicy>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CellPolicy {
    package: String,
    axis: String,
    targets: BTreeSet<String>,
    invariants: BTreeSet<String>,
    metrics: Vec<MetricPolicy>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MetricPolicy {
    id: String,
    unit: String,
    lower: f64,
    upper: f64,
    minimum_samples: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SignedRecord {
    reviewer: String,
    /// Exact UTF-8 bytes authenticated by the detached signature.
    payload: String,
    signature: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema_version: u32,
    policy_sha256: String,
    source_fingerprint: String,
    host_identity: String,
    clean_checkout: bool,
    frozen_unix: u64,
    disclosed_unix: u64,
    completed_unix: u64,
    holdouts: Vec<RawArtifact>,
    reproductions: Vec<Reproduction>,
    cells: Vec<CellMeasurement>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Reproduction {
    target: String,
    published_archive_build: ReproductionStep,
    downstream_compilation: ReproductionStep,
    authenticated_execution: ReproductionStep,
    failure_replay: ReproductionStep,
    regenerated_documents: ReproductionStep,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReproductionStep {
    command: Vec<String>,
    exit_code: i32,
    raw_record: RawArtifact,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CellMeasurement {
    package: String,
    axis: String,
    target: String,
    #[serde(deserialize_with = "unique_invariant_counts")]
    invariant_violations: BTreeMap<String, u64>,
    metrics: Vec<Measurement>,
}

fn unique_invariant_counts<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, u64>, D::Error> {
    struct Counts;
    impl<'de> serde::de::Visitor<'de> for Counts {
        type Value = BTreeMap<String, u64>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("unique invariant identities and violation counts")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut input: M,
        ) -> Result<Self::Value, M::Error> {
            let mut counts = BTreeMap::new();
            while let Some((name, count)) = input.next_entry::<String, u64>()? {
                if counts.insert(name, count).is_some() {
                    return Err(serde::de::Error::custom("duplicate invariant identity"));
                }
            }
            Ok(counts)
        }
    }
    deserializer.deserialize_map(Counts)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Measurement {
    id: String,
    unit: String,
    minimum: f64,
    maximum: f64,
    samples: u64,
}

pub(super) struct Assessment {
    pub passed: BTreeMap<String, usize>,
    pub reason: Option<String>,
}

/// Missing, stale, malformed, incomplete or unauthenticated evidence qualifies no cell.
pub(super) fn assess(root: &Path, packages: &[String]) -> Assessment {
    let result = (|| {
        let policy_text = read_document(root, POLICY_PATH)?;
        let policy: Policy = toml::from_str(&policy_text)
            .map_err(|error| format!("invalid {POLICY_PATH}: {error}"))?;
        validate_policy(&policy, packages)?;
        let records_text = read_document(root, RECORDS_PATH)?;
        let records: Vec<SignedRecord> = serde_json::from_str(&records_text)
            .map_err(|error| format!("invalid {RECORDS_PATH}: {error}"))?;
        let current = crate::source_provenance::capture(root)?;
        if !current.ends_with(":dirty=false") {
            return Err("qualification requires a clean source checkout".into());
        }
        let commit = crate::source_provenance::recorded_commit(&current)
            .ok_or("qualification source has no recorded commit")?;
        validate_records(root, &policy, &policy_text, &records, |source| {
            crate::source_provenance::resolves_against(root, source, commit)
        })?;
        Ok::<_, String>(
            packages
                .iter()
                .map(|name| (name.clone(), ENGINEERING_AXES.len()))
                .collect(),
        )
    })();
    match result {
        Ok(passed) => Assessment {
            passed,
            reason: None,
        },
        Err(reason) => Assessment {
            passed: BTreeMap::new(),
            reason: Some(reason),
        },
    }
}

fn read_document(root: &Path, relative: &str) -> Result<String, String> {
    let path = root.join(relative);
    let metadata = path
        .symlink_metadata()
        .map_err(|error| document_error(relative, &error))?;
    if !metadata.is_file() || metadata.len() > MAX_DOCUMENT_BYTES {
        return Err(format!(
            "{relative} must be a regular file within the qualification document limit"
        ));
    }
    let file = File::open(&path).map_err(|error| document_error(relative, &error))?;
    let mut reader: Take<File> = file.take(MAX_DOCUMENT_BYTES + 1);
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|error| document_error(relative, &error))?;
    if text.len() as u64 > MAX_DOCUMENT_BYTES {
        return Err(format!(
            "{relative} exceeds the qualification document limit"
        ));
    }
    Ok(text)
}

fn document_error(relative: &str, error: &std::io::Error) -> String {
    match error.kind() {
        std::io::ErrorKind::NotFound => format!("missing qualification document: {relative}"),
        kind => format!("cannot read qualification document {relative}: {kind:?}"),
    }
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn decode_hex<const N: usize>(value: &str) -> Result<[u8; N], String> {
    if value.len() != N * 2 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("expected {} hexadecimal bytes", N));
    }
    let mut result = [0; N];
    for (index, output) in result.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|error| format!("invalid hexadecimal byte: {error}"))?;
    }
    Ok(result)
}

fn validate_policy(policy: &Policy, packages: &[String]) -> Result<(), String> {
    if policy.schema_version != VERSION || packages.is_empty() {
        return Err(
            "qualification requires the current policy schema and a nonempty workspace".into(),
        );
    }
    let mut keys = BTreeSet::new();
    for (reviewer, key) in &policy.reviewers {
        let key = decode_hex::<32>(key)?;
        VerifyingKey::from_bytes(&key).map_err(|error| format!("invalid reviewer key: {error}"))?;
        if !nonempty(reviewer) || !keys.insert(key) {
            return Err(
                "qualification reviewers must have distinct identities and public keys".into(),
            );
        }
    }
    if keys.len() < 2 || policy.training_inputs.is_empty() {
        return Err("qualification requires two trusted independent reviewers and a registered training corpus".into());
    }
    for digest in &policy.training_inputs {
        decode_hex::<32>(digest)?;
    }
    let expected: BTreeSet<_> = packages
        .iter()
        .flat_map(|package| {
            ENGINEERING_AXES
                .iter()
                .map(move |axis| (package.as_str(), *axis))
        })
        .collect();
    let mut actual = BTreeSet::new();
    for cell in &policy.cells {
        if !actual.insert((cell.package.as_str(), cell.axis.as_str())) {
            return Err("qualification policy contains a duplicate package/axis cell".into());
        }
        if cell.targets.is_empty()
            || cell.invariants.is_empty()
            || cell.metrics.is_empty()
            || cell
                .targets
                .iter()
                .chain(&cell.invariants)
                .any(|value| !nonempty(value))
        {
            return Err(format!(
                "{} / {} lacks targets, zero-tolerance invariants or measured thresholds",
                cell.package, cell.axis
            ));
        }
        let mut ids = BTreeSet::new();
        for metric in &cell.metrics {
            if !nonempty(&metric.id)
                || !nonempty(&metric.unit)
                || !ids.insert(&metric.id)
                || !metric.lower.is_finite()
                || !metric.upper.is_finite()
                || metric.lower > metric.upper
                || metric.minimum_samples == 0
            {
                return Err(format!(
                    "{} / {} has an invalid measured threshold",
                    cell.package, cell.axis
                ));
            }
        }
    }
    if actual != expected {
        return Err("qualification policy must cover every live package and every declared axis exactly once".into());
    }
    Ok(())
}

fn validate_records(
    root: &Path,
    policy: &Policy,
    policy_text: &str,
    signed: &[SignedRecord],
    mut source_matches: impl FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let policy_digest: [u8; 32] = Sha256::digest(policy_text.as_bytes()).into();
    if signed.len() < 2 || signed.len() > policy.reviewers.len() {
        return Err("qualification requires distinct registered independent reproductions".into());
    }
    let training: BTreeSet<_> = policy
        .training_inputs
        .iter()
        .map(|value| decode_hex::<32>(value))
        .collect::<Result<_, _>>()?;
    let targets: BTreeSet<_> = policy
        .cells
        .iter()
        .flat_map(|cell| cell.targets.iter())
        .collect();
    let mut remaining_raw_bytes = MAX_RAW_BYTES;
    let mut reviewers = BTreeSet::new();
    let mut hosts = BTreeSet::new();
    let mut checked_files = BTreeMap::new();
    for signed_record in signed {
        let key = policy
            .reviewers
            .get(&signed_record.reviewer)
            .ok_or("untrusted qualification reviewer")?;
        let key = VerifyingKey::from_bytes(&decode_hex::<32>(key)?)
            .map_err(|error| format!("invalid reviewer key: {error}"))?;
        let signature = Signature::from_bytes(&decode_hex::<64>(&signed_record.signature)?);
        key.verify_strict(signed_record.payload.as_bytes(), &signature)
            .map_err(|error| {
                format!("qualification signature does not authenticate its payload: {error}")
            })?;
        let record: Record = serde_json::from_str(&signed_record.payload)
            .map_err(|error| format!("invalid qualification payload: {error}"))?;
        if record.schema_version != VERSION
            || decode_hex::<32>(&record.policy_sha256)? != policy_digest
        {
            return Err("qualification receipt uses a stale schema or different policy".into());
        }
        if !reviewers.insert(&signed_record.reviewer)
            || !nonempty(&record.host_identity)
            || !hosts.insert(record.host_identity.clone())
        {
            return Err("qualification requires distinct reviewers and reproduction hosts".into());
        }
        if !record.clean_checkout || !record.source_fingerprint.ends_with(":dirty=false") {
            return Err("qualification reproduction did not use a clean checkout".into());
        }
        source_matches(&record.source_fingerprint)?;
        if record.frozen_unix == 0
            || record.frozen_unix >= record.disclosed_unix
            || record.disclosed_unix > record.completed_unix
            || record.holdouts.is_empty()
        {
            return Err("qualification requires holdout disclosure after source freeze and before completion".into());
        }
        let mut holdouts = BTreeSet::new();
        for artifact in &record.holdouts {
            let digest =
                check_artifact(root, artifact, &mut checked_files, &mut remaining_raw_bytes)?;
            if training.contains(&digest) || !holdouts.insert(digest) {
                return Err(
                    "qualification holdouts overlap training inputs or duplicate each other".into(),
                );
            }
        }
        let mut reproduced = BTreeSet::new();
        for reproduction in &record.reproductions {
            let Reproduction {
                target,
                published_archive_build,
                downstream_compilation,
                authenticated_execution,
                failure_replay,
                regenerated_documents,
            } = reproduction;
            if !reproduced.insert(target) {
                return Err("duplicate qualification reproduction target".into());
            }
            for step in [
                published_archive_build,
                downstream_compilation,
                authenticated_execution,
                failure_replay,
                regenerated_documents,
            ] {
                if step.exit_code != 0
                    || step.command.is_empty()
                    || step.command.iter().any(|argument| !nonempty(argument))
                {
                    return Err("qualification reproduction requires successful recorded commands for every role".into());
                }
                check_artifact(
                    root,
                    &step.raw_record,
                    &mut checked_files,
                    &mut remaining_raw_bytes,
                )?;
            }
        }
        if targets != reproduced {
            return Err("qualification reproduction does not cover every claimed target".into());
        }
        validate_measurements(policy, &record.cells)?;
    }
    if reviewers.len() < 2 {
        return Err("qualification lacks two independent authenticated reproductions".into());
    }
    Ok(())
}

fn check_artifact(
    root: &Path,
    artifact: &RawArtifact,
    checked: &mut BTreeMap<String, [u8; 32]>,
    remaining: &mut u64,
) -> Result<[u8; 32], String> {
    let expected = decode_hex::<32>(&artifact.sha256)?;
    let actual = if let Some(actual) = checked.get(&artifact.path) {
        *actual
    } else {
        let relative = Path::new(&artifact.path);
        if relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
            || relative.as_os_str().is_empty()
        {
            return Err(
                "qualification raw records must use workspace-relative paths without traversal"
                    .into(),
            );
        }
        let root = root.canonicalize().map_err(|error| error.to_string())?;
        let path = root
            .join(relative)
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if !path.starts_with(&root) {
            return Err("qualification raw record escapes the workspace".into());
        }
        if !path
            .metadata()
            .map_err(|error| error.to_string())?
            .is_file()
        {
            return Err("qualification raw record must be a regular file".into());
        }
        let file = File::open(&path).map_err(|error| error.to_string())?;
        let metadata = file.metadata().map_err(|error| error.to_string())?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_RAW_BYTES {
            return Err(
                "qualification raw record must be a nonempty regular file within the size limit"
                    .into(),
            );
        }
        let mut reader = file.take(MAX_RAW_BYTES + 1);
        let mut hasher = Sha256::new();
        let mut buffer = [0; 65536];
        let mut total = 0_u64;
        loop {
            let count = reader
                .read(&mut buffer)
                .map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            total += count as u64;
            if total > MAX_RAW_BYTES {
                return Err("qualification raw record grew beyond the size limit".into());
            }
            charge_raw_bytes(remaining, count as u64)?;
            hasher.update(&buffer[..count]);
        }
        if total == 0 {
            return Err("qualification raw record became empty".into());
        }
        let digest = hasher.finalize().into();
        checked.insert(artifact.path.clone(), digest);
        digest
    };
    if actual != expected {
        return Err(format!(
            "qualification raw record digest mismatch: {}",
            artifact.path
        ));
    }
    Ok(actual)
}

fn charge_raw_bytes(remaining: &mut u64, count: u64) -> Result<(), String> {
    *remaining = remaining
        .checked_sub(count)
        .ok_or("qualification raw records exceed the aggregate byte limit")?;
    Ok(())
}

fn validate_measurements(policy: &Policy, measurements: &[CellMeasurement]) -> Result<(), String> {
    let expected: BTreeMap<_, _> = policy
        .cells
        .iter()
        .flat_map(|cell| {
            cell.targets.iter().map(move |target| {
                (
                    (cell.package.as_str(), cell.axis.as_str(), target.as_str()),
                    cell,
                )
            })
        })
        .collect();
    let mut seen = BTreeSet::new();
    for measured in measurements {
        let identity = (
            measured.package.as_str(),
            measured.axis.as_str(),
            measured.target.as_str(),
        );
        let cell = expected
            .get(&identity)
            .ok_or("qualification contains an undeclared package, axis or target")?;
        if !seen.insert(identity) {
            return Err("duplicate qualification measurement cell".into());
        }
        if !measured
            .invariant_violations
            .keys()
            .eq(cell.invariants.iter())
            || measured
                .invariant_violations
                .values()
                .any(|violations| *violations != 0)
        {
            return Err(format!(
                "{} / {} lacks an invariant result or violates a zero-tolerance invariant",
                cell.package, cell.axis
            ));
        }
        let thresholds: BTreeMap<_, _> = cell
            .metrics
            .iter()
            .map(|metric| (metric.id.as_str(), metric))
            .collect();
        let mut ids = BTreeSet::new();
        for observation in &measured.metrics {
            let threshold = thresholds
                .get(observation.id.as_str())
                .ok_or("qualification contains an undeclared metric")?;
            if !ids.insert(&observation.id)
                || observation.unit != threshold.unit
                || !observation.minimum.is_finite()
                || !observation.maximum.is_finite()
                || observation.minimum > observation.maximum
                || observation.minimum < threshold.lower
                || observation.maximum > threshold.upper
                || observation.samples < threshold.minimum_samples
            {
                return Err(format!(
                    "{} / {} fails measured threshold {}",
                    cell.package, cell.axis, threshold.id
                ));
            }
        }
        if ids.len() != cell.metrics.len() {
            return Err("qualification lacks a required measured threshold".into());
        }
    }
    if seen.len() != expected.len() {
        return Err("qualification lacks a package/axis/target measurement cell".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "evidence_tests.rs"]
mod tests;
