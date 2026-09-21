use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::tree_walk::{self, BUILD_OUTPUT_AND_VCS};

use super::records::{
    ObservedThresholdConst, ThresholdPolicyArtifact, ThresholdPolicyDocument,
    ThresholdPolicyEvidenceRow, ThresholdPolicyFinding, ThresholdPolicyTomlRow,
    THRESHOLD_POLICY_ARTIFACT, THRESHOLD_POLICY_OWNER_LANE, THRESHOLD_POLICY_SCHEMA_VERSION,
    THRESHOLD_POLICY_SOURCE, THRESHOLD_SUFFIXES,
};
use super::rules::read_text_bounded;

pub(crate) fn collect_threshold_policy(vyre_root: &Path) -> ThresholdPolicyArtifact {
    let observed = scan_threshold_constants(vyre_root);
    let mut findings = Vec::new();
    let mut blockers = Vec::new();
    let policy_path = vyre_root.join(THRESHOLD_POLICY_SOURCE);
    let document = match fs::read_to_string(&policy_path) {
        Ok(text) => match toml::from_str::<ThresholdPolicyDocument>(&text) {
            Ok(document) => Some(document),
            Err(error) => {
                blockers.push(format!(
                    "{} is not valid threshold policy TOML: {error}. Fix: repair the TOML schema before release.",
                    THRESHOLD_POLICY_SOURCE
                ));
                None
            }
        },
        Err(error) => {
            blockers.push(format!(
                "missing {}: {error}. Fix: add unit, provenance, config tier, override path, evidence link, and release rule for every threshold-shaped const.",
                THRESHOLD_POLICY_SOURCE
            ));
            None
        }
    };
    let Some(document) = document else {
        return ThresholdPolicyArtifact {
            schema_version: THRESHOLD_POLICY_SCHEMA_VERSION,
            source_manifest: THRESHOLD_POLICY_SOURCE,
            evidence_artifact: THRESHOLD_POLICY_ARTIFACT.to_string(),
            owner_lane: THRESHOLD_POLICY_OWNER_LANE.to_string(),
            threshold_const_count: observed.len(),
            registered_policy_count: 0,
            rows: Vec::new(),
            findings,
            blockers,
        };
    };
    if document.schema_version != THRESHOLD_POLICY_SCHEMA_VERSION {
        blockers.push(format!(
            "{} schema_version={} must be {THRESHOLD_POLICY_SCHEMA_VERSION}. Fix: update the threshold policy reader and manifest together.",
            THRESHOLD_POLICY_SOURCE, document.schema_version
        ));
    }
    if document.owner_lane != THRESHOLD_POLICY_OWNER_LANE {
        blockers.push(format!(
            "{} owner_lane `{}` must be `{THRESHOLD_POLICY_OWNER_LANE}`. Fix: keep threshold evidence under the hygiene/testing lane.",
            THRESHOLD_POLICY_SOURCE, document.owner_lane
        ));
    }
    if document.evidence_artifact != THRESHOLD_POLICY_ARTIFACT {
        blockers.push(format!(
            "{} evidence_artifact `{}` must be `{THRESHOLD_POLICY_ARTIFACT}`. Fix: point the policy at the generated hygiene sibling artifact.",
            THRESHOLD_POLICY_SOURCE, document.evidence_artifact
        ));
    }
    let mut observed_by_key = BTreeMap::new();
    for threshold in observed {
        observed_by_key.insert(threshold_key(&threshold.path, &threshold.name), threshold);
    }
    let mut policy_by_key = BTreeMap::new();
    for row in &document.threshold {
        let row_key = threshold_key(&row.path, &row.name);
        if let Some(previous) = policy_by_key.insert(row_key.clone(), row.clone()) {
            blockers.push(format!(
                "{} duplicates threshold policy key `{}` for ids `{}` and `{}`. Fix: keep exactly one row per path/name threshold.",
                THRESHOLD_POLICY_SOURCE, row_key, previous.id, row.id
            ));
        }
        validate_threshold_policy_row(row, &mut blockers);
    }
    let mut rows = Vec::new();
    for (key, threshold) in &observed_by_key {
        let Some(policy) = policy_by_key.get(key) else {
            findings.push(ThresholdPolicyFinding {
                path: threshold.path.clone(),
                line: threshold.line,
                name: threshold.name.clone(),
                finding: "unregistered-threshold-const".to_string(),
                fix: format!(
                    "Fix: add `{}`/`{}` to {} with unit, provenance, config_tier, override_path, evidence_link, and release_rule.",
                    threshold.path, threshold.name, THRESHOLD_POLICY_SOURCE
                ),
            });
            blockers.push(format!(
                "{}:{} threshold const `{}` is missing from {}. Fix: register its unit, provenance, config tier, override path, evidence link, and VX release rule.",
                threshold.path, threshold.line, threshold.name, THRESHOLD_POLICY_SOURCE
            ));
            continue;
        };
        rows.push(ThresholdPolicyEvidenceRow {
            id: policy.id.clone(),
            path: threshold.path.clone(),
            line: threshold.line,
            name: threshold.name.clone(),
            observed_value: threshold.value.clone(),
            unit: policy.unit.clone(),
            provenance: policy.provenance.clone(),
            config_tier: policy.config_tier.clone(),
            override_path: policy.override_path.clone(),
            evidence_link: policy.evidence_link.clone(),
            release_rule: policy.release_rule.clone(),
        });
    }
    for (key, policy) in &policy_by_key {
        if !observed_by_key.contains_key(key) {
            findings.push(ThresholdPolicyFinding {
                path: policy.path.clone(),
                line: 1,
                name: policy.name.clone(),
                finding: "stale-threshold-policy-row".to_string(),
                fix: format!(
                    "Fix: remove or update stale threshold policy row `{}` after moving the source constant.",
                    policy.id
                ),
            });
            blockers.push(format!(
                "{} row `{}` points at `{}`/`{}` but no matching threshold const was observed. Fix: update or remove the stale policy row.",
                THRESHOLD_POLICY_SOURCE, policy.id, policy.path, policy.name
            ));
        }
    }
    rows.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.line.cmp(&right.line))
            .then(left.name.cmp(&right.name))
    });
    ThresholdPolicyArtifact {
        schema_version: THRESHOLD_POLICY_SCHEMA_VERSION,
        source_manifest: THRESHOLD_POLICY_SOURCE,
        evidence_artifact: THRESHOLD_POLICY_ARTIFACT.to_string(),
        owner_lane: document.owner_lane,
        threshold_const_count: observed_by_key.len(),
        registered_policy_count: policy_by_key.len(),
        rows,
        findings,
        blockers,
    }
}

pub(crate) fn validate_threshold_policy_row(
    row: &ThresholdPolicyTomlRow,
    blockers: &mut Vec<String>,
) {
    for (field, value) in [
        ("id", row.id.as_str()),
        ("path", row.path.as_str()),
        ("name", row.name.as_str()),
        ("unit", row.unit.as_str()),
        ("provenance", row.provenance.as_str()),
        ("config_tier", row.config_tier.as_str()),
        ("override_path", row.override_path.as_str()),
        ("evidence_link", row.evidence_link.as_str()),
        ("release_rule", row.release_rule.as_str()),
    ] {
        if value.trim().is_empty() {
            blockers.push(format!(
                "{} row `{}` has blank {field}. Fix: every threshold policy row must carry unit, provenance, tier, override, evidence, and VX ownership.",
                THRESHOLD_POLICY_SOURCE, row.id
            ));
        }
    }
    if !matches!(row.config_tier.as_str(), "tier_a" | "tier_b" | "structural") {
        blockers.push(format!(
            "{} row `{}` uses config_tier `{}`. Fix: use `tier_a`, `tier_b`, or `structural`.",
            THRESHOLD_POLICY_SOURCE, row.id, row.config_tier
        ));
    }
    if row.config_tier == "tier_a"
        && !(row.override_path.contains("tool.toml") && row.override_path.contains("CLI"))
    {
        blockers.push(format!(
            "{} row `{}` is Tier A but override_path does not name tool.toml and CLI override behavior. Fix: record compiled default -> tool.toml -> CLI precedence.",
            THRESHOLD_POLICY_SOURCE, row.id
        ));
    }
    if row.config_tier == "tier_b" && !row.override_path.contains("TOML data") {
        blockers.push(format!(
            "{} row `{}` is Tier B but override_path does not name TOML data ownership. Fix: keep community/data thresholds out of CLI flags.",
            THRESHOLD_POLICY_SOURCE, row.id
        ));
    }
    if row.config_tier == "structural" && !row.override_path.contains("not operator configurable") {
        blockers.push(format!(
            "{} row `{}` is structural but override_path does not say `not operator configurable`. Fix: separate wire/ABI bounds from runtime knobs.",
            THRESHOLD_POLICY_SOURCE, row.id
        ));
    }
    if row.evidence_link != THRESHOLD_POLICY_ARTIFACT {
        blockers.push(format!(
            "{} row `{}` evidence_link `{}` must be `{THRESHOLD_POLICY_ARTIFACT}`.",
            THRESHOLD_POLICY_SOURCE, row.id, row.evidence_link
        ));
    }
    if row.release_rule != "VX-475" {
        blockers.push(format!(
            "{} row `{}` release_rule `{}` must be `VX-475`.",
            THRESHOLD_POLICY_SOURCE, row.id, row.release_rule
        ));
    }
}

pub(crate) fn scan_threshold_constants(vyre_root: &Path) -> Vec<ObservedThresholdConst> {
    let mut thresholds = Vec::new();
    for root in threshold_scan_roots(vyre_root) {
        if !root.exists() {
            continue;
        }
        for entry in tree_walk::pruned_by(&root, |name| {
            !BUILD_OUTPUT_AND_VCS.contains(&name) && name != "tests"
        }) {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
                continue;
            }
            let Ok(text) = read_text_bounded(path) else {
                thresholds.push(ObservedThresholdConst {
                    path: relative_to_vyre(vyre_root, path),
                    line: 1,
                    name: "unreadable-threshold-source".to_string(),
                    value: "unreadable".to_string(),
                });
                continue;
            };
            for (line_index, line) in text.lines().enumerate() {
                let Some((name, value)) = parse_threshold_const(line) else {
                    continue;
                };
                thresholds.push(ObservedThresholdConst {
                    path: relative_to_vyre(vyre_root, path),
                    line: line_index + 1,
                    name,
                    value,
                });
            }
        }
    }
    thresholds
}

pub(crate) fn threshold_scan_roots(vyre_root: &Path) -> Vec<PathBuf> {
    [
        "vyre-foundation/src/optimizer",
        "vyre-runtime/src/resident_work_queue",
        "vyre-driver-wgpu/src/runtime",
        "vyre-driver-wgpu/src/buffer",
    ]
    .iter()
    .map(|relative| vyre_root.join(relative))
    .collect()
}

pub(crate) fn parse_threshold_const(line: &str) -> Option<(String, String)> {
    let code = line.split("//").next().unwrap_or(line).trim();
    let const_index = code.find("const ")?;
    let rest = &code[const_index + "const ".len()..];
    let colon_index = rest.find(':')?;
    let name = rest[..colon_index].trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
        || !THRESHOLD_SUFFIXES
            .iter()
            .any(|suffix| name.ends_with(suffix))
    {
        return None;
    }
    let equals_index = rest.find('=')?;
    let value = rest[equals_index + 1..].split(';').next()?.trim();
    if !value.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some((name.to_string(), value.to_string()))
}

pub(crate) fn threshold_key(path: &str, name: &str) -> String {
    format!("{path}::{name}")
}

pub(crate) fn relative_to_vyre(vyre_root: &Path, path: &Path) -> String {
    path.strip_prefix(vyre_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
