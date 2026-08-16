//! Shared duplicate-family report schema for dedup gates.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::hash::sha256_hex;

pub(crate) const DUPLICATE_FAMILY_SCHEMA_VERSION: u32 = 2;

/// One duplicate-detection run, as written to its JSON artifact.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicateFamilyReport {
    /// Artifact schema version, so a stale file is rejected.
    pub schema_version: u32,
    /// Command that regenerates this artifact.
    pub generator_command: String,
    /// Detector that produced the findings.
    pub detector_family: String,
    /// Number of families in `families`.
    pub family_count: usize,
    /// Every family the detector found.
    pub families: Vec<DuplicateFamilyFinding>,
}

/// One pair of subjects a detector judged to be the same family.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicateFamilyFinding {
    /// Stable id for the pair, so a finding can be tracked across runs.
    pub family_id: String,
    /// Detector that found it.
    pub detector: String,
    /// Severity band derived from `score`.
    pub severity: &'static str,
    /// Similarity in `0.0..=1.0`.
    pub score: f64,
    /// First subject.
    pub left: DuplicateSubject,
    /// Second subject.
    pub right: DuplicateSubject,
    /// Lane that owns the importing side.
    pub import_owner: String,
    /// Lane the import points at.
    pub import_target: String,
    /// How the score was computed and what to do about it.
    pub evidence: DuplicateEvidence,
}

/// One side of a duplicate finding, and how it was measured.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicateSubject {
    /// Operation id or source path naming the subject.
    pub id: String,
    /// Lane that owns it.
    pub owner_lane: String,
    /// Structural fingerprint, when the detector computes one.
    pub fingerprint: Option<String>,
    /// Token or node count, when the detector measures one.
    pub tokens: Option<usize>,
    /// Source size, when the detector measures one.
    pub bytes: Option<u64>,
}

/// How the similarity between the two subjects was computed and what to do.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicateEvidence {
    /// Metric the score came from.
    pub similarity_metric: &'static str,
    /// The metric read from the left subject.
    pub left_metric: String,
    /// The metric read from the right subject.
    pub right_metric: String,
    /// What resolving this duplicate requires.
    pub dedup_action: &'static str,
}

/// Assemble a report from a detector run, dropping repeated families.
pub fn duplicate_family_report(
    generator_command: &str,
    detector_family: &str,
    families: Vec<DuplicateFamilyFinding>,
) -> DuplicateFamilyReport {
    let families = deduplicate_families(families);
    DuplicateFamilyReport {
        schema_version: DUPLICATE_FAMILY_SCHEMA_VERSION,
        generator_command: generator_command.to_string(),
        detector_family: detector_family.to_string(),
        family_count: families.len(),
        families,
    }
}

fn deduplicate_families(families: Vec<DuplicateFamilyFinding>) -> Vec<DuplicateFamilyFinding> {
    let mut by_family_id = BTreeMap::<String, DuplicateFamilyFinding>::new();
    for finding in families {
        match by_family_id.get_mut(&finding.family_id) {
            Some(existing) => merge_duplicate_family(existing, finding),
            None => {
                by_family_id.insert(finding.family_id.clone(), finding);
            }
        }
    }
    by_family_id.into_values().collect()
}

fn merge_duplicate_family(existing: &mut DuplicateFamilyFinding, incoming: DuplicateFamilyFinding) {
    let detector = merged_detector_label(&existing.detector, &incoming.detector);
    if incoming.score > existing.score {
        *existing = incoming;
    }
    existing.detector = detector;
    existing.severity = duplicate_severity(existing.score);
}

fn merged_detector_label(left: &str, right: &str) -> String {
    if left == right {
        return left.to_string();
    }
    let mut detectors = left.split('+').chain(right.split('+')).collect::<Vec<_>>();
    detectors.sort_unstable();
    detectors.dedup();
    detectors.join("+")
}

/// Write `report` as pretty JSON, creating parent directories.
pub fn write_duplicate_report_json(
    path: &Path,
    report: &DuplicateFamilyReport,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create parent directory `{}`: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(report)
        .map_err(|error| format!("serialize duplicate family report: {error}"))?;
    fs::write(path, format!("{json}\n"))
        .map_err(|error| format!("write duplicate family report: {error}"))
}

/// Resolve the artifact path from a `--duplicate-report-json` argument.
///
/// A value that looks like another flag is rejected, because that means the
/// path was omitted and the next flag was swallowed as one.
pub fn duplicate_report_json_path(
    flag: &str,
    raw: Option<&str>,
    missing_message: &str,
) -> Result<PathBuf, String> {
    let Some(path) = raw else {
        return Err(missing_message.to_string());
    };
    if path.starts_with("--") {
        return Err(format!("{flag} requires a path, not another flag"));
    }
    Ok(PathBuf::from(path))
}

/// The command line that regenerates an artifact, recorded inside it.
pub fn duplicate_report_generator_command(prefix: &str, output_path: &Path) -> String {
    format!(
        "xtask {prefix} --duplicate-report-json {}",
        output_path.display()
    )
}

/// Blockers against a recorded artifact: bad JSON, wrong schema version, or a
/// generator command that does not match the one that would rebuild it.
pub fn validate_duplicate_family_report_artifact(
    bytes: &[u8],
    expected_generator_command: &str,
) -> Vec<String> {
    let mut blockers = Vec::new();
    let value = match serde_json::from_slice::<serde_json::Value>(bytes) {
        Ok(value) => value,
        Err(error) => {
            return vec![format!(
                "duplicate family artifact is not valid JSON: {error}"
            )];
        }
    };
    if value.get("schema_version").and_then(|raw| raw.as_u64())
        != Some(u64::from(DUPLICATE_FAMILY_SCHEMA_VERSION))
    {
        blockers.push(format!(
            "duplicate family artifact must use schema_version={DUPLICATE_FAMILY_SCHEMA_VERSION}"
        ));
    }
    if value.get("generator_command").and_then(|raw| raw.as_str())
        != Some(expected_generator_command)
    {
        blockers.push(
            "duplicate family artifact generator_command must match the release evidence command"
                .to_string(),
        );
    }
    let Some(families) = value.get("families").and_then(|raw| raw.as_array()) else {
        blockers.push("duplicate family artifact must contain a families array".to_string());
        return blockers;
    };
    if value.get("family_count").and_then(|raw| raw.as_u64()) != Some(families.len() as u64) {
        blockers
            .push("duplicate family artifact family_count must match families length".to_string());
    }
    for (index, family) in families.iter().enumerate() {
        let family_id = family
            .get("family_id")
            .and_then(|raw| raw.as_str())
            .unwrap_or_default();
        if !family_id.starts_with("duplicate-family:v1:") {
            blockers.push(format!(
                "duplicate family artifact family[{index}].family_id is missing or unstable"
            ));
        }
        if family
            .get("detector")
            .and_then(|raw| raw.as_str())
            .unwrap_or_default()
            .is_empty()
        {
            blockers.push(format!(
                "duplicate family artifact family[{index}].detector is missing"
            ));
        }
        for side in ["left", "right"] {
            let fingerprint = family
                .get(side)
                .and_then(|subject| subject.get("fingerprint"))
                .and_then(|raw| raw.as_str())
                .unwrap_or_default();
            if fingerprint.is_empty() {
                blockers.push(format!(
                    "duplicate family artifact family[{index}].{side}.fingerprint is missing"
                ));
            } else if !duplicate_subject_fingerprint_is_supported(fingerprint) {
                blockers.push(format!(
                    "duplicate family artifact family[{index}].{side}.fingerprint uses an unsupported namespace"
                ));
            }
        }
    }
    blockers
}

/// A distinguishing substring of every blocker
/// [`validate_duplicate_family_report_artifact`] can emit for a single drifted
/// family.
///
/// The validator is asserted twice: here, against itself, and in xtask-evidence,
/// against the artifact inspector that routes through it. The two lists were
/// written by hand and had already drifted apart, so one caller judged the
/// missing `detector` field and the other did not.
const DUPLICATE_FAMILY_BLOCKER_MARKERS: [&str; 6] = [
    "schema_version=2",
    "generator_command",
    "family[0].family_id",
    "family[0].detector",
    "left.fingerprint",
    "right.fingerprint",
];

/// Which duplicate-family findings the given blockers fail to report.
#[must_use]
pub fn missing_duplicate_family_blocker_markers(blockers: &[String]) -> Vec<&'static str> {
    DUPLICATE_FAMILY_BLOCKER_MARKERS
        .into_iter()
        .filter(|marker| !blockers.iter().any(|blocker| blocker.contains(marker)))
        .collect()
}

fn duplicate_subject_fingerprint_is_supported(fingerprint: &str) -> bool {
    fingerprint.starts_with("registered-op-ir-fingerprint:v1:")
}

pub(crate) fn duplicate_family_id(detector: &str, left: &str, right: &str) -> String {
    let (first, second) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    let material =
        format!("duplicate-family:v1\ndetector={detector}\nleft={first}\nright={second}\n");
    format!("duplicate-family:v1:{}", sha256_hex(material.as_bytes()))
}

/// Stable family id for a pair of registered operations.
pub fn registered_op_duplicate_family_id(left: &str, right: &str) -> String {
    duplicate_family_id("registered-op", left, right)
}

/// Describe one registered operation as a duplicate-finding subject.
pub fn registered_op_duplicate_subject(
    op_id: &str,
    fingerprint: &[u8],
    node_count: usize,
) -> DuplicateSubject {
    DuplicateSubject {
        id: op_id.to_string(),
        owner_lane: registered_op_owner_lane(op_id).to_string(),
        fingerprint: Some(registered_op_fingerprint(fingerprint)),
        tokens: Some(node_count),
        bytes: Some(fingerprint.len() as u64),
    }
}

fn registered_op_fingerprint(fingerprint: &[u8]) -> String {
    format!(
        "registered-op-ir-fingerprint:v1:{}",
        sha256_hex(fingerprint)
    )
}

/// Severity band for a similarity score.
pub fn duplicate_severity(score: f64) -> &'static str {
    if score >= 0.95 {
        "duplicate"
    } else if score >= 0.86 {
        "very_similar"
    } else if score >= 0.50 {
        "similar"
    } else {
        "related"
    }
}

/// Structural similarity for registered-op IR fingerprints.
///
/// The metric compares byte-bigram frequency vectors with cosine similarity,
/// so adjacent node-kind order matters instead of only set membership.
pub fn structural_similarity(a: &[u8], b: &[u8]) -> f64 {
    if a.len() < 4 || b.len() < 4 {
        return 0.0;
    }
    let a_bigrams = bigram_counts(a);
    let b_bigrams = bigram_counts(b);
    let mut dot = 0i64;
    let mut a_norm = 0i64;
    let mut b_norm = 0i64;
    for (bg, &ac) in &a_bigrams {
        let bc = b_bigrams.get(bg).copied().unwrap_or(0);
        dot += (ac as i64) * (bc as i64);
        a_norm += (ac as i64).pow(2);
    }
    for &bc in b_bigrams.values() {
        b_norm += (bc as i64).pow(2);
    }
    if a_norm == 0 || b_norm == 0 {
        return 0.0;
    }
    dot as f64 / ((a_norm as f64).sqrt() * (b_norm as f64).sqrt())
}

fn bigram_counts(bytes: &[u8]) -> HashMap<(u8, u8), u32> {
    let mut out: HashMap<(u8, u8), u32> = HashMap::new();
    for window in bytes.windows(2) {
        *out.entry((window[0], window[1])).or_insert(0) += 1;
    }
    out
}

/// The lane that owns a registered operation, derived from its id.
pub fn registered_op_owner_lane(op_id: &str) -> &'static str {
    if op_id.starts_with("vyre-primitives::hardware::") {
        "lower_emit"
    } else if op_id.starts_with("vyre-libs::graph::")
        || op_id.starts_with("vyre-libs::bitset::")
        || op_id.starts_with("vyre-libs::fixpoint::")
        || op_id.starts_with("vyre-libs::graph::")
        || op_id.starts_with("vyre-libs::dataflow::")
    {
        "graph_flow_compiler"
    } else if op_id.starts_with("vyre-libs::matching::")
        || op_id.starts_with("vyre-libs::text::")
        || op_id.starts_with("vyre-libs::nfa::")
        || op_id.starts_with("vyre-libs::scan::")
        || op_id.starts_with("vyre-libs::matching::")
    {
        "scan_automata"
    } else if op_id.starts_with("vyre-libs::parsing::")
        || op_id.starts_with("vyre-libs::parsing::")
    {
        "parser_frontend"
    } else if op_id.starts_with("vyre-libs::security::")
        || op_id.starts_with("vyre-libs::borrowck::")
        || op_id.starts_with("vyre-libs::rule::")
        || op_id.starts_with("vyre-libs::predicate::")
    {
        "security_dataflow"
    } else if op_id.starts_with("vyre-libs::math::")
        || op_id.starts_with("vyre-libs::reduce::")
        || op_id.starts_with("vyre-libs::hash::")
        || op_id.starts_with("vyre-libs::decode::")
        || op_id.starts_with("vyre-libs::label::")
        || op_id.starts_with("vyre-libs::math::")
        || op_id.starts_with("vyre-libs::nn::")
        || op_id.starts_with("vyre-libs::quant::")
        || op_id.starts_with("vyre-libs::hash::")
        || op_id.starts_with("vyre-libs::decode::")
    {
        "sparse_math_ai"
    } else if op_id.starts_with("vyre-libs::visual::")
        || op_id.starts_with("vyre-libs::visual::")
    {
        "product_dogfood"
    } else if op_id.starts_with("vyre-primitives::") {
        "foundation_optimizer"
    } else if op_id.starts_with("vyre-libs::") {
        "security_dataflow"
    } else {
        "op_matrix"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_family_id_is_pair_order_and_detector_stable() {
        assert_eq!(
            duplicate_family_id("registered-op", "right", "left"),
            duplicate_family_id("registered-op", "left", "right")
        );
        assert_ne!(
            duplicate_family_id("registered-op", "left", "right"),
            duplicate_family_id("other-detector", "left", "right")
        );
    }

    #[test]
    fn registered_op_duplicate_subject_uses_shared_owner_and_fingerprint() {
        let subject =
            registered_op_duplicate_subject("vyre-libs::scan::literal_set", &[1, 2, 3, 4], 17);

        assert_eq!(subject.owner_lane, "scan_automata");
        assert_eq!(subject.tokens, Some(17));
        assert_eq!(subject.bytes, Some(4));
        assert!(subject
            .fingerprint
            .as_deref()
            .is_some_and(|value| value.starts_with("registered-op-ir-fingerprint:v1:")));
    }

    #[test]
    fn duplicate_family_report_counts_families() {
        let finding = DuplicateFamilyFinding {
            family_id: registered_op_duplicate_family_id("left", "right"),
            detector: "registered-op".to_string(),
            severity: duplicate_severity(0.96),
            score: 0.96,
            left: registered_op_duplicate_subject("left", &[1, 2, 3], 3),
            right: registered_op_duplicate_subject("right", &[1, 2, 4], 3),
            import_owner: "op_matrix".to_string(),
            import_target: "op_matrix".to_string(),
            evidence: DuplicateEvidence {
                similarity_metric: "ordered-ir-word-equality",
                left_metric: "words=3".to_string(),
                right_metric: "words=3".to_string(),
                dedup_action: "reuse_registered_operation",
            },
        };

        let report = duplicate_family_report("xtask whats-similar", "registered-op", vec![finding]);

        assert_eq!(report.schema_version, DUPLICATE_FAMILY_SCHEMA_VERSION);
        assert_eq!(report.family_count, 1);
        assert_eq!(report.families[0].severity, "duplicate");
    }

    #[test]
    fn duplicate_family_report_merges_same_family_id() {
        let family_id = registered_op_duplicate_family_id("left", "right");
        let left = DuplicateSubject {
            id: "left".to_string(),
            owner_lane: "scan_automata".to_string(),
            fingerprint: Some("registered-op-ir-fingerprint:v1:left".to_string()),
            tokens: Some(10),
            bytes: Some(10),
        };
        let right = DuplicateSubject {
            id: "right".to_string(),
            owner_lane: "scan_automata".to_string(),
            fingerprint: Some("registered-op-ir-fingerprint:v1:right".to_string()),
            tokens: Some(11),
            bytes: Some(11),
        };
        let evidence = DuplicateEvidence {
            similarity_metric: "test",
            left_metric: "left".to_string(),
            right_metric: "right".to_string(),
            dedup_action: "extract_shared_module_or_import_existing_owner",
        };
        let report = duplicate_family_report(
            "xtask test",
            "registered-op",
            vec![
                DuplicateFamilyFinding {
                    family_id: family_id.clone(),
                    detector: "lego-audit:no-reinvention".to_string(),
                    severity: duplicate_severity(0.90),
                    score: 0.90,
                    left: left.clone(),
                    right: right.clone(),
                    import_owner: "scan_automata".to_string(),
                    import_target: "left".to_string(),
                    evidence: evidence.clone(),
                },
                DuplicateFamilyFinding {
                    family_id,
                    detector: "lego-audit:operand-shape".to_string(),
                    severity: duplicate_severity(0.96),
                    score: 0.96,
                    left,
                    right,
                    import_owner: "scan_automata".to_string(),
                    import_target: "left".to_string(),
                    evidence,
                },
            ],
        );

        assert_eq!(report.family_count, 1);
        assert_eq!(
            report.families[0].detector,
            "lego-audit:no-reinvention+lego-audit:operand-shape"
        );
        assert_eq!(report.families[0].score, 0.96);
        assert_eq!(report.families[0].severity, "duplicate");
    }

    #[test]
    fn duplicate_report_json_path_rejects_missing_and_next_flag() {
        assert_eq!(
            duplicate_report_json_path(
                "--duplicate-report-json",
                None,
                "--duplicate-report-json requires a path"
            ),
            Err("--duplicate-report-json requires a path".to_string())
        );
        assert_eq!(
            duplicate_report_json_path(
                "--duplicate-report-json",
                Some("--with-repo"),
                "--duplicate-report-json requires a path"
            ),
            Err("--duplicate-report-json requires a path, not another flag".to_string())
        );
        assert_eq!(
            duplicate_report_json_path(
                "--duplicate-report-json",
                Some("release/evidence/dedup/report.json"),
                "--duplicate-report-json requires a path"
            ),
            Ok(PathBuf::from("release/evidence/dedup/report.json"))
        );
    }

    #[test]
    fn duplicate_report_generator_command_includes_output_path() {
        assert_eq!(
            duplicate_report_generator_command(
                "whats-similar --all",
                Path::new("release/evidence/dedup/registered-op-duplicates.json")
            ),
            "xtask whats-similar --all --duplicate-report-json release/evidence/dedup/registered-op-duplicates.json"
        );
    }

    #[test]
    fn duplicate_family_artifact_validation_rejects_drift() {
        let blockers = validate_duplicate_family_report_artifact(
            br#"{"schema_version":1,"generator_command":"wrong","family_count":1,"families":[{"left":{},"right":{}}]}"#,
            "xtask whats-similar --all --duplicate-report-json release/evidence/dedup/registered-op-duplicates.json",
        );

        assert_eq!(
            missing_duplicate_family_blocker_markers(&blockers),
            Vec::<&str>::new(),
            "Fix: every drifted field must stay a visible blocker; got {blockers:?}"
        );
    }

    #[test]
    fn duplicate_family_artifact_validation_rejects_unknown_fingerprint_namespace() {
        let blockers = validate_duplicate_family_report_artifact(
            br#"{
  "schema_version": 2,
  "generator_command": "xtask whats-similar --all --duplicate-report-json release/evidence/dedup/registered-op-duplicates.json",
  "family_count": 1,
  "families": [
    {
      "family_id": "duplicate-family:v1:abc",
      "detector": "registered-op",
      "left": {"fingerprint": "unknown:v1:left"},
      "right": {"fingerprint": "registered-op-ir-fingerprint:v1:right"}
    }
  ]
}"#,
            "xtask whats-similar --all --duplicate-report-json release/evidence/dedup/registered-op-duplicates.json",
        );

        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("unsupported namespace")));
    }

    #[test]
    fn structural_similarity_is_order_sensitive() {
        assert_eq!(structural_similarity(&[1, 2, 3], &[1, 2, 3]), 0.0);
        assert!((structural_similarity(&[1, 2, 3, 4], &[1, 2, 3, 4]) - 1.0).abs() < 1e-12);
        assert!(structural_similarity(&[1, 2, 3, 4], &[4, 3, 2, 1]) < 1.0);
    }

    #[test]
    fn registered_op_owner_lane_classifies_major_namespaces() {
        assert_eq!(
            registered_op_owner_lane("vyre-libs::graph::csr_forward_traverse"),
            "graph_flow_compiler"
        );
        assert_eq!(
            registered_op_owner_lane("vyre-libs::dataflow::semi_naive_join"),
            "graph_flow_compiler"
        );
        assert_eq!(
            registered_op_owner_lane("vyre-libs::scan::literal_set"),
            "scan_automata"
        );
        assert_eq!(
            registered_op_owner_lane("vyre-libs::parsing::python312_lexer"),
            "parser_frontend"
        );
        assert_eq!(
            registered_op_owner_lane("vyre-libs::parsing::ssa_dominance_scan"),
            "parser_frontend"
        );
        assert_eq!(
            registered_op_owner_lane("vyre-libs::nn::softmax"),
            "sparse_math_ai"
        );
        assert_eq!(
            registered_op_owner_lane("vyre-libs::predicate::call_to"),
            "security_dataflow"
        );
        assert_eq!(
            registered_op_owner_lane("vyre-libs::visual::blur"),
            "product_dogfood"
        );
    }

    /// WHY: the severity bands are exact `>=` comparisons, and every duplicate
    /// finding is triaged by the band rather than the score. An off-by-one-epsilon
    /// boundary silently downgrades a duplicate to `very_similar`, which is the
    /// difference between a release blocker and a note. Pin each edge and the
    /// value just below it.
    #[test]
    fn severity_bands_are_closed_at_their_lower_edge() {
        for (score, expected) in [
            (1.0, "duplicate"),
            (0.95, "duplicate"),
            (0.949_999, "very_similar"),
            (0.86, "very_similar"),
            (0.859_999, "similar"),
            (0.50, "similar"),
            (0.499_999, "related"),
            (0.0, "related"),
        ] {
            assert_eq!(
                duplicate_severity(score),
                expected,
                "score {score} must be `{expected}`"
            );
        }
    }

    /// WHY: an inverted comparison would still pass a per-band spot check while
    /// reporting the least similar pairs as duplicates. Severity must never fall
    /// as the score rises.
    #[test]
    fn severity_never_weakens_as_the_score_rises() {
        let rank = |band| match band {
            "related" => 0,
            "similar" => 1,
            "very_similar" => 2,
            "duplicate" => 3,
            other => panic!("unclassified severity band `{other}`"),
        };
        let mut previous = 0;
        for step in 0..=100 {
            let current = rank(duplicate_severity(f64::from(step) / 100.0));
            assert!(current >= previous, "severity fell at score {step}/100");
            previous = current;
        }
        assert_eq!(previous, 3);
    }
}
