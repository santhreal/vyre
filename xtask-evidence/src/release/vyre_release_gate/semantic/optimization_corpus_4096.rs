use std::path::Path;

use crate::bench::benchmark_evidence_semantics::duplicate_nonblank_object_array_field_values;

use super::super::checks::*;
use super::super::gate_inputs::Requirement;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    const REQUIRED_FAMILIES: [&str; 8] = [
        "scalar-algebra",
        "strength-reduction",
        "fusion-cse",
        "dead-code",
        "memory-dataflow",
        "loop-transform",
        "control-flow",
        "canonicalization",
    ];

    let Some(corpus) =
        first_json_evidence(requirement, base_dir, "optimization-corpus.json", failures)
    else {
        return;
    };
    let generated = u64_field(&corpus, "generated_cases", 0);
    let verified = u64_field(&corpus, "verified_cases", 0);
    let optimized = u64_field(&corpus, "optimized_cases", 0);
    let non_converged = u64_field(&corpus, "non_converged_cases", u64::MAX);
    let pass_instances = u64_field(&corpus, "pass_instance_count", 0);
    let changed_pass_instances = u64_field(&corpus, "changed_pass_instances", 0);
    let required = u64_field(&corpus, "required_min_cases", 4_096);
    let blockers = array_len(&corpus, "blockers");

    if required < 4_096 {
        failures.push(format!(
            "requirement `optimization-corpus-4096` required_min_cases={required}; release floor is 4096"
        ));
    }
    if generated < required || generated < 4_096 {
        failures.push(format!(
            "requirement `optimization-corpus-4096` generated {generated} semantic Program cases; needs at least {required} and never below 4096"
        ));
    }
    if verified != generated {
        failures.push(format!(
            "requirement `optimization-corpus-4096` verified {verified}/{generated} generated semantic Programs"
        ));
    }
    if optimized == 0 || changed_pass_instances == 0 {
        failures.push(format!(
            "requirement `optimization-corpus-4096` reports {optimized} optimized cases and {changed_pass_instances} changed pass instances; the corpus is not proving live semantic optimizer execution"
        ));
    }
    if pass_instances < generated {
        failures.push(format!(
            "requirement `optimization-corpus-4096` reports {pass_instances} pass instances for {generated} cases"
        ));
    }
    if non_converged != 0 || blockers != 0 {
        failures.push(format!(
            "requirement `optimization-corpus-4096` reports {non_converged} non-converged case(s) and {blockers} blocker(s)"
        ));
    }

    for suffix in [
        "optimization-corpus-contracts.json",
        "optimization-family-manifest.json",
        "optimization-case-manifest.json",
        "optimizer-pass-manifest.json",
    ] {
        check_json_evidence_has_no_blockers(requirement, base_dir, suffix, failures);
    }

    if let Some(family_manifest) = first_json_evidence(
        requirement,
        base_dir,
        "optimization-family-manifest.json",
        failures,
    ) {
        let families = family_manifest
            .get("families")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let declared_required = u64_field(&family_manifest, "required_family_count", 0);
        let missing_required = array_len(&family_manifest, "missing_required_families");
        if families.len() != REQUIRED_FAMILIES.len()
            || declared_required != REQUIRED_FAMILIES.len() as u64
        {
            failures.push(format!(
                "requirement `optimization-corpus-4096` family manifest has {} row(s) and declares {declared_required}; needs exactly {} source-owned semantic families",
                families.len(),
                REQUIRED_FAMILIES.len()
            ));
        }
        if missing_required != 0 {
            failures.push(format!(
                "requirement `optimization-corpus-4096` family manifest reports {missing_required} missing required semantic family/families"
            ));
        }
        check_duplicate_rows(
            &family_manifest,
            "families",
            "family",
            "family manifest has duplicate family rows",
            failures,
        );
        for required_family in REQUIRED_FAMILIES {
            let cases = families
                .iter()
                .find(|family| {
                    family.get("family").and_then(serde_json::Value::as_str)
                        == Some(required_family)
                })
                .and_then(|family| family.get("cases").and_then(serde_json::Value::as_u64))
                .unwrap_or(0);
            if cases < 512 {
                failures.push(format!(
                    "requirement `optimization-corpus-4096` required semantic family `{required_family}` has {cases} generated case(s), needs at least 512"
                ));
            }
        }
    }

    if let Some(case_manifest) = first_json_evidence(
        requirement,
        base_dir,
        "optimization-case-manifest.json",
        failures,
    ) {
        let unique_case_ids = u64_field(&case_manifest, "unique_case_ids", 0);
        let manifest_generated = u64_field(&case_manifest, "generated_cases", 0);
        let entries = case_manifest
            .get("entries")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if manifest_generated != generated
            || unique_case_ids != generated
            || entries.len() as u64 != generated
        {
            failures.push(format!(
                "requirement `optimization-corpus-4096` case manifest generated={manifest_generated}, unique={unique_case_ids}, entries={}, corpus generated={generated}",
                entries.len()
            ));
        }
        check_duplicate_rows(
            &case_manifest,
            "entries",
            "id",
            "case manifest has duplicate entry ids",
            failures,
        );
        let malformed_entries = entries
            .iter()
            .filter(|entry| {
                entry
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                    || entry
                        .get("family")
                        .and_then(serde_json::Value::as_str)
                        .is_none_or(str::is_empty)
                    || u64_field(entry, "node_count", 0) == 0
                    || u64_field(entry, "instruction_count", 0) == 0
                    || entry
                        .get("program_fingerprint")
                        .and_then(serde_json::Value::as_str)
                        .is_none_or(|fingerprint| fingerprint.len() != 64)
            })
            .count();
        if malformed_entries != 0 {
            failures.push(format!(
                "requirement `optimization-corpus-4096` case manifest contains {malformed_entries} malformed semantic Program case(s)"
            ));
        }
    }

    if let Some(pass_manifest) = first_json_evidence(
        requirement,
        base_dir,
        "optimizer-pass-manifest.json",
        failures,
    ) {
        let executable_passes = u64_field(&pass_manifest, "executable_passes", 0);
        let catalog_entries = u64_field(&pass_manifest, "catalog_entries", 0);
        let entries = pass_manifest
            .get("entries")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if executable_passes == 0
            || catalog_entries < executable_passes
            || entries.len() as u64 != catalog_entries
        {
            failures.push(format!(
                "requirement `optimization-corpus-4096` optimizer pass manifest executable={executable_passes}, catalog={catalog_entries}, entries={}",
                entries.len()
            ));
        }
        check_duplicate_rows(
            &pass_manifest,
            "entries",
            "id",
            "optimizer pass manifest has duplicate ids",
            failures,
        );
        let malformed_entries = entries
            .iter()
            .filter(|entry| {
                [
                    "id",
                    "kind",
                    "owner",
                    "phase",
                    "boundary",
                    "invariant",
                    "termination",
                    "proof",
                ]
                .iter()
                .any(|field| {
                    entry
                        .get(*field)
                        .and_then(serde_json::Value::as_str)
                        .is_none_or(str::is_empty)
                }) || entry
                    .get("benchmark")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
            })
            .count();
        if malformed_entries != 0 {
            failures.push(format!(
                "requirement `optimization-corpus-4096` optimizer pass manifest contains {malformed_entries} row(s) without live ownership, proof invariant, or benchmark/non-benchmark disposition"
            ));
        }
    }
}

fn u64_field(value: &serde_json::Value, field: &str, default: u64) -> u64 {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(default)
}

fn array_len(value: &serde_json::Value, field: &str) -> usize {
    value
        .get(field)
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len)
}

fn check_duplicate_rows(
    manifest: &serde_json::Value,
    array_field: &str,
    value_field: &str,
    label: &str,
    failures: &mut Vec<String>,
) {
    let duplicates =
        duplicate_nonblank_object_array_field_values(manifest, array_field, value_field);
    if !duplicates.is_empty() {
        let duplicates = duplicates.into_iter().collect::<Vec<_>>().join(", ");
        failures.push(format!(
            "requirement `optimization-corpus-4096` {label}: {duplicates}"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimization_corpus_gate_rejects_duplicate_family_rows() {
        let family_manifest = serde_json::json!({
            "required_family_count": 8,
            "missing_required_families": [],
            "families": [
                {"family": "scalar-algebra", "cases": 512},
                {"family": "scalar-algebra", "cases": 512}
            ],
            "blockers": []
        });
        let failures = run_optimization_corpus_gate_with_manifests(
            family_manifest,
            valid_case_manifest(),
            valid_pass_manifest(),
        );

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("duplicate family rows: scalar-algebra")),
            "Fix: optimization corpus gate must reject duplicate family manifest rows; failures={failures:?}"
        );
    }

    #[test]
    fn optimization_corpus_gate_rejects_duplicate_case_entry_ids() {
        let mut case_manifest = valid_case_manifest();
        case_manifest["entries"] = serde_json::json!([
            {
                "id": "foundation.optimizer.scalar-algebra.0001",
                "family": "scalar-algebra",
                "node_count": 1,
                "instruction_count": 3,
                "memory_op_count": 1,
                "control_flow_count": 0,
                "program_fingerprint": "0".repeat(64)
            },
            {
                "id": "foundation.optimizer.scalar-algebra.0001",
                "family": "strength-reduction",
                "node_count": 1,
                "instruction_count": 5,
                "memory_op_count": 1,
                "control_flow_count": 0,
                "program_fingerprint": "1".repeat(64)
            }
        ]);
        let failures = run_optimization_corpus_gate_with_manifests(
            valid_family_manifest(),
            case_manifest,
            valid_pass_manifest(),
        );

        assert!(
            failures.iter().any(|failure| {
                failure
                    .contains("duplicate entry ids: foundation.optimizer.scalar-algebra.0001")
            }),
            "Fix: optimization corpus gate must reject duplicate semantic Program case ids; failures={failures:?}"
        );
    }

    #[test]
    fn optimization_corpus_gate_rejects_pass_rows_without_benchmark_disposition() {
        let mut pass_manifest = valid_pass_manifest();
        pass_manifest["entries"][0]["benchmark"] = serde_json::json!("");
        let failures = run_optimization_corpus_gate_with_manifests(
            valid_family_manifest(),
            valid_case_manifest(),
            pass_manifest,
        );

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("benchmark/non-benchmark disposition")),
            "Fix: every optimizer pass row needs a benchmark or reviewed non-benchmark disposition; failures={failures:?}"
        );
    }

    fn run_optimization_corpus_gate_with_manifests(
        family_manifest: serde_json::Value,
        case_manifest: serde_json::Value,
        pass_manifest: serde_json::Value,
    ) -> Vec<String> {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temp workspace for optimization corpus gate test.");
        let base_dir = dir.path();
        std::fs::write(
            base_dir.join("optimization-corpus.json"),
            serde_json::json!({
                "required_min_cases": 4096,
                "generated_cases": 4096,
                "verified_cases": 4096,
                "optimized_cases": 4096,
                "non_converged_cases": 0,
                "pass_instance_count": 8192,
                "changed_pass_instances": 4096,
                "blockers": []
            })
            .to_string(),
        )
        .expect("Fix: write optimization corpus gate fixture.");
        for (name, value) in [
            ("optimization-family-manifest.json", family_manifest),
            ("optimization-case-manifest.json", case_manifest),
            ("optimizer-pass-manifest.json", pass_manifest),
            (
                "optimization-corpus-contracts.json",
                serde_json::json!({"blockers": []}),
            ),
        ] {
            std::fs::write(base_dir.join(name), value.to_string())
                .expect("Fix: write optimization corpus gate fixture.");
        }
        let requirement = Requirement {
            id: "optimization-corpus-4096".to_string(),
            title: "optimization corpus".to_string(),
            status: "required".to_string(),
            evidence: [
                "optimization-corpus.json",
                "optimization-corpus-contracts.json",
                "optimization-family-manifest.json",
                "optimization-case-manifest.json",
                "optimizer-pass-manifest.json",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            minimum_evidence: 5,
        };
        let mut failures = Vec::new();

        check(&requirement, base_dir, &mut failures);

        failures
    }

    fn valid_family_manifest() -> serde_json::Value {
        serde_json::json!({
            "required_family_count": 8,
            "missing_required_families": [],
            "families": [
                {"family": "scalar-algebra", "cases": 512},
                {"family": "strength-reduction", "cases": 512},
                {"family": "fusion-cse", "cases": 512},
                {"family": "dead-code", "cases": 512},
                {"family": "memory-dataflow", "cases": 512},
                {"family": "loop-transform", "cases": 512},
                {"family": "control-flow", "cases": 512},
                {"family": "canonicalization", "cases": 512}
            ],
            "blockers": []
        })
    }

    fn valid_case_manifest() -> serde_json::Value {
        let entries = (0..4096)
            .map(|index| {
                serde_json::json!({
                    "id": format!("foundation.optimizer.case.{index:04}"),
                    "family": "scalar-algebra",
                    "node_count": 1,
                    "instruction_count": 3,
                    "memory_op_count": 1,
                    "control_flow_count": 0,
                    "program_fingerprint": "0".repeat(64)
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "generated_cases": 4096,
            "unique_case_ids": 4096,
            "entries": entries,
            "blockers": []
        })
    }

    fn valid_pass_manifest() -> serde_json::Value {
        serde_json::json!({
            "executable_passes": 1,
            "catalog_entries": 1,
            "entries": [{
                "id": "const-fold",
                "kind": "executable-pass",
                "owner": "vyre-foundation",
                "phase": "Canonicalize",
                "boundary": "Program",
                "invariant": "semantic equivalence",
                "benchmark": "reviewed-non-benchmark: correctness-only"
            }],
            "blockers": []
        })
    }
}
