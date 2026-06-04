//! Weir analysis API and integration evidence.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Serialize)]
struct WeirMatrix {
    schema_version: u32,
    analyses: Vec<WeirAnalysis>,
    inventory_registered_count: usize,
    required_api_item_count: usize,
    missing_api_item_count: usize,
    property_test_count: usize,
    parity_test_count: usize,
    adversarial_test_count: usize,
    perf_test_count: usize,
    fuzz_test_count: usize,
    gap_test_count: usize,
    standalone_example_count: usize,
    standalone_serde_evidence_count: usize,
    standalone_serde_feature_guard_count: usize,
    standalone_example_scan_errors: Vec<String>,
    standalone_examples: Vec<ComponentFile>,
    untested_analyses: Vec<&'static str>,
    integration_tests: Vec<WeirTest>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct WeirAnalysis {
    id: &'static str,
    path: String,
    exists: bool,
    public_exported: bool,
    source_bytes: usize,
    has_public_api: bool,
    required_api_items: Vec<&'static str>,
    missing_api_items: Vec<&'static str>,
    required_policy_items: Vec<&'static str>,
    missing_policy_items: Vec<&'static str>,
    declares_op_id: bool,
    inventory_registered: bool,
    unresolved_markers: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct WeirTest {
    id: &'static str,
    path: String,
    exists: bool,
    source_bytes: usize,
    has_test_entrypoint: bool,
    assertion_count: usize,
    unresolved_markers: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct ComponentFile {
    path: String,
    exists: bool,
    source_bytes: usize,
    read_error: Option<String>,
    has_main: bool,
    uses_weir_crate: bool,
    has_serde_evidence: bool,
    api_reference_count: usize,
    unresolved_markers: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct WeirIntegrationEvidence {
    schema_version: u32,
    tests: Vec<WeirTest>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct WeirReadmeEvidence {
    schema_version: u32,
    path: String,
    exists: bool,
    source_bytes: usize,
    required_tokens: Vec<&'static str>,
    missing_tokens: Vec<&'static str>,
    example_count: usize,
    blockers: Vec<String>,
}

const ANALYSES: &[(&str, &str)] = &[
    ("ssa", "src/ssa.rs"),
    ("def_use", "src/def_use.rs"),
    ("reaching", "src/reaching.rs"),
    ("reaching_def", "src/reaching_def.rs"),
    ("points_to", "src/points_to.rs"),
    ("may_alias", "src/may_alias.rs"),
    ("ifds", "src/ifds.rs"),
    ("ifds_gpu", "src/ifds_gpu.rs"),
    ("callgraph", "src/callgraph.rs"),
    ("control_dependence", "src/control_dependence.rs"),
    ("cross_language", "src/cross_language.rs"),
    ("dominators", "src/dominators.rs"),
    ("escape", "src/escape.rs"),
    ("escapes", "src/escapes.rs"),
    ("live", "src/live.rs"),
    ("live_at", "src/live_at.rs"),
    ("slice", "src/slice.rs"),
    ("summary", "src/summary.rs"),
    ("loop_sum", "src/loop_sum.rs"),
    ("must_init", "src/must_init.rs"),
    ("post_dominates", "src/post_dominates.rs"),
    ("range", "src/range.rs"),
    ("range_check", "src/range_check.rs"),
    ("reachability_witness", "src/reachability_witness.rs"),
    ("scc_query", "src/scc_query.rs"),
    ("soundness", "src/soundness.rs"),
    ("value_set", "src/value_set.rs"),
];

const TESTS: &[(&str, &str)] = &[
    ("adversarial_oracles", "tests/df_adversarial_oracles.rs"),
    ("anchor_bit_codegen", "tests/df_anchor_bit_codegen.rs"),
    ("cross_arm_raw_atomic", "tests/df_cross_arm_raw_atomic.rs"),
    ("construction_def_use", "tests/df_def_use.rs"),
    (
        "construction_dominators",
        "tests/df_dominators_construction.rs",
    ),
    ("construction_ifds", "tests/df_ifds_construction.rs"),
    ("construction_live", "tests/df_live_construction.rs"),
    (
        "construction_may_alias",
        "tests/df_may_alias_construction.rs",
    ),
    ("construction_reaching", "tests/df_reaching_construction.rs"),
    (
        "construction_range_check",
        "tests/df_range_check_construction.rs",
    ),
    ("cross_language", "tests/df_cross_language.rs"),
    (
        "cross_primitive_composition",
        "tests/df_cross_primitive_composition.rs",
    ),
    (
        "escape_callgraph_range",
        "tests/df_escape_callgraph_range.rs",
    ),
    ("live_at_escapes", "tests/df_live_at_escapes.rs"),
    ("must_init_scc_query", "tests/df_must_init_scc_query.rs"),
    (
        "parity_exact_primitives",
        "tests/df_parity_exact_primitives.rs",
    ),
    ("parity_dominators", "tests/df_parity_dominators.rs"),
    (
        "parity_inventory_sweep",
        "tests/df_parity_inventory_sweep.rs",
    ),
    ("parity_may_alias", "tests/df_parity_may_alias.rs"),
    ("reachability_witness", "tests/df_reachability_witness.rs"),
    (
        "slice_reaching_def_control_dep",
        "tests/df_slice_reaching_def_control_dep.rs",
    ),
    ("soundness_tags", "tests/df_soundness_tags.rs"),
    ("ssa_dominators", "tests/df_ssa_dominators.rs"),
    (
        "value_set_post_dominates",
        "tests/df_value_set_post_dominates.rs",
    ),
    ("property_points_to", "tests/df_property_points_to.rs"),
    ("property_may_alias", "tests/df_property_may_alias.rs"),
    ("property_ifds", "tests/df_property_ifds.rs"),
    (
        "property_control_dependence",
        "tests/df_property_control_dependence.rs",
    ),
    (
        "property_cross_language",
        "tests/df_property_cross_language.rs",
    ),
    ("property_def_use", "tests/df_property_def_use.rs"),
    ("property_dominators", "tests/df_property_dominators.rs"),
    ("property_range_check", "tests/df_property_range_check.rs"),
    ("property_range_escape", "tests/df_property_range_escape.rs"),
    (
        "property_reachability_witness",
        "tests/df_property_reachability_witness.rs",
    ),
    (
        "property_reaching_def_escapes",
        "tests/df_property_reaching_def_escapes.rs",
    ),
    ("property_slice", "tests/df_property_slice_construction.rs"),
    ("property_ssa", "tests/df_property_ssa_dominators.rs"),
    (
        "property_summary_callgraph",
        "tests/df_property_summary_callgraph.rs",
    ),
    ("property_value_set", "tests/df_property_value_set.rs"),
    (
        "property_bitset_oracles",
        "tests/df_property_bitset_oracles.rs",
    ),
    ("fuzz_bitset_oracles", "tests/df_fuzz_bitset_oracles.rs"),
    (
        "gap_bitset_oracle_edges",
        "tests/df_gap_bitset_oracle_edges.rs",
    ),
    ("resolve_family_node39", "tests/df_resolve_family_node39.rs"),
    ("summary_loop_points", "tests/df_summary_loop_points.rs"),
    ("three_arm_fusion", "tests/df_three_arm_fusion.rs"),
    ("perf_oracle", "tests/df_perf_oracle_throughput.rs"),
    ("scale_oracle", "tests/df_scale_oracle_no_oom.rs"),
];

const UNRESOLVED_MARKERS: &[&str] = &[
    "todo",
    "fixme",
    "placeholder",
    "stub",
    "todo!",
    "unimplemented!",
    "panic!(\"not implemented",
    "tbd",
];

const MIN_PROPERTY_TEST_FAMILIES: usize = 15;
const MIN_PARITY_TEST_FAMILIES: usize = 4;
const MIN_ADVERSARIAL_TEST_FAMILIES: usize = 1;
const MIN_PERF_TEST_FAMILIES: usize = 2;
const MIN_FUZZ_TEST_FAMILIES: usize = 1;
const MIN_GAP_TEST_FAMILIES: usize = 1;
const MAX_WEIR_EVIDENCE_SOURCE_BYTES: u64 = 2_097_152;

pub(crate) fn run(args: &[String]) {
    let output = match parse_output(args) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let weir_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(|root| root.join("libs/dataflow/weir"))
        .unwrap_or_else(|| PathBuf::from("../../../../libs/dataflow/weir"));
    let mut blockers = Vec::new();
    let lib_rs_path = weir_root.join("src/lib.rs");
    let lib_rs = match read_text_bounded(&lib_rs_path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "Weir public export scan could not read {}: {error}",
                lib_rs_path.display()
            ));
            String::new()
        }
    };
    let mut analyses = Vec::new();
    for &(id, relative) in ANALYSES {
        let path = weir_root.join(relative);
        let exists = path.is_file();
        let text = if exists {
            match read_text_bounded(&path) {
                Ok(text) => text,
                Err(error) => {
                    blockers.push(format!(
                        "Weir analysis `{id}` could not be read at {}: {error}",
                        path.display()
                    ));
                    String::new()
                }
            }
        } else {
            String::new()
        };
        let lowered = text.to_ascii_lowercase();
        let module_scope_text =
            analysis_module_scope_text(&weir_root, relative, &text, &mut blockers);
        let has_public_api = module_scope_text.contains("pub fn ")
            || module_scope_text.contains("pub struct ")
            || module_scope_text.contains("pub enum ")
            || module_scope_text.contains("pub type ")
            || text.contains("pub use ");
        let declares_op_id = text.contains("OP_ID");
        let inventory_registered = module_scope_text.contains("inventory::submit!")
            && module_scope_text.contains("vyre_harness::OpEntry::new");
        let unresolved_markers = UNRESOLVED_MARKERS
            .iter()
            .copied()
            .filter(|marker| lowered.contains(marker))
            .collect::<Vec<_>>();
        let module_name = relative
            .strip_prefix("src/")
            .and_then(|value| value.strip_suffix(".rs"))
            .unwrap_or(id);
        let public_exported = lib_rs.contains(&format!("pub mod {module_name};"));
        if !exists {
            blockers.push(format!(
                "Weir analysis `{id}` is missing at {}",
                path.display()
            ));
        } else if text.trim().is_empty() {
            blockers.push(format!("Weir analysis `{id}` source file is empty"));
        }
        if !public_exported {
            blockers.push(format!(
                "Weir analysis `{id}` is not publicly exported from src/lib.rs"
            ));
        }
        if exists && !has_public_api {
            blockers.push(format!("Weir analysis `{id}` exposes no public API item"));
        }
        let required_api_items = required_api_items_for(id);
        let missing_api_items = required_api_items
            .iter()
            .copied()
            .filter(|required| !text.contains(required))
            .collect::<Vec<_>>();
        for required in &missing_api_items {
            blockers.push(format!(
                "Weir analysis `{id}` is missing required public API item `{required}`"
            ));
        }
        let required_policy_items = required_policy_items_for(id);
        let missing_policy_items = required_policy_items
            .iter()
            .copied()
            .filter(|required| !text.contains(required))
            .collect::<Vec<_>>();
        if id == "soundness" {
            for required in &missing_policy_items {
                blockers.push(format!(
                    "Weir soundness API is missing required policy item `{required}`"
                ));
            }
        }
        if exists && declares_op_id && !inventory_registered {
            blockers.push(format!(
                "Weir analysis `{id}` declares OP_ID but does not submit a vyre_harness::OpEntry"
            ));
        }
        for marker in &unresolved_markers {
            blockers.push(format!(
                "Weir analysis `{id}` contains unresolved marker `{marker}`"
            ));
        }
        analyses.push(WeirAnalysis {
            id,
            path: path.display().to_string(),
            exists,
            public_exported,
            source_bytes: text.len(),
            has_public_api,
            required_api_items,
            missing_api_items,
            required_policy_items,
            missing_policy_items,
            declares_op_id,
            inventory_registered,
            unresolved_markers,
        });
    }
    let mut integration_tests = Vec::new();
    for &(id, relative) in TESTS {
        let path = weir_root.join(relative);
        let exists = path.is_file();
        let text = if exists {
            match read_text_bounded(&path) {
                Ok(text) => text,
                Err(error) => {
                    blockers.push(format!(
                        "Weir integration test `{id}` could not be read at {}: {error}",
                        path.display()
                    ));
                    String::new()
                }
            }
        } else {
            String::new()
        };
        let lowered = text.to_ascii_lowercase();
        let has_test_entrypoint = text.contains("#[test]") || text.contains("proptest!");
        let assertion_count = assertion_count(&text);
        let unresolved_markers = UNRESOLVED_MARKERS
            .iter()
            .copied()
            .filter(|marker| lowered.contains(marker))
            .collect::<Vec<_>>();
        if !exists {
            blockers.push(format!(
                "Weir integration test `{id}` is missing at {}",
                path.display()
            ));
        } else if text.trim().is_empty() {
            blockers.push(format!("Weir integration test `{id}` is empty"));
        }
        if exists && !has_test_entrypoint {
            blockers.push(format!(
                "Weir integration test `{id}` has no #[test] or proptest! entrypoint"
            ));
        }
        if exists && assertion_count == 0 {
            blockers.push(format!(
                "Weir integration test `{id}` has no assertion or property assertion"
            ));
        }
        for marker in &unresolved_markers {
            blockers.push(format!(
                "Weir integration test `{id}` contains unresolved marker `{marker}`"
            ));
        }
        integration_tests.push(WeirTest {
            id,
            path: path.display().to_string(),
            exists,
            source_bytes: text.len(),
            has_test_entrypoint,
            assertion_count,
            unresolved_markers,
        });
    }
    let property_test_count = integration_tests
        .iter()
        .filter(|test| test.id.starts_with("property_"))
        .count();
    let parity_test_count = integration_tests
        .iter()
        .filter(|test| test.id.starts_with("parity_"))
        .count();
    let adversarial_test_count = integration_tests
        .iter()
        .filter(|test| test.id.contains("adversarial"))
        .count();
    let perf_test_count = integration_tests
        .iter()
        .filter(|test| test.id.contains("perf") || test.id.contains("scale"))
        .count();
    let fuzz_test_count = integration_tests
        .iter()
        .filter(|test| test.id.contains("fuzz"))
        .count();
    let gap_test_count = integration_tests
        .iter()
        .filter(|test| test.id.contains("gap"))
        .count();
    let untested_analyses = analyses
        .iter()
        .filter(|analysis| !analysis_has_release_test(analysis.id, &integration_tests))
        .map(|analysis| analysis.id)
        .collect::<Vec<_>>();
    for analysis in &untested_analyses {
        blockers.push(format!(
            "Weir analysis `{analysis}` has no release integration, property, parity, fuzz, gap, perf, or scale test"
        ));
    }
    if property_test_count < MIN_PROPERTY_TEST_FAMILIES {
        blockers.push(format!(
            "Weir matrix has {property_test_count} property test families; release requires at least {MIN_PROPERTY_TEST_FAMILIES}"
        ));
    }
    if parity_test_count < MIN_PARITY_TEST_FAMILIES {
        blockers.push(format!(
            "Weir matrix has {parity_test_count} parity test families; release requires at least {MIN_PARITY_TEST_FAMILIES}"
        ));
    }
    if adversarial_test_count < MIN_ADVERSARIAL_TEST_FAMILIES {
        blockers.push(format!(
            "Weir matrix has {adversarial_test_count} adversarial test families; release requires at least {MIN_ADVERSARIAL_TEST_FAMILIES}"
        ));
    }
    if perf_test_count < MIN_PERF_TEST_FAMILIES {
        blockers.push(format!(
            "Weir matrix has {perf_test_count} perf/scale test families; release requires at least {MIN_PERF_TEST_FAMILIES}"
        ));
    }
    if fuzz_test_count < MIN_FUZZ_TEST_FAMILIES {
        blockers.push(format!(
            "Weir matrix has {fuzz_test_count} fuzz test families; release requires at least {MIN_FUZZ_TEST_FAMILIES}"
        ));
    }
    if gap_test_count < MIN_GAP_TEST_FAMILIES {
        blockers.push(format!(
            "Weir matrix has {gap_test_count} gap test families; release requires at least {MIN_GAP_TEST_FAMILIES}"
        ));
    }
    let mut standalone_example_scan_errors = Vec::new();
    let standalone_examples = collect_standalone_examples(
        &weir_root,
        &mut blockers,
        &mut standalone_example_scan_errors,
    );
    if standalone_examples.len() < 2 {
        blockers.push(format!(
            "Weir matrix has {} standalone example(s); release requires at least 2 examples outside tests",
            standalone_examples.len()
        ));
    }
    let standalone_serde_evidence_count = standalone_examples
        .iter()
        .filter(|example| example.has_serde_evidence)
        .count();
    if standalone_serde_evidence_count == 0 {
        blockers.push(
            "Weir matrix has no standalone example proving serde evidence for witness or soundness API types"
                .to_string(),
        );
    }
    let cargo_toml_path = weir_root.join("Cargo.toml");
    let cargo_toml = match read_text_bounded(&cargo_toml_path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "Weir Cargo.toml could not be read at {}: {error}",
                cargo_toml_path.display()
            ));
            String::new()
        }
    };
    let standalone_serde_feature_guard_count = usize::from(
        cargo_toml.contains("name = \"serde_evidence\"")
            && cargo_toml.contains("required-features = [\"serde\"]"),
    );
    if standalone_serde_evidence_count > 0 && standalone_serde_feature_guard_count == 0 {
        blockers.push(
            "Weir serde evidence example must declare required-features = [\"serde\"] in Cargo.toml"
                .to_string(),
        );
    }
    for example in &standalone_examples {
        if !example.exists {
            blockers.push(format!(
                "Weir standalone example {} is missing",
                example.path
            ));
        } else if let Some(error) = &example.read_error {
            blockers.push(format!(
                "Weir standalone example {} could not be read: {error}",
                example.path
            ));
        } else if example.source_bytes == 0 {
            blockers.push(format!("Weir standalone example {} is empty", example.path));
        }
        if example.exists && !example.has_main {
            blockers.push(format!(
                "Weir standalone example {} has no runnable fn main",
                example.path
            ));
        }
        if example.exists && !example.uses_weir_crate {
            blockers.push(format!(
                "Weir standalone example {} does not import or reference the weir crate",
                example.path
            ));
        }
        if example.exists && example.api_reference_count < 2 {
            blockers.push(format!(
                "Weir standalone example {} references {} dataflow API token(s); release requires at least 2",
                example.path, example.api_reference_count
            ));
        }
        for marker in &example.unresolved_markers {
            blockers.push(format!(
                "Weir standalone example {} contains unresolved marker `{marker}`",
                example.path
            ));
        }
    }
    let matrix = WeirMatrix {
        schema_version: 2,
        inventory_registered_count: analyses
            .iter()
            .filter(|analysis| analysis.inventory_registered)
            .count(),
        required_api_item_count: analyses
            .iter()
            .map(|analysis| analysis.required_api_items.len())
            .sum(),
        missing_api_item_count: analyses
            .iter()
            .map(|analysis| analysis.missing_api_items.len())
            .sum(),
        property_test_count,
        parity_test_count,
        adversarial_test_count,
        perf_test_count,
        fuzz_test_count,
        gap_test_count,
        standalone_example_count: standalone_examples.len(),
        standalone_serde_evidence_count,
        standalone_serde_feature_guard_count,
        standalone_example_scan_errors,
        standalone_examples,
        untested_analyses,
        analyses,
        integration_tests,
        blockers,
    };
    let json = match serde_json::to_string_pretty(&matrix) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize Weir matrix: {error}");
            std::process::exit(1);
        }
    };
    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = fs::write(&output, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", output.display());
        std::process::exit(1);
    }
    write_sibling_artifacts(&output, &matrix);
    println!("weir-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn collect_standalone_examples(
    weir_root: &Path,
    blockers: &mut Vec<String>,
    scan_errors: &mut Vec<String>,
) -> Vec<ComponentFile> {
    let examples_root = weir_root.join("examples");
    let entries = match fs::read_dir(&examples_root) {
        Ok(entries) => entries,
        Err(error) => {
            let message = format!(
                "Weir examples directory could not be read at {}: {error}",
                examples_root.display()
            );
            blockers.push(message.clone());
            scan_errors.push(message);
            return Vec::new();
        }
    };
    let mut examples = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                let message = format!(
                    "Weir examples entry could not be read in {}: {error}",
                    examples_root.display()
                );
                blockers.push(message.clone());
                scan_errors.push(message);
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let (text, read_error) = match read_text_bounded(&path) {
            Ok(text) => (text, None),
            Err(error) => (String::new(), Some(error.to_string())),
        };
        let lowered = text.to_ascii_lowercase();
        let unresolved_markers = UNRESOLVED_MARKERS
            .iter()
            .copied()
            .filter(|marker| lowered.contains(marker))
            .collect::<Vec<_>>();
        let api_reference_count = [
            "ssa",
            "reaching",
            "reachingdef",
            "points",
            "alias",
            "ifds",
            "callgraph",
            "slice",
            "summary",
            "loop",
            "fixpoint",
            "soundness",
        ]
        .iter()
        .filter(|token| lowered.contains(**token))
        .count();
        let has_serde_evidence = lowered.contains("serde")
            && lowered.contains("serialize")
            && (lowered.contains("deserialize") || lowered.contains("deserializeowned"))
            && (lowered.contains("pathseed") || lowered.contains("soundness"));
        examples.push(ComponentFile {
            path: path.display().to_string(),
            exists: path.is_file(),
            source_bytes: text.len(),
            read_error,
            has_main: text.contains("fn main(") || text.contains("fn main ()"),
            uses_weir_crate: text.contains("weir::") || text.contains("use weir"),
            has_serde_evidence,
            api_reference_count,
            unresolved_markers,
        });
    }
    examples.sort_by(|left, right| left.path.cmp(&right.path));
    examples
}

fn write_sibling_artifacts(output: &Path, matrix: &WeirMatrix) {
    let Some(parent) = output.parent() else {
        eprintln!(
            "Fix: Weir matrix output `{}` has no parent directory.",
            output.display()
        );
        std::process::exit(1);
    };
    let blockers = matrix
        .integration_tests
        .iter()
        .flat_map(|test| {
            let mut blockers = Vec::new();
            if !test.exists {
                blockers.push(format!(
                    "Weir integration test `{}` is missing at {}",
                    test.id, test.path
                ));
            }
            if test.exists && test.source_bytes == 0 {
                blockers.push(format!("Weir integration test `{}` is empty", test.id));
            }
            if test.exists && !test.has_test_entrypoint {
                blockers.push(format!(
                    "Weir integration test `{}` has no #[test] or proptest! entrypoint",
                    test.id
                ));
            }
            if test.exists && test.assertion_count == 0 {
                blockers.push(format!(
                    "Weir integration test `{}` has no assertion or property assertion",
                    test.id
                ));
            }
            for marker in &test.unresolved_markers {
                blockers.push(format!(
                    "Weir integration test `{}` contains unresolved marker `{marker}`",
                    test.id
                ));
            }
            blockers
        })
        .collect::<Vec<_>>();
    write_json(
        &parent.join("weir-vyre-integration-tests.json"),
        &WeirIntegrationEvidence {
            schema_version: 2,
            tests: matrix.integration_tests.clone(),
            blockers,
        },
    );
    write_weir_readme_artifact(parent);
}

fn write_weir_readme_artifact(parent: &Path) {
    let weir_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(|root| root.join("libs/dataflow/weir"))
        .unwrap_or_else(|| PathBuf::from("../../../../libs/dataflow/weir"));
    let readme = weir_root.join("README.md");
    let exists = readme.is_file();
    let mut blockers = Vec::new();
    let text = if exists {
        match read_text_bounded(&readme) {
            Ok(text) => text,
            Err(error) => {
                blockers.push(format!(
                    "Weir README could not be read at {}: {error}",
                    readme.display()
                ));
                String::new()
            }
        }
    } else {
        String::new()
    };
    let lowered = text.to_ascii_lowercase();
    let required_tokens = vec![
        "0.1.0",
        "dataflow",
        "vyre",
        "ssa",
        "def-use",
        "reaching",
        "reaching-definition",
        "points-to",
        "may-alias",
        "ifds",
        "callgraph",
        "control-dependence",
        "cross-language",
        "dominators",
        "escape",
        "live",
        "must-init",
        "post-dominates",
        "range",
        "range-check",
        "scc",
        "slice",
        "summary",
        "value-set",
        "soundness",
        "serde",
        "default feature",
        "serde_evidence",
        "required-features",
        "precisioncontract",
        "primitive soundness",
        "cargo add weir",
    ];
    let missing_tokens = required_tokens
        .iter()
        .copied()
        .filter(|token| !lowered.contains(&token.to_ascii_lowercase()))
        .collect::<Vec<_>>();
    let example_count = text.matches("```rust").count() + text.matches("```toml").count();
    if !exists {
        blockers.push(format!("Weir README is missing at {}", readme.display()));
    }
    if exists && text.trim().is_empty() {
        blockers.push("Weir README is empty".to_string());
    }
    for token in &missing_tokens {
        blockers.push(format!("Weir README is missing required token `{token}`"));
    }
    if example_count == 0 {
        blockers
            .push("Weir README must include at least one Rust or TOML example block".to_string());
    }
    write_json(
        &parent.join("weir-readme-contracts.json"),
        &WeirReadmeEvidence {
            schema_version: 2,
            path: readme.display().to_string(),
            exists,
            source_bytes: text.len(),
            required_tokens,
            missing_tokens,
            example_count,
            blockers,
        },
    );
}

fn write_json(path: &Path, value: &impl Serialize) {
    let json = match serde_json::to_string_pretty(value) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize `{}`: {error}", path.display());
            std::process::exit(1);
        }
    };
    if let Err(error) = fs::write(path, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    let mut output = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --output requires a path.".to_string());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- weir-matrix [--output PATH]\n\n\
                     Writes Weir analysis API and integration evidence."
                );
                std::process::exit(0);
            }
            other => return Err(format!("Fix: unknown weir-matrix option `{other}`.")),
        }
    }
    Ok(output.unwrap_or_else(default_output))
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/weir/weir-analysis-api-matrix.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/weir/weir-analysis-api-matrix.json"))
}

fn analysis_module_scope_text(
    weir_root: &Path,
    relative: &str,
    top_level_text: &str,
    blockers: &mut Vec<String>,
) -> String {
    let mut scope = String::from(top_level_text);
    let Some(module_name) = relative
        .strip_prefix("src/")
        .and_then(|value| value.strip_suffix(".rs"))
    else {
        return scope;
    };
    let module_dir = weir_root.join("src").join(module_name);
    if !module_dir.is_dir() {
        return scope;
    }
    let entries = match fs::read_dir(&module_dir) {
        Ok(entries) => entries,
        Err(error) => {
            blockers.push(format!(
                "Weir analysis module `{module_name}` could not be scanned at {}: {error}",
                module_dir.display()
            ));
            return scope;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                blockers.push(format!(
                    "Weir analysis module `{module_name}` had unreadable entry in {}: {error}",
                    module_dir.display()
                ));
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        match read_text_bounded(&path) {
            Ok(text) => {
                scope.push('\n');
                scope.push_str(&text);
            }
            Err(error) => blockers.push(format!(
                "Weir analysis module `{module_name}` could not read {} while scanning registration scope: {error}",
                path.display()
            )),
        }
    }
    scope
}

fn required_api_items_for(id: &str) -> Vec<&'static str> {
    match id {
        "ssa" => vec![
            "SsaForm",
            "Cfg",
            "ssa_phi_placement_step",
            "compute_dominators",
            "compute_dominance_frontiers",
            "place_phi_nodes",
            "rename_variables",
            "Ssa",
        ],
        "def_use" => vec![
            "def_use_chain",
            "def_use_chain_bitset",
            "def_use_query",
            "cpu_ref",
            "DefUse",
        ],
        "reaching" => vec!["reaching_defs_step", "ReachingDefs"],
        "reaching_def" => vec!["reaching_def", "cpu_ref", "ReachingDef"],
        "points_to" => vec![
            "andersen_points_to",
            "andersen_points_to_with_shape",
            "cpu_subset_closure",
            "PointsTo",
        ],
        "may_alias" => vec!["may_alias", "cpu_ref", "MayAlias"],
        "ifds" => vec!["ifds_reach_step", "ifds_reach_step_exploded", "Ifds"],
        "ifds_gpu" => vec![
            "solve_cpu",
            "IfdsShape",
            "ifds_gpu_step",
            "ifds_gpu",
            "IfdsGpu",
        ],
        "callgraph" => vec!["callgraph_build", "callgraph_build_with_count", "Callgraph"],
        "control_dependence" => {
            vec!["control_dependence", "cpu_ref", "ControlDependence"]
        }
        "cross_language" => vec![
            "EDGE_KIND_FFI",
            "EDGE_KIND_ALL",
            "cross_language",
            "cpu_ref",
            "CrossLanguage",
        ],
        "dominators" => vec![
            "dominates",
            "cpu_ref",
            "compute_cpu",
            "compute_bitmap_bytes",
            "Dominators",
        ],
        "escape" => vec!["escape_analyze", "escape_analyze_with_count", "Escape"],
        "escapes" => vec!["escapes", "cpu_ref", "Escapes"],
        "live" => vec!["live_step", "Liveness"],
        "live_at" => vec!["live_at", "cpu_ref", "LiveAt"],
        "slice" => vec![
            "backward_slice",
            "backward_slice_with_shape",
            "BackwardSlice",
        ],
        "summary" => vec![
            "summarize_function",
            "summarize_function_with_count",
            "Summary",
        ],
        "loop_sum" => vec!["loop_summarize", "loop_summarize_with_count", "LoopSum"],
        "must_init" => vec!["must_init", "cpu_ref", "MustInit"],
        "post_dominates" => vec!["post_dominates", "cpu_ref", "PostDominates"],
        "range" => vec!["range_propagate", "range_propagate_with_count", "Range"],
        "range_check" => vec!["range_check", "cpu_ref", "RangeCheck"],
        "reachability_witness" => vec![
            "PathSeed",
            "ExtractedPath",
            "PreparedWitnessGraph",
            "exploded_reachability_to_statement_mask",
            "extract_path",
            "prepare_witness_graph",
            "extract_path_prepared",
            "NodeAttr",
        ],
        "scc_query" => vec!["scc_query", "cpu_ref", "SccQuery"],
        "soundness" => vec![
            "Soundness",
            "PrecisionContract",
            "PrimitiveSoundness",
            "SoundnessViolation",
            "SoundnessTagged",
            "validate_pipeline",
            "validate_primitive",
        ],
        "value_set" => vec!["value_set", "cpu_ref", "ValueSet"],
        _ => Vec::new(),
    }
}

fn required_policy_items_for(id: &str) -> Vec<&'static str> {
    if id == "soundness" {
        vec![
            "PrecisionContract",
            "PrimitiveSoundness",
            "SoundnessViolation",
            "SoundnessTagged",
            "validate_pipeline",
            "validate_primitive",
        ]
    } else {
        Vec::new()
    }
}

fn analysis_has_release_test(id: &str, tests: &[WeirTest]) -> bool {
    let aliases = analysis_test_aliases(id);
    tests.iter().any(|test| {
        test.exists
            && test.has_test_entrypoint
            && test.assertion_count > 0
            && aliases.iter().any(|alias| test.id.contains(alias))
    })
}

fn analysis_test_aliases(id: &str) -> Vec<&str> {
    match id {
        "reaching_def" => vec!["reaching_def", "slice_reaching_def"],
        "points_to" => vec!["points_to", "points"],
        "may_alias" => vec!["may_alias", "alias"],
        "ifds_gpu" => vec!["ifds_gpu", "ifds"],
        "control_dependence" => vec!["control_dependence", "control_dep"],
        "escape" => vec!["escape", "range_escape"],
        "escapes" => vec!["escapes", "live_at_escapes"],
        "live_at" => vec!["live_at", "live_at_escapes"],
        "summary" => vec!["summary", "summary_loop_points"],
        "loop_sum" => vec!["loop_sum", "summary_loop_points"],
        "must_init" => vec!["must_init", "must_init_scc_query"],
        "post_dominates" => vec!["post_dominates", "value_set_post_dominates"],
        "range" => vec!["range", "range_escape"],
        "scc_query" => vec!["scc_query", "must_init_scc_query"],
        "soundness" => vec!["soundness", "soundness_tags"],
        other => vec![other],
    }
}

fn assertion_count(text: &str) -> usize {
    [
        "assert!(",
        "assert_eq!(",
        "assert_ne!(",
        "prop_assert!(",
        "prop_assert_eq!(",
        "prop_assert_ne!(",
    ]
    .iter()
    .map(|needle| text.matches(needle).count())
    .sum()
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_WEIR_EVIDENCE_SOURCE_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_WEIR_EVIDENCE_SOURCE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_WEIR_EVIDENCE_SOURCE_BYTES} byte Weir evidence read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
