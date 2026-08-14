//! `cargo xtask op-matrix`  -  generate and check the canonical op matrix.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process;

use vyre_foundation::operation::{
    classify_operation_id as classify_op_id, OperationTier as OpTier,
};

const DEFAULT_MATRIX_PATH: &str = "docs/optimization/OP_MATRIX.toml";
const MAX_OP_MATRIX_TEXT_BYTES: u64 = 4_194_304;

const SCAN_CONSTRUCT_MATRIX: &str = r#"# Manual scan construct tier data owned by VX-621/VX-622. Generated `[[op]]`
# rows below remain generator-owned.
scan_construct_tier_values = [
  "supported",
  "rejected",
  "approximated",
  "accelerator-only",
  "verifier-required",
]

scan_construct_route_values = [
  "native",
  "unsupported",
  "prefilter",
  "verifier",
  "external-accelerator",
  "host-reference",
]

[[scan_construct]]
id = "regular_exact_core"
tier = "supported"
dialect_class = "regular"
constructs = ["literal", "concatenation", "alternation", "bounded_repeat"]
diagnostic_code = "VYRE_SCAN_OK_EXACT_CORE"
user_diagnostic = "Exact regular constructs are eligible for native CPU and accelerator routes when backend capability checks pass."
approximation_policy = "exact"
verifier_required = false
accelerator_only = false
backend_routes = { cpu_ref = "native", cuda = "native", wgpu = "native", metal = "native", hyperscan = "native", vectorscan = "native", rust_regex = "native", dpu = "unsupported", fpga = "unsupported" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth.rs", "vyre-libs/tests/scan_conformance_matrix.rs"]
bench_targets = []

[[scan_construct]]
id = "unsupported_backtracking_constructs"
tier = "rejected"
dialect_class = "pcre-compatible"
constructs = ["backreference", "conditional_reference", "recursion", "subroutine_call"]
diagnostic_code = "VYRE_SCAN_UNSUPPORTED_BACKTRACKING_CONSTRUCT"
user_diagnostic = "Backtracking-only constructs are rejected unless a future verifier route has exact bounded semantics for the requested dialect."
approximation_policy = "none"
verifier_required = false
accelerator_only = false
backend_routes = { cpu_ref = "unsupported", cuda = "unsupported", wgpu = "unsupported", metal = "unsupported", hyperscan = "unsupported", vectorscan = "unsupported", rust_regex = "unsupported", dpu = "unsupported", fpga = "unsupported" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth.rs", "vyre-libs/tests/scan_conformance_matrix.rs"]
bench_targets = []

[[scan_construct]]
id = "lookaround_prefilter_constructs"
tier = "approximated"
dialect_class = "pcre-compatible"
constructs = ["positive_lookahead", "negative_lookahead", "fixed_width_lookbehind", "negative_lookbehind"]
diagnostic_code = "VYRE_SCAN_APPROXIMATED_LOOKAROUND_REQUIRES_VERIFIER"
user_diagnostic = "Lookaround constructs can only enter an approximate prefilter route when final match offsets are proven by a verifier."
approximation_policy = "broader-prefilter-plus-verifier"
verifier_required = true
accelerator_only = false
backend_routes = { cpu_ref = "host-reference", cuda = "prefilter", wgpu = "prefilter", metal = "prefilter", hyperscan = "prefilter", vectorscan = "prefilter", rust_regex = "unsupported", dpu = "unsupported", fpga = "prefilter" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth.rs", "vyre-libs/tests/scan_conformance_matrix.rs"]
bench_targets = []

[[scan_construct]]
id = "hardware_rule_database_constructs"
tier = "accelerator-only"
dialect_class = "external-rule-database"
constructs = ["bluefield_rule_set", "rof2_rule_database", "fpga_rule_image", "rxp_job"]
diagnostic_code = "VYRE_SCAN_ACCELERATOR_RULE_DATABASE_REQUIRED"
user_diagnostic = "External hardware rule databases are accelerator-only artifacts and must name the compiled rule digest before dispatch."
approximation_policy = "hardware-rule-database"
verifier_required = false
accelerator_only = true
backend_routes = { cpu_ref = "unsupported", cuda = "unsupported", wgpu = "unsupported", metal = "unsupported", hyperscan = "unsupported", vectorscan = "unsupported", rust_regex = "unsupported", dpu = "external-accelerator", fpga = "external-accelerator" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth.rs", "vyre-libs/tests/scan_conformance_matrix.rs"]
bench_targets = []

[[scan_construct]]
id = "capture_extraction_constructs"
tier = "verifier-required"
dialect_class = "capture"
constructs = ["capture_group", "named_capture", "submatch_offsets", "repeated_capture"]
diagnostic_code = "VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER"
user_diagnostic = "Capture extraction routes must preserve submatch spans through verifier output even when the accelerator only reports whole-match offsets."
approximation_policy = "whole-match-accelerator-plus-capture-verifier"
verifier_required = true
accelerator_only = false
backend_routes = { cpu_ref = "native", cuda = "verifier", wgpu = "verifier", metal = "verifier", hyperscan = "verifier", vectorscan = "verifier", rust_regex = "native", dpu = "unsupported", fpga = "verifier" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth.rs", "vyre-libs/tests/scan_conformance_matrix.rs"]
bench_targets = []

"#;

#[derive(Clone)]
struct OpRecord {
    family: String,
    tier: OpTier,
    owners: Vec<String>,
    ops: Vec<String>,
    registry_sources: Vec<String>,
    duplicate_ok: bool,
    reference: &'static str,
    foundation_ir: &'static str,
    cuda: &'static str,
    wgpu: &'static str,
    spirv: &'static str,
    release_blocking_notes: String,
    tests: Vec<String>,
    bench_targets: Vec<String>,
}

pub(crate) fn run(args: &[String]) {
    let mut check = false;
    let mut write = false;
    let mut path = DEFAULT_MATRIX_PATH.to_string();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--check" => check = true,
            "--write" => {
                write = true;
                if let Some(next) = args.get(i + 1).filter(|value| !value.starts_with("--")) {
                    path = next.clone();
                    i += 1;
                }
            }
            other => {
                eprintln!(
                    "Fix: unknown op-matrix argument `{other}`. Use --check or --write [PATH]."
                );
                process::exit(1);
            }
        }
        i += 1;
    }

    if !check && !write {
        write = true;
    }

    let matrix = match build_matrix() {
        Ok(matrix) => matrix,
        Err(error) => {
            eprintln!("{error}");
            process::exit(1);
        }
    };

    if check {
        let current = match read_text_bounded(Path::new(&path)) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("Fix: read `{path}` before op-matrix check: {error}");
                process::exit(1);
            }
        };
        if normalize_newline(&current) != normalize_newline(&matrix) {
            eprintln!(
                "Fix: `{path}` is not the source-backed op matrix. Run `cargo_full run --bin xtask -- op-matrix --write`."
            );
            process::exit(1);
        }
    }

    if write {
        if let Some(parent) = Path::new(&path).parent() {
            if let Err(error) = fs::create_dir_all(parent) {
                eprintln!(
                    "Fix: create `{}` before writing op matrix: {error}",
                    parent.display()
                );
                process::exit(1);
            }
        }
        if let Err(error) = fs::write(&path, matrix) {
            eprintln!("Fix: write `{path}`: {error}");
            process::exit(1);
        }
    }
}

fn normalize_newline(value: &str) -> String {
    value.replace("\r\n", "\n")
}

fn build_matrix() -> Result<String, String> {
    let mut records = manual_records();
    records.extend(registered_records()?);
    validate_records(&records)?;

    records.sort_by(|left, right| {
        (
            left.tier.matrix_value(),
            left.family.as_str(),
            left.ops.first().map(String::as_str),
        )
            .cmp(&(
                right.tier.matrix_value(),
                right.family.as_str(),
                right.ops.first().map(String::as_str),
            ))
    });

    Ok(render_matrix(&records))
}

fn manual_records() -> Vec<OpRecord> {
    vec![
        OpRecord {
            family: "integer_strength_reduction".to_string(),
            tier: OpTier::Foundation,
            owners: vec!["vyre-foundation/src/optimizer/passes/algebraic/strength_reduce".to_string()],
            ops: vec![
                "mul_power_of_two_to_shift".to_string(),
                "div_power_of_two_to_shift".to_string(),
                "mod_power_of_two_to_and".to_string(),
                "shift_add_decomposition".to_string(),
                "constant_division".to_string(),
            ],
            registry_sources: vec!["manual.foundation_ir".to_string()],
            duplicate_ok: false,
            reference: "not_applicable",
            foundation_ir: "supported",
            cuda: "not_applicable",
            wgpu: "not_applicable",
            spirv: "not_applicable",
            release_blocking_notes:
                "Backend rows are not applicable because the original IR should be rewritten before lowering."
                    .to_string(),
            tests: vec![
                "vyre-foundation/src/optimizer/passes/algebraic/strength_reduce/tests/mod.rs"
                    .to_string(),
            ],
            bench_targets: vec!["integer_arithmetic_micro".to_string()],
        },
        OpRecord {
            family: "elementwise_add".to_string(),
            tier: OpTier::Foundation,
            owners: vec!["vyre-bench/src/cases/elementwise.rs".to_string()],
            ops: vec!["f32_add".to_string()],
            registry_sources: vec!["manual.bench".to_string()],
            duplicate_ok: false,
            reference: "supported",
            foundation_ir: "supported",
            cuda: "supported",
            wgpu: "supported",
            spirv: "experimental",
            release_blocking_notes:
                "CUDA is the canonical performance backend for this release; active-time benchmark target is in BENCH_TARGETS.toml."
                    .to_string(),
            tests: vec!["vyre-driver-cuda/tests/resident_dispatch_contracts.rs".to_string()],
            bench_targets: vec!["foundation.elementwise.add.1m".to_string()],
        },
    ]
}

fn registered_records() -> Result<Vec<OpRecord>, String> {
    let mut ids = BTreeMap::<String, BTreeSet<String>>::new();

    let registry = vyre_foundation::operation::OperationRegistry::global();
    for entry in registry.iter() {
        push_registered(&mut ids, entry.id, "vyre-foundation::operation")?;
    }

    ids.into_iter()
        .map(|(id, sources)| record_for_registered_id(&id, sources))
        .collect()
}

fn push_registered(
    ids: &mut BTreeMap<String, BTreeSet<String>>,
    id: &str,
    source: &str,
) -> Result<(), String> {
    let sources = ids.entry(id.to_string()).or_default();
    if !sources.insert(source.to_string()) {
        return Err(format!(
            "Fix: duplicate op id `{id}` registered more than once by `{source}`. \
             Keep one canonical registration in that registry."
        ));
    }
    Ok(())
}

fn record_for_registered_id(
    id: &str,
    sources: BTreeSet<String>,
) -> Result<OpRecord, String> {
    let tier = classify_op_id(id);
    if tier == OpTier::Unknown {
        return Err(format!(
            "Fix: op id `{id}` from `{sources:?}` has no canonical tier namespace."
        ));
    }
    if sources.len() > 1 {
        return Err(format!(
            "Fix: op id `{id}` is registered by multiple semantic sources {sources:?}."
        ));
    }

    let record = OpRecord {
        family: id.to_string(),
        tier,
        owners: owner_paths(id, tier),
        ops: vec![id.to_string()],
        duplicate_ok: sources.len() > 1,
        registry_sources: sources.into_iter().collect(),
        reference: "supported",
        foundation_ir: "supported",
        cuda: "supported",
        wgpu: "supported",
        spirv: "experimental",
        release_blocking_notes: release_notes(id, tier),
        tests: test_paths(id, tier),
        bench_targets: Vec::new(),
    };

    Ok(record)
}

fn owner_paths(id: &str, tier: OpTier) -> Vec<String> {
    match tier {
        OpTier::Intrinsic => vec![namespace_source_dir(id)],
        OpTier::Library => {
            let domain = namespace_domain(id, "vyre-libs::");
            let owner = match domain {
                "optim" => "vyre-libs/src/nn/optim".to_string(),
                "quant" => "vyre-libs/src/nn/quant".to_string(),
                "substrate" => "vyre-libs/src/substrate_catalog.rs".to_string(),
                _ => format!("vyre-libs/src/{domain}"),
            };
            vec![owner]
        }
        OpTier::External => vec!["docs/optimization/README.md".to_string()],
        OpTier::Foundation | OpTier::Unknown => Vec::new(),
        _ => Vec::new(),
    }
}

/// `vyre-primitives::graph::toposort` becomes `vyre-primitives/src/graph`.
///
/// Read from the id so a namespace move carries its owner path with it instead
/// of leaving a second hardcoded crate name in this generator.
fn namespace_source_dir(id: &str) -> String {
    match id.split_once("::") {
        Some((crate_name, rest)) => {
            let domain = rest.split("::").next().unwrap_or("unknown");
            format!("{crate_name}/src/{domain}")
        }
        None => String::new(),
    }
}

fn namespace_domain<'a>(id: &'a str, prefix: &str) -> &'a str {
    id.strip_prefix(prefix)
        .and_then(|rest| rest.split("::").next())
        .unwrap_or("unknown")
}

fn test_paths(id: &str, tier: OpTier) -> Vec<String> {
    let mut tests = match tier {
        OpTier::Intrinsic => {
            let crate_name = id.split_once("::").map_or("", |(name, _)| name);
            let domain = id.split("::").nth(1).unwrap_or("");
            if domain == "hardware" {
                vec![format!("{crate_name}/tests/hardware_conform.rs")]
            } else {
                vec![format!("{crate_name}/tests/integration.rs")]
            }
        }
        OpTier::Library | OpTier::External => {
            vec!["vyre-libs/tests/universal_harness.rs".to_string()]
        }
        OpTier::Foundation | OpTier::Unknown => Vec::new(),
        _ => Vec::new(),
    };
    tests.push("conform/vyre-conform/tests/op_matrix_truth.rs".to_string());
    tests
}

fn release_notes(_id: &str, tier: OpTier) -> String {
    match tier {
        OpTier::Intrinsic => {
            "Source-backed row generated from the Category C operation catalog; a hardware-intrinsic id stays in its owning crate's namespace and passes hardware_conform.".to_string()
        }
        OpTier::Library => {
            "Source-backed row generated from vyre-foundation::operation; library ids must stay in the vyre-libs namespace.".to_string()
        }
        OpTier::External => {
            "Source-backed row generated from vyre-foundation::operation for an external consumer crate.".to_string()
        }
        OpTier::Foundation | OpTier::Unknown => String::new(),
        _ => String::new(),
    }
}

fn validate_records(records: &[OpRecord]) -> Result<(), String> {
    let mut families = BTreeSet::new();
    let mut ops = BTreeMap::<&str, &str>::new();
    for record in records {
        if !families.insert(record.family.as_str()) {
            return Err(format!(
                "Fix: duplicate OP_MATRIX family `{}`.",
                record.family
            ));
        }
        if record.owners.is_empty() {
            return Err(format!(
                "Fix: OP_MATRIX row `{}` has no owners.",
                record.family
            ));
        }
        if record.tests.is_empty() {
            return Err(format!(
                "Fix: OP_MATRIX row `{}` has no tests.",
                record.family
            ));
        }
        for op in &record.ops {
            if let Some(first_family) = ops.insert(op, record.family.as_str()) {
                return Err(format!(
                    "Fix: op `{op}` appears in both OP_MATRIX families `{first_family}` and `{}`.",
                    record.family
                ));
            }
            // ROADMAP S7: an op id's namespace classification must match
            // its row's declared tier. A Category C record must not carry
            // `vyre-libs::` ops, and a Category A record must not carry
            // `vyre-primitives::` ops. Mismatches were the root cause of
            // the original S7 finding (some intrinsics shipped under
            // Category A ids, making op truth ambiguous to the matrix).
            let observed = classify_op_id(op);
            if observed != OpTier::Unknown && tier_id_mismatch(record.tier, observed) {
                return Err(format!(
                    "Fix: op `{op}` is namespaced as {observed:?} but lives in OP_MATRIX family \
                     `{}` declared as {:?}. Move the id to the matching namespace, change the \
                     row tier, or split the row.",
                    record.family, record.tier,
                ));
            }
        }
    }
    Ok(())
}

/// Two operation tiers mismatch when one is `Intrinsic` and the other
/// is `Library` (or vice versa), the ownership distinction guarded here.
fn tier_id_mismatch(declared: OpTier, observed: OpTier) -> bool {
    matches!(
        (declared, observed),
        (OpTier::Intrinsic, OpTier::Library) | (OpTier::Library, OpTier::Intrinsic)
    )
}

fn render_matrix(records: &[OpRecord]) -> String {
    let mut out = String::new();
    out.push_str("# Canonical op/backend optimization and coverage matrix.\n");
    out.push_str("# Generated by `cargo_full run --bin xtask -- op-matrix --write` from inventory registries plus manual foundation rows.\n");
    out.push_str(
        "# Do not hand-edit generated rows; change the source registry or generator instead.\n\n",
    );
    out.push_str("schema = 2\n\n");
    out.push_str("backend_status_values = [\n");
    out.push_str("  \"supported\",\n  \"experimental\",\n  \"not_applicable\",\n  \"blocked_release\",\n]\n\n");
    out.push_str("tier_values = [\n");
    out.push_str("  \"foundation_ir\",\n  \"intrinsic\",\n  \"libs\",\n  \"external\",\n]\n\n");
    out.push_str(SCAN_CONSTRUCT_MATRIX);

    for record in records {
        out.push_str("[[op]]\n");
        push_string(&mut out, "family", &record.family);
        push_string(&mut out, "tier", record.tier.matrix_value());
        push_array(&mut out, "owners", &record.owners);
        push_array(&mut out, "ops", &record.ops);
        push_array(&mut out, "registry_sources", &record.registry_sources);
        if record.duplicate_ok {
            out.push_str("duplicate_ok = true\n");
        }
        push_string(&mut out, "reference", record.reference);
        push_string(&mut out, "foundation_ir", record.foundation_ir);
        push_string(&mut out, "cuda", record.cuda);
        push_string(&mut out, "wgpu", record.wgpu);
        push_string(&mut out, "spirv", record.spirv);
        push_string(
            &mut out,
            "release_blocking_notes",
            &record.release_blocking_notes,
        );
        push_array(&mut out, "tests", &record.tests);
        push_array(&mut out, "bench_targets", &record.bench_targets);
        out.push('\n');
    }
    out
}

fn push_string(out: &mut String, key: &str, value: &str) {
    out.push_str(key);
    out.push_str(" = ");
    out.push_str(&format!("{value:?}"));
    out.push('\n');
}

fn push_array(out: &mut String, key: &str, values: &[String]) {
    out.push_str(key);
    out.push_str(" = [");
    if values.is_empty() {
        out.push_str("]\n");
        return;
    }
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{value:?}"));
    }
    out.push_str("]\n");
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_OP_MATRIX_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_OP_MATRIX_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_OP_MATRIX_TEXT_BYTES} byte op matrix read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
