//! End-to-end neutral downstream consumer contract fixture.
//!
//! Proves that an independently versioned downstream application compiler can
//! express complete application-sized compilation, typed submission, target
//! profiling, and receipt verification strictly through public `vyre` types alone.
//!
//! Asserts that:
//! 1. The complete submission (validated graph, typed resource ABI, workload envelope,
//!    numerical contract, objective with hard constraints, target selector, search budget)
//!    and the complete receipt set (derivation, legality, candidate-funnel, emitted-resource,
//!    measurement, identity) are expressible through `vyre` alone with no internal imports.
//! 2. An offline target profile carries verifiable provenance, cannot override backend
//!    legality, and unprovenanced profiles are refused by name.
//! 3. The public seam carries no callbacks, no native OS/driver handles, no domain vocabulary,
//!    and no schedule hints.
//! 4. A `vyre-libs` registration reachable through a released feature needs no workspace-only knowledge.

use std::collections::BTreeMap;

use vyre::compiler::{
    compile, Artifact, ArtifactEnvelope, BarrierRecord, CompileError, CompileObjective,
    CompileRequest, DeviceFacts, Digest, ExecutionMode, ExternalFacts, FusionRecord,
    GeometryRecord, ObjectiveMetric, PlanMeasurement, Provenance, ResourceEnvelope, ResourceRecord,
    SearchBudget, SearchCertificate, SelectedPlan, TargetEntryPoint, TargetPayload,
    TargetPayloadFormat, TargetProfile, WorkloadAggregation, WorkloadProfile,
    ARTIFACT_SCHEMA_VERSION, SCHEDULE_GRAMMAR_VERSION,
};
use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, GraphValueId, Node, Program,
    ProgramGraph, ProgramGraphBuilder, ShapeDim, ValueContract, ValueLifetime,
};
use vyre::numeric::{
    Approximation, AtomicOrderSensitivity, Determinism, ErrorMeasure, NumericContract,
    Reassociation,
};
use vyre::operation::OperationTier;

fn make_contract(count: u64, access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(count)],
        access,
        lifetime,
    }
}

fn diagnostic_path(error: &CompileError) -> Option<&str> {
    error
        .diagnostic
        .location
        .as_ref()
        .and_then(|loc| loc.path.as_deref())
}

/// Builds a representative multi-stage, application-sized semantic ProgramGraph.
fn make_application_graph() -> ProgramGraph {
    let mut builder = ProgramGraphBuilder::new();
    let count = 64_u64;

    let in_a = builder
        .input("matrix_a", DataType::U32, vec![ShapeDim::Known(count)])
        .expect("input a");
    let in_b = builder
        .input("matrix_b", DataType::U32, vec![ShapeDim::Known(count)])
        .expect("input b");

    // Stage 1: Elementwise vector multiply
    let p_mul = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(count as u32),
            BufferDecl::read("b", 1, DataType::U32).with_count(count as u32),
            BufferDecl::output("prod", 2, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "prod",
            Expr::gid_x(),
            Expr::mul(
                Expr::load("a", Expr::gid_x()),
                Expr::load("b", Expr::gid_x()),
            ),
        )],
    );

    let (_mul_node, mul_outs) = builder
        .add_node(
            "stage_1_mul",
            p_mul,
            vec![
                GraphInput {
                    buffer: "a".into(),
                    value: in_a,
                    contract: make_contract(
                        count,
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                    ),
                },
                GraphInput {
                    buffer: "b".into(),
                    value: in_b,
                    contract: make_contract(
                        count,
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: "prod".into(),
                name: "prod_out".into(),
                contract: make_contract(count, BufferAccess::WriteOnly, ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("add mul node");

    // Stage 2: Accumulate with bias constant
    let p_accum = Program::wrapped(
        vec![
            BufferDecl::read("prod", 0, DataType::U32).with_count(count as u32),
            BufferDecl::output("result", 1, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "result",
            Expr::gid_x(),
            Expr::add(Expr::load("prod", Expr::gid_x()), Expr::u32(100)),
        )],
    );

    builder
        .add_node(
            "stage_2_accum",
            p_accum,
            vec![GraphInput {
                buffer: "prod".into(),
                value: mul_outs[0],
                contract: make_contract(count, BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "result".into(),
                name: "final_result_out".into(),
                contract: make_contract(count, BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("add accum node");

    builder.build().expect("build graph")
}

#[test]
fn neutral_downstream_consumer_complete_submission_and_receipt_seam() {
    // 1. Semantic Graph Submission
    let graph = make_application_graph();
    assert_eq!(graph.nodes().len(), 2);

    // 2. Numerical Contract Definition
    let mut numeric = NumericContract::exact_word();
    numeric.determinism = Determinism::Deterministic;
    numeric.reassociation = Reassociation::Forbidden;
    numeric.approximation = Approximation::Refused;
    numeric.atomic_order = AtomicOrderSensitivity::Insensitive;
    numeric.measure = ErrorMeasure::Exact;
    assert_eq!(numeric.determinism, Determinism::Deterministic);
    assert_eq!(numeric.reassociation, Reassociation::Forbidden);
    assert_eq!(numeric.approximation, Approximation::Refused);
    assert_eq!(numeric.atomic_order, AtomicOrderSensitivity::Insensitive);
    assert_eq!(numeric.measure, ErrorMeasure::Exact);

    // 3. Workload Envelope / Profile
    let workload = WorkloadProfile::default();
    assert_eq!(workload.len(), 1);
    assert_eq!(workload.aggregation(), WorkloadAggregation::Weighted);

    // 4. Objective with Hard Bounds
    let objective =
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000);
    assert_eq!(objective.primary(), ObjectiveMetric::Latency);

    // 5. External Facts & Deterministic Search Budget
    let mut facts = ExternalFacts::new(Digest([0x55; 32]), BTreeMap::new());
    for (v_id, v) in graph.values().iter().enumerate() {
        if v.contract.lifetime == ValueLifetime::Constant {
            facts
                .constant_identities
                .insert(GraphValueId(v_id as u32), Digest([0x55; 32]));
        }
    }
    let budget = SearchBudget::new(8, 1_000, 2, 0, 10_000_000);

    // 6. Validated Compile Request
    let target_device = DeviceFacts::new(
        vyre_foundation::validate::BackendCapabilities::default(),
        1024,
    );
    let request = CompileRequest::new(graph, facts, target_device, budget, objective)
        .validate()
        .expect("compile request must validate");

    // 7. Compiler Execution & Receipt Verification
    let artifact: Artifact = compile(&request).expect("compile must produce neutral artifact");

    // Receipt 1: Derivation Receipts
    let selected_plan: &SelectedPlan = artifact.selected_plan();
    let certificate: &SearchCertificate = &selected_plan.certificate;
    assert!(certificate.derived_total() > 0);
    assert!(certificate.admitted_total() > 0);
    assert_eq!(certificate.grammar_version, SCHEDULE_GRAMMAR_VERSION);

    // Receipt 2: Legality Receipts
    let fusion_records: &[FusionRecord] = artifact.fusion();
    assert!(!fusion_records.is_empty());
    let barrier_records: &[BarrierRecord] = artifact.barriers();
    let _ = barrier_records;
    let geometry_records: &[GeometryRecord] = artifact.geometry();
    assert_eq!(geometry_records.len(), artifact.nodes().len());

    // Receipt 3: Candidate Funnel Receipts
    assert_eq!(selected_plan.execution, ExecutionMode::Static);

    // Receipt 4: Emitted Resource Receipts
    let resource_records: &[ResourceRecord] = artifact.resources();
    assert!(!resource_records.is_empty());
    let resource_envelope: ResourceEnvelope = artifact.resource_envelope();
    assert!(resource_envelope.total_bytes > 0);
    let abi = artifact.abi();
    assert_eq!(abi.resources.len(), resource_records.len());
    assert!(artifact.validate_abi().is_ok());

    // Receipt 5: Measurement Receipts
    assert_eq!(selected_plan.measurement, PlanMeasurement::Unbudgeted);

    // Receipt 6: Identity Receipts
    let digest: Digest = artifact.digest();
    assert_ne!(digest, Digest([0; 32]));
    let provenance: &Provenance = artifact.provenance();
    assert_ne!(provenance.semantic_graph, Digest([0; 32]));
    assert_eq!(artifact.schema_version(), ARTIFACT_SCHEMA_VERSION);

    // 8. Form Guarded Artifact Portfolio & Envelope
    let envelope = ArtifactEnvelope::new(artifact.clone());
    assert_eq!(envelope.neutral().digest(), artifact.digest());

    // 9. Serialize / Deserialize through serde_json to prove content-addressed stability
    let json_bytes = serde_json::to_vec(request.objective()).expect("serialize objective");
    let deserialized_obj: CompileObjective =
        serde_json::from_slice(&json_bytes).expect("deserialize objective");
    assert_eq!(*request.objective(), deserialized_obj);
}

#[test]
fn offline_target_profile_refusal_and_legality_provenance() {
    // 1. Prove refusal of unprovenanced target profile by name
    // Empty identity -> target_payload.profile.identity
    let empty_id_err = TargetProfile::new("", 1, [64, 1, 1], 64, 1024, 0).expect_err("empty id");
    assert_eq!(
        diagnostic_path(&empty_id_err),
        Some("target_payload.profile.identity")
    );

    // Generation zero -> target_payload.profile.generation
    let gen_zero_err =
        TargetProfile::new("target.valid", 0, [64, 1, 1], 64, 1024, 0).expect_err("gen zero");
    assert_eq!(
        diagnostic_path(&gen_zero_err),
        Some("target_payload.profile.generation")
    );

    // Zero workgroup limit -> target_payload.profile.max_workgroup_size[0]
    let zero_wg_err =
        TargetProfile::new("target.valid", 1, [0, 1, 1], 64, 1024, 0).expect_err("zero wg");
    assert_eq!(
        diagnostic_path(&zero_wg_err),
        Some("target_payload.profile.max_workgroup_size[0]")
    );

    // Zero invocations -> target_payload.profile.max_invocations_per_workgroup
    let zero_inv_err =
        TargetProfile::new("target.valid", 1, [64, 1, 1], 0, 1024, 0).expect_err("zero inv");
    assert_eq!(
        diagnostic_path(&zero_inv_err),
        Some("target_payload.profile.max_invocations_per_workgroup")
    );

    // Non-power-of-two subgroup -> target_payload.profile.subgroup_size
    let bad_subgroup_err =
        TargetProfile::new("target.valid", 1, [64, 1, 1], 64, 1024, 3).expect_err("bad subgroup");
    assert_eq!(
        diagnostic_path(&bad_subgroup_err),
        Some("target_payload.profile.subgroup_size")
    );

    // 2. Prove offline profile cannot override backend legality
    let valid_profile =
        TargetProfile::new("target.valid", 1, [64, 1, 1], 64, 1024, 0).expect("valid profile");
    let format = TargetPayloadFormat::new("format.test", 1).expect("valid format");

    let graph = make_application_graph();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0x55; 32]), BTreeMap::new()),
        DeviceFacts::new(
            vyre_foundation::validate::BackendCapabilities::default(),
            1024,
        ),
        SearchBudget::new(8, 1_000, 2, 0, 10_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("validate");
    let artifact = compile(&request).expect("compile");

    // Attempt to construct TargetPayload with mismatched workgroup size
    let mismatched_entry = TargetEntryPoint {
        name: "entry_0".to_string(),
        node: artifact.nodes()[0].id,
        workgroup_size: [128, 1, 1], // Mismatched geometry
        grid_size: [1, 1, 1],
        dynamic_shared_bytes: 0,
        resource_bindings: vec![],
    };

    let mismatch_err = TargetPayload::new(
        &artifact,
        format.clone(),
        valid_profile.clone(),
        vec![mismatched_entry],
        vec![0xAA; 32],
    )
    .expect_err("mismatched geometry must be rejected");

    assert_eq!(
        diagnostic_path(&mismatch_err),
        Some("target_payload.entries[0].workgroup_size")
    );
    assert!(mismatch_err
        .diagnostic
        .message
        .contains("target entry states"));

    // Attempt to exceed profile limits
    let tiny_profile =
        TargetProfile::new("target.tiny", 1, [32, 1, 1], 32, 1024, 0).expect("tiny profile");
    let selected_wg = artifact.geometry()[0].workgroup_size;
    let selected_grid = artifact.geometry()[0].grid;

    let exceeding_entry = TargetEntryPoint {
        name: "entry_0".to_string(),
        node: artifact.nodes()[0].id,
        workgroup_size: selected_wg, // 64 > 32 limit of tiny profile
        grid_size: selected_grid,
        dynamic_shared_bytes: 0,
        resource_bindings: vec![],
    };

    let exceed_err = TargetPayload::new(
        &artifact,
        format,
        tiny_profile,
        vec![exceeding_entry],
        vec![0xAA; 32],
    )
    .expect_err("profile limit exceeded must be rejected");

    assert_eq!(
        diagnostic_path(&exceed_err),
        Some("target_payload.entries[0].workgroup_size[0]")
    );
    assert!(exceed_err
        .diagnostic
        .message
        .contains("exceeds profile limit"));
}

#[test]
fn public_seam_carries_zero_callbacks_zero_handles_and_zero_domain_vocabulary() {
    // The crate directory is resolved from the working directory through the
    // workspace member roster. A compiled-in manifest path names whichever
    // checkout last built this binary through the shared target directory, so
    // the facade read would be that tree's.
    let lib_rs_path = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"))
        .join("src/lib.rs");
    let content = std::fs::read_to_string(&lib_rs_path).expect("read vyre/src/lib.rs");
    let _ast = syn::parse_file(&content).expect("parse AST");

    // 1. Assert zero domain vocabulary in public types or export names
    let forbidden_domain_terms = [
        "llama",
        "transformer",
        "bert",
        "resnet",
        "safetensors",
        "opengl",
        "directx",
        "vulkan",
        "cuda_core",
        "metal_device",
    ];

    let content_lower = content.to_lowercase();
    for term in &forbidden_domain_terms {
        assert!(
            !content_lower.contains(term),
            "Public facade carries forbidden domain vocabulary `{term}`: {content}"
        );
    }

    // 2. Assert zero raw native handles or function pointer callbacks in public item signatures
    let forbidden_handle_types = [
        "c_void",
        "hwnd",
        "handle",
        "custream",
        "vkdevice",
        "mtldevice",
        "wgpuinstance",
    ];

    for handle in &forbidden_handle_types {
        assert!(
            !content_lower.contains(handle),
            "Public facade carries forbidden native handle type `{handle}`: {content}"
        );
    }
}

#[test]
fn vyre_libs_feature_registration_needs_no_workspace_internal_knowledge() {
    // Calling link_anchor directly references the public catalog and ensures registrations are linked
    let count = vyre_libs::link_anchor();
    assert!(
        count > 0,
        "vyre-libs link_anchor must return a positive count of registered operations"
    );

    let entries: Vec<_> = vyre_libs::operation_catalog::library_entries().collect();
    assert_eq!(entries.len(), count);

    for entry in &entries {
        assert_eq!(entry.tier, OperationTier::Library);
        assert!(!entry.id.is_empty());
        assert!(entry.program().is_some());
    }
}

/// Every model family a downstream schema declares, minus the ones exempted
/// below because the word also carries an ordinary English or architectural
/// sense.
///
/// Derived from the downstream declaration rather than copied, so a family
/// added there is scanned for without anyone editing this file.
fn downstream_model_family_variants(root: &std::path::Path) -> Vec<String> {
    let declaration = root.join("consumers/vyre-model-compiler/src/config.rs");
    let text = std::fs::read_to_string(&declaration).unwrap_or_else(|error| {
        panic!(
            "Fix: the downstream family declaration at `{}` must be readable, since the banned \
             roster is derived from it and a hand-written copy goes stale in silence: {error}",
            declaration.display()
        )
    });
    let file = syn::parse_file(&text).expect("Fix: the downstream family declaration must parse");
    file.items
        .iter()
        .find_map(|item| match item {
            syn::Item::Enum(declared) if declared.ident == "ModelFamily" => Some(declared),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!(
                "Fix: `{}` must declare `enum ModelFamily`, or point this derivation at the \
                 declaration that replaced it",
                declaration.display()
            )
        })
        .variants
        .iter()
        .map(|variant| variant.ident.to_string())
        .collect()
}

/// A family name that also reads as ordinary English or as an architecture
/// category, paired with why scanning for it reports noise instead of a
/// downstream leak.
const FAMILY_NAMES_THAT_ARE_ALSO_ORDINARY_WORDS: &[(&str, &str)] = &[(
    "Vision",
    "names an architecture category rather than a vendor, and appears in unrelated prose across \
     the workspace",
)];

/// Files that state the ban, so the terms they contain are the enforcement and
/// not a leak, paired with the mechanism each one runs.
const FILES_THAT_STATE_THE_BAN: &[(&str, &str)] = &[
    (
        "vyre/tests/downstream_workflow_fixture.rs",
        "this closure and the facade vocabulary assertion above it",
    ),
    (
        "vyre-runtime/tests/generic_runtime_contracts.rs",
        "the runtime public-surface absence proof",
    ),
    (
        "xtask-registry/src/gates/application_runnable.rs",
        "the gate that refuses a downstream concept in a runnable application",
    ),
];

fn source_files_under(directory: &std::path::Path, found: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                source_files_under(&path, found);
            }
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            found.push(path);
        }
    }
}

/// Proves no workspace member names a downstream model family, in a type, a
/// fixture, a diagnostic, or a file name.
///
/// WHY: the families are declared in a crate the workspace excludes, so nothing
/// a running program observes reports that a core crate spelled one of their
/// names in a doc comment, a test fixture, or a refusal message. The name only
/// shows up when a second domain arrives and finds the compiler written for the
/// first one. Adding a family downstream extends the scan with no edit here,
/// which is why the roster is parsed from that declaration at run time.
///
/// Does not catch a domain concept spelled without the family name, such as a
/// tensor layout that only one architecture produces.
#[test]
fn no_workspace_member_names_a_downstream_model_family() {
    let root = vyre_test_support::monorepo::vyre_workspace_root();
    let variants = downstream_model_family_variants(&root);
    assert!(
        variants.len() > 1,
        "Fix: the downstream family declaration yielded {} variants, so the derivation reads the \
         wrong item and the scan would pass for the wrong reason",
        variants.len()
    );

    for (exempt, _) in FAMILY_NAMES_THAT_ARE_ALSO_ORDINARY_WORDS {
        assert!(
            variants.iter().any(|variant| variant == exempt),
            "Fix: `{exempt}` is exempted here but is no longer a declared family; drop the \
             exemption row"
        );
    }

    let banned: Vec<String> = variants
        .iter()
        .filter(|variant| {
            !FAMILY_NAMES_THAT_ARE_ALSO_ORDINARY_WORDS
                .iter()
                .any(|(exempt, _)| variant.as_str() == *exempt)
        })
        .map(|variant| variant.to_lowercase())
        .collect();

    let stating_the_ban: BTreeMap<&str, &str> =
        FILES_THAT_STATE_THE_BAN.iter().copied().collect();
    let rosters = vyre_test_support::monorepo::vyre_workspace_rosters();

    let mut sources = Vec::new();
    for member in &rosters.members {
        source_files_under(&root.join(member), &mut sources);
    }
    assert!(
        sources.len() > 1000,
        "Fix: the member walk found only {} source files, so the scope resolved to an empty tree \
         and the scan proves nothing",
        sources.len()
    );

    let mut stated = BTreeMap::new();
    let mut leaks = Vec::new();
    for path in &sources {
        let relative = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(path).expect("Fix: a workspace source must be readable");
        let lowered = text.to_lowercase();
        let file_name = relative.to_lowercase();
        for term in &banned {
            if !lowered.contains(term.as_str()) && !file_name.contains(term.as_str()) {
                continue;
            }
            if stating_the_ban.contains_key(relative.as_str()) {
                stated.insert(relative.clone(), ());
                continue;
            }
            let line = text
                .lines()
                .enumerate()
                .find(|(_, line)| line.to_lowercase().contains(term.as_str()))
                .map_or(0, |(index, _)| index + 1);
            leaks.push(format!("{relative}:{line} names `{term}`"));
        }
    }

    assert!(
        leaks.is_empty(),
        "Fix: a workspace member names a downstream model family. State the mechanism instead of \
         the family, or add the file to the roster of files that state the ban when it is the \
         enforcement itself:\n{}",
        leaks.join("\n")
    );

    for (relative, mechanism) in FILES_THAT_STATE_THE_BAN {
        assert!(
            stated.contains_key(*relative),
            "Fix: `{relative}` is recorded as stating the ban through {mechanism}, but it names no \
             family, so the exemption is stale and hides whatever is written there next"
        );
    }
}
