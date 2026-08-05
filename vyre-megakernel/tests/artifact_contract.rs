//! Positive, negative, boundary, and adversarial compiler-boundary contracts.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, TensorContract, ValueLifetime,
};
use vyre_megakernel::{
    compile, ArtifactRoute, CompileOptions, DependencyKind, DiagnosticCode, Digest,
    FusionPermission, OrderConstraint, ValidatedCompileRequest,
};

const LIMIT: u64 = 1_000_000;

fn contract(access: BufferAccess, lifetime: ValueLifetime, shape: Vec<ShapeDim>) -> TensorContract {
    TensorContract {
        dtype: DataType::U32,
        shape,
        access,
        lifetime,
    }
}

fn unary_program(input: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            vyre_foundation::ir::Expr::u32(0),
            vyre_foundation::ir::Expr::load(input, vyre_foundation::ir::Expr::u32(0)),
        )],
    )
}

fn permutation_fixture(reverse: bool) -> ProgramGraph {
    let input_contract = contract(
        BufferAccess::ReadOnly,
        ValueLifetime::Invocation,
        vec![ShapeDim::Symbol("items".into())],
    );
    let output_contract = contract(
        BufferAccess::ReadWrite,
        ValueLifetime::Output,
        vec![ShapeDim::Symbol("items".into())],
    );
    let mut graph = ProgramGraph::new();
    let names = if reverse { ["right", "left"] } else { ["left", "right"] };
    let mut ids = BTreeMap::new();
    for name in names {
        ids.insert(
            name,
            graph
                .add_external_value(name, input_contract.clone())
                .expect("fixture external value must connect"),
        );
    }
    let node_names = if reverse { ["beta", "alpha"] } else { ["alpha", "beta"] };
    for name in node_names {
        let input = if name == "alpha" { "left" } else { "right" };
        graph
            .add_node(
                name,
                unary_program(input, &format!("{name}_out")),
                vec![GraphInput {
                    buffer: input.into(),
                    value: ids[input],
                    contract: input_contract.clone(),
                }],
                vec![GraphOutput {
                    buffer: format!("{name}_out"),
                    name: format!("{name}.result"),
                    contract: output_contract.clone(),
                    state_successor_of: None,
                }],
            )
            .expect("fixture node must connect");
    }
    graph
}

fn options(route: ArtifactRoute) -> CompileOptions {
    let mut options = CompileOptions::new(route, BTreeMap::from([("items".into(), 17)]), LIMIT);
    options.order_constraints.push(OrderConstraint {
        before: "alpha".into(),
        after: "beta".into(),
    });
    options.fusion_permissions.push(FusionPermission {
        before: "alpha".into(),
        after: "beta".into(),
        legality_digest: Digest([7; 32]),
    });
    options
}

/// Positive: canonical bytes authenticate, round-trip, and expose every neutral record tier.
#[test]
fn canonical_artifact_round_trip_preserves_digest_and_records() {
    for route in [ArtifactRoute::Static, ArtifactRoute::Persistent] {
        let request = ValidatedCompileRequest::new(permutation_fixture(false), options(route))
            .expect("valid request must pass validation");
        let artifact = compile(&request).expect("valid request must compile");
        let bytes = artifact.to_bytes().expect("artifact must encode");
        let decoded = vyre_megakernel::MegakernelArtifact::from_bytes(&bytes)
            .expect("canonical artifact must decode");

        assert_eq!(decoded, artifact);
        assert_eq!(decoded.digest(), artifact.digest());
        assert_eq!(decoded.route(), route);
        assert_eq!(decoded.nodes().len(), 2);
        assert!(decoded.dependencies().iter().any(|edge| edge.kind == DependencyKind::Order));
        assert_eq!(decoded.fusion().len(), 1);
        assert_eq!(decoded.fusion()[0].members.len(), 2);
        assert_eq!(decoded.geometry().len(), 2);
        assert_eq!(decoded.resources().len(), 4);
        assert_eq!(decoded.resource_envelope().total_bytes, 17 * 4 * 4);
        assert_eq!(decoded.materializations().len(), 2);
    }
}
/// Positive: conservative compilation keeps unproven nodes separate and records their barrier.
#[test]
fn unproven_fusion_produces_singletons_and_dependency_barrier() {
    let mut compile_options = options(ArtifactRoute::Static);
    compile_options.fusion_permissions.clear();
    let request = ValidatedCompileRequest::new(permutation_fixture(false), compile_options)
        .expect("conservative request must validate");
    let artifact = compile(&request).expect("conservative request must compile");

    assert_eq!(artifact.fusion().len(), 2);
    assert!(artifact.fusion().iter().all(|group| group.members.len() == 1));
    assert_eq!(artifact.barriers().len(), 1);
    assert_eq!(artifact.barriers()[0].before_stage, 0);
    assert_eq!(artifact.barriers()[0].after_stage, 1);
    assert_eq!(artifact.barriers()[0].dependencies.len(), 1);
}


/// Adversarial: declaration-order permutations cannot perturb source, request, artifact, or digest identity.
#[test]
fn graph_permutation_produces_identical_canonical_artifact() {
    let forward = ValidatedCompileRequest::new(permutation_fixture(false), options(ArtifactRoute::Static))
        .expect("forward request must validate");
    let reverse = ValidatedCompileRequest::new(permutation_fixture(true), options(ArtifactRoute::Static))
        .expect("reverse request must validate");
    let forward = compile(&forward).expect("forward request must compile");
    let reverse = compile(&reverse).expect("reverse request must compile");

    assert_eq!(forward.provenance(), reverse.provenance());
    assert_eq!(forward.digest(), reverse.digest());
    assert_eq!(forward.to_bytes().unwrap(), reverse.to_bytes().unwrap());
}

/// Negative: unknown endpoints and self edges fail before an invalid request value can exist.
#[test]
fn malformed_named_edges_have_stable_paths_and_codes() {
    let mut unknown = options(ArtifactRoute::Static);
    unknown.order_constraints[0].after = "absent".into();
    let error = ValidatedCompileRequest::new(permutation_fixture(false), unknown)
        .err()
        .expect("unknown endpoint must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::UnknownNode);
    assert_eq!(error.diagnostic.path, "options.order_constraints[0].after");

    let mut self_edge = options(ArtifactRoute::Static);
    self_edge.order_constraints[0].after = "alpha".into();
    let error = ValidatedCompileRequest::new(permutation_fixture(false), self_edge)
        .err()
        .expect("self edge must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::SelfEdge);
}

/// Adversarial: individually well-formed order facts cannot smuggle a cycle into the artifact.
#[test]
fn cyclic_order_constraints_are_rejected_deterministically() {
    let mut options = options(ArtifactRoute::Static);
    options.fusion_permissions.clear();
    options.order_constraints.push(OrderConstraint {
        before: "beta".into(),
        after: "alpha".into(),
    });
    let request = ValidatedCompileRequest::new(permutation_fixture(false), options)
        .expect("named facts are individually well formed");
    let error = compile(&request).expect_err("cycle must fail compilation");
    assert_eq!(error.diagnostic.code, DiagnosticCode::DependencyCycle);
    assert_eq!(error.diagnostic.path, "artifact.dependencies");
}

/// Boundary: exact artifact limits succeed while one byte below the canonical length fails.
#[test]
fn artifact_byte_limit_is_inclusive_and_checked() {
    let request = ValidatedCompileRequest::new(
        permutation_fixture(false),
        options(ArtifactRoute::Static),
    )
    .expect("baseline request must validate");
    let len = compile(&request)
        .expect("baseline request must compile")
        .to_bytes()
        .unwrap()
        .len() as u64;

    let mut exact = options(ArtifactRoute::Static);
    exact.max_artifact_bytes = len;
    let exact = ValidatedCompileRequest::new(permutation_fixture(false), exact).unwrap();
    compile(&exact).expect("exact byte limit is inclusive");

    let mut short = options(ArtifactRoute::Static);
    short.max_artifact_bytes = len - 1;
    let short = ValidatedCompileRequest::new(permutation_fixture(false), short).unwrap();
    let error = compile(&short).expect_err("one-byte-short limit must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::ArtifactLimit);
}

/// Boundary: checked dimension multiplication reports overflow rather than wrapping a resource size.
#[test]
fn resource_shape_overflow_has_stable_diagnostic() {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "huge",
            contract(
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
                vec![ShapeDim::Known(u64::MAX), ShapeDim::Known(2)],
            ),
        )
        .unwrap();
    let request = ValidatedCompileRequest::new(
        graph,
        CompileOptions::new(ArtifactRoute::Static, BTreeMap::new(), LIMIT),
    )
    .unwrap();
    let error = compile(&request).expect_err("overflow must fail closed");
    assert_eq!(error.diagnostic.code, DiagnosticCode::ResourceOverflow);
    assert_eq!(error.diagnostic.path, "graph.values[huge].shape");
}

/// Negative: framing schema skew is diagnosed before body decoding or digest admission.
#[test]
fn artifact_version_skew_has_stable_diagnostic() {
    let request = ValidatedCompileRequest::new(
        permutation_fixture(false),
        options(ArtifactRoute::Static),
    )
    .unwrap();
    let artifact = compile(&request).unwrap();
    let mut bytes = artifact.to_bytes().unwrap();
    bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
    let error = vyre_megakernel::MegakernelArtifact::from_bytes(&bytes)
        .expect_err("unknown schema must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::VersionSkew);
    assert_eq!(error.diagnostic.path, "artifact.schema_version");
}

/// Adversarial: body mutation with intact framing is rejected by the domain-separated digest.
#[test]
fn artifact_body_tampering_is_detected() {
    let request = ValidatedCompileRequest::new(
        permutation_fixture(false),
        options(ArtifactRoute::Static),
    )
    .unwrap();
    let artifact = compile(&request).unwrap();
    let mut bytes = artifact.to_bytes().unwrap();
    bytes[10] ^= 1;
    let error = vyre_megakernel::MegakernelArtifact::from_bytes(&bytes)
        .expect_err("tampered body must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::DigestMismatch);
}

/// Boundary: the published crate has one internal production edge and no substrate-specific vocabulary.
#[test]
fn manifest_and_public_boundary_remain_neutral() {
    let manifest = include_str!("../Cargo.toml");
    let source = include_str!("../src/lib.rs").to_ascii_lowercase();
    assert!(manifest.contains("vyre-foundation.workspace = true"));
    for forbidden_dependency in ["vyre-driver", "vyre-runtime", "vyre-lower", "vyre-primitives"] {
        assert!(!manifest.contains(forbidden_dependency));
    }
    for forbidden_term in ["cuda", "wgpu", "metal", "qwen", "device", "model"] {
        assert!(!source.contains(forbidden_term), "found forbidden term {forbidden_term}");
    }
    assert_eq!(source.matches("pub fn compile(").count(), 1);
}
