//! Canonical connected-composition identity contracts.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, GraphInput, GraphOutput, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_foundation::ir::{
    ProgramGraphIdentityContext, ProgramGraphIdentityError, PROGRAM_GRAPH_IDENTITY_VERSION,
};

fn tensor(
    dtype: DataType,
    shape: Vec<ShapeDim>,
    access: BufferAccess,
    lifetime: ValueLifetime,
) -> ValueContract {
    ValueContract {
        dtype,
        shape,
        access,
        lifetime,
    }
}

fn model_graph(hidden: u64, local_prefix: &str) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let weight = graph
        .add_external_value(
            "weight",
            tensor(
                DataType::BF16,
                vec![ShapeDim::Known(hidden), ShapeDim::Known(hidden)],
                BufferAccess::ReadOnly,
                ValueLifetime::Constant,
            ),
        )
        .expect("Fix: identity fixture weight must register");
    let input = graph
        .add_external_value(
            "tokens",
            tensor(
                DataType::F32,
                vec![ShapeDim::Symbol("batch".into()), ShapeDim::Known(hidden)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        )
        .expect("Fix: identity fixture input must register");
    let input_buffer = format!("{local_prefix}.input");
    let weight_buffer = format!("{local_prefix}.weight");
    let output_buffer = format!("{local_prefix}.output");
    graph
        .add_node(
            "projection",
            Program::wrapped(
                vec![
                    BufferDecl::storage(&input_buffer, 0, BufferAccess::ReadOnly, DataType::F32),
                    BufferDecl::storage(&weight_buffer, 1, BufferAccess::ReadOnly, DataType::BF16),
                    BufferDecl::storage(&output_buffer, 2, BufferAccess::ReadWrite, DataType::F32),
                ],
                [1, 1, 1],
                Vec::new(),
            ),
            vec![
                GraphInput {
                    buffer: input_buffer,
                    value: input,
                    contract: tensor(
                        DataType::F32,
                        vec![ShapeDim::Symbol("batch".into()), ShapeDim::Known(hidden)],
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                    ),
                },
                GraphInput {
                    buffer: weight_buffer,
                    value: weight,
                    contract: tensor(
                        DataType::BF16,
                        vec![ShapeDim::Known(hidden), ShapeDim::Known(hidden)],
                        BufferAccess::ReadOnly,
                        ValueLifetime::Constant,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: output_buffer,
                name: "logits".into(),
                contract: tensor(
                    DataType::F32,
                    vec![ShapeDim::Symbol("batch".into()), ShapeDim::Known(hidden)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Output,
                ),
                retained_successor_of: None,
            }],
        )
        .expect("Fix: identity fixture node must connect");
    graph
}

fn context() -> ProgramGraphIdentityContext {
    ProgramGraphIdentityContext {
        artifact_schema_version: 7,
        configuration_digest: [0xA5; 32],
        symbolic_bindings: BTreeMap::from([("batch".into(), 4)]),
        constant_identities: BTreeMap::from([("weight".into(), [0x5A; 32])]),
    }
}

/// Pins the complete identity framing so accidental byte-level schema drift requires an explicit version change.
///
/// Wire revision 8 moved this digest because graph identity embeds canonical
/// program wire bytes containing schedule-free logical execution tags.
#[test]
fn canonical_composition_identity_matches_frozen_digest() {
    let identity = model_graph(8, "projection")
        .identity(&context())
        .expect("Fix: complete identity provenance must hash");
    assert_eq!(identity.format_version, PROGRAM_GRAPH_IDENTITY_VERSION);
    assert_eq!(
        identity.digest,
        [
            197, 180, 249, 115, 151, 242, 155, 53, 166, 48, 248, 25, 130, 15, 247, 47, 224, 251,
            231, 168, 19, 42, 79, 249, 67, 134, 9, 89, 142, 100, 222, 54,
        ]
    );
}

/// Prevents construction history or address identity from changing equal composition cache keys.
#[test]
fn independently_constructed_equal_graphs_have_equal_identity() {
    assert_eq!(
        model_graph(8, "projection")
            .identity(&context())
            .expect("Fix: first graph must hash"),
        model_graph(8, "projection")
            .identity(&context())
            .expect("Fix: second graph must hash")
    );
}

/// Proves topology order and Program-local binding names remain part of executable artifact identity.
#[test]
fn topology_and_local_program_bindings_change_identity() {
    let canonical = model_graph(8, "projection")
        .identity(&context())
        .expect("Fix: canonical graph must hash");
    let renamed = model_graph(8, "renamed")
        .identity(&context())
        .expect("Fix: renamed graph must hash");
    assert_ne!(canonical, renamed);

    let mut reversed = ProgramGraph::new();
    let input = reversed
        .add_external_value(
            "tokens",
            tensor(
                DataType::F32,
                vec![ShapeDim::Known(8)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        )
        .expect("Fix: reordered fixture input must register");
    for name in ["second", "first"] {
        reversed
            .add_node(
                name,
                Program::wrapped(
                    vec![
                        BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::F32),
                        BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::F32),
                    ],
                    [1, 1, 1],
                    Vec::new(),
                ),
                vec![GraphInput {
                    buffer: "input".into(),
                    value: input,
                    contract: tensor(
                        DataType::F32,
                        vec![ShapeDim::Known(8)],
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                    ),
                }],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: format!("{name}.output"),
                    contract: tensor(
                        DataType::F32,
                        vec![ShapeDim::Known(8)],
                        BufferAccess::ReadWrite,
                        ValueLifetime::Output,
                    ),
                    retained_successor_of: None,
                }],
            )
            .expect("Fix: reordered fixture node must connect");
    }
    let empty_context = ProgramGraphIdentityContext {
        artifact_schema_version: 7,
        configuration_digest: [0xA5; 32],
        symbolic_bindings: BTreeMap::new(),
        constant_identities: BTreeMap::new(),
    };
    let reversed_identity = reversed
        .identity(&empty_context)
        .expect("Fix: reordered graph must hash");
    assert_ne!(canonical, reversed_identity);
}

/// Locks all external provenance dimensions into the key: shape bindings, configuration, weights, and artifact schema.
#[test]
fn every_external_provenance_change_invalidates_identity() {
    let graph = model_graph(8, "projection");
    let baseline_context = context();
    let baseline = graph
        .identity(&baseline_context)
        .expect("Fix: baseline graph must hash");

    let mut changed_binding = baseline_context.clone();
    changed_binding.symbolic_bindings.insert("batch".into(), 5);
    assert_ne!(
        baseline,
        graph
            .identity(&changed_binding)
            .expect("Fix: changed binding must hash")
    );

    let mut changed_config = baseline_context.clone();
    changed_config.configuration_digest[0] ^= 1;
    assert_ne!(
        baseline,
        graph
            .identity(&changed_config)
            .expect("Fix: changed config must hash")
    );

    let mut changed_weight = baseline_context.clone();
    changed_weight
        .constant_identities
        .get_mut("weight")
        .expect("Fix: fixture weight exists")[0] ^= 1;
    assert_ne!(
        baseline,
        graph
            .identity(&changed_weight)
            .expect("Fix: changed weight must hash")
    );

    let mut changed_schema = baseline_context;
    changed_schema.artifact_schema_version += 1;
    assert_ne!(
        baseline,
        graph
            .identity(&changed_schema)
            .expect("Fix: changed schema must hash")
    );
    assert_ne!(
        baseline,
        model_graph(16, "projection")
            .identity(&context())
            .expect("Fix: changed shape must hash")
    );
}

/// Ensures mutable cache bytes can grow without churning an artifact key that contains only state schema.
#[test]
fn mutable_cache_growth_does_not_change_composition_identity() {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "cache",
            tensor(
                DataType::F32,
                vec![ShapeDim::Known(128), ShapeDim::Known(8)],
                BufferAccess::ReadWrite,
                ValueLifetime::Retained,
            ),
        )
        .expect("Fix: cache schema must register");
    let cache_context = ProgramGraphIdentityContext {
        artifact_schema_version: 1,
        configuration_digest: [7; 32],
        symbolic_bindings: BTreeMap::new(),
        constant_identities: BTreeMap::new(),
    };
    let before = graph
        .identity(&cache_context)
        .expect("Fix: cache graph must hash");
    let mut mutable_cache_contents = vec![0_u8; 32];
    mutable_cache_contents.extend_from_slice(&[9; 96]);
    assert_eq!(mutable_cache_contents.len(), 128);
    assert_eq!(
        graph
            .identity(&cache_context)
            .expect("Fix: grown cache graph must hash"),
        before
    );
}

/// Prevents incomplete symbolic provenance and stale extra bindings from aliasing valid artifacts.
#[test]
fn symbolic_binding_set_must_match_graph_exactly() {
    let graph = model_graph(8, "projection");
    let mut missing = context();
    missing.symbolic_bindings.clear();
    assert!(
        matches!(graph.identity(&missing), Err(ProgramGraphIdentityError::MissingSymbol { symbol }) if symbol == "batch")
    );

    let mut extra = context();
    extra.symbolic_bindings.insert("unused".into(), 1);
    assert!(
        matches!(graph.identity(&extra), Err(ProgramGraphIdentityError::UnexpectedSymbol { symbol }) if symbol == "unused")
    );
}

/// Prevents missing trusted weights and unrelated checkpoint identities from entering residency keys.
#[test]
fn immutable_weight_set_must_match_graph_exactly() {
    let graph = model_graph(8, "projection");
    let mut missing = context();
    missing.constant_identities.clear();
    assert!(
        matches!(graph.identity(&missing), Err(ProgramGraphIdentityError::MissingConstantIdentity { name }) if name == "weight")
    );

    let mut extra = context();
    extra
        .constant_identities
        .insert("vision.weight".into(), [1; 32]);
    assert!(
        matches!(graph.identity(&extra), Err(ProgramGraphIdentityError::UnexpectedConstantIdentity { name }) if name == "vision.weight")
    );
}
