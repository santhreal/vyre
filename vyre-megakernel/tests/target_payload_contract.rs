//! Regression contracts for canonical neutral artifacts with attached target payloads.

use vyre_foundation::ir::Program;
use vyre_megakernel::{
    attach_target, compile_selected_modules, Artifact, ArtifactEnvelope, ArtifactNodeId,
    ArtifactValueId, CompileError, EmittedTargetModule, FusionGroupId, TargetCompileError,
    TargetCompiler, TargetEntryPoint, TargetModuleBundle, TargetModuleImage, TargetPayload,
    TargetPayloadFormat, TargetProfile, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};

#[path = "../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;

use artifact_fixtures::{entry_point, neutral_artifact};

fn diagnostic_path(error: &CompileError) -> Option<&str> {
    error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
}

fn format(version: u16) -> TargetPayloadFormat {
    TargetPayloadFormat::new("test.target-binary", version).expect("fixture format must be valid")
}

fn profile(version: u16) -> TargetProfile {
    TargetProfile::new(
        "test.target-binary",
        u64::from(version),
        [64, 1, 1],
        64,
        1_024,
        0,
    )
    .expect("fixture profile must be valid")
}
struct FixtureCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl TargetCompiler for FixtureCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        let entries = vec![entry_point()];
        TargetPayload::new(
            artifact,
            self.format.clone(),
            self.profile.clone(),
            entries,
            vec![4, 2],
        )
        .map_err(Into::into)
    }
}

/// WHY: absent geometry cannot be represented by a zero extent and later
/// defaulted to one. Every target entry must authenticate complete positive
/// workgroup and grid geometry before admission.
#[test]
fn target_payload_rejects_every_zero_geometry_class() {
    let neutral = neutral_artifact([8, 1, 1]);
    for (field, workgroup_size, grid_size) in [
        ("workgroup_size", [0, 1, 1], [1, 1, 1]),
        ("grid_size", [8, 1, 1], [0, 1, 1]),
    ] {
        let mut entry = entry_point();
        entry.workgroup_size = workgroup_size;
        entry.grid_size = grid_size;
        let error = TargetPayload::new(&neutral, format(1), profile(1), vec![entry], vec![1])
            .expect_err("zero target geometry must fail instead of defaulting to one");
        assert_eq!(
            diagnostic_path(&error),
            Some(format!("target_payload.entries[0].{field}[0]").as_str())
        );
        assert!(error.to_string().contains("geometry extent is zero"));
        assert!(error.to_string().contains("materialize explicit positive"));
    }
}

/// WHY: every product must associate pure target output through one authenticated envelope seam.
#[test]
fn target_compiler_attaches_exactly_one_authenticated_payload() {
    let neutral = neutral_artifact([8, 1, 1]);
    let neutral_digest = neutral.digest();
    let compiler = FixtureCompiler {
        format: format(9),
        profile: profile(9),
    };

    let envelope = attach_target(neutral, &compiler).expect("target attachment must succeed");

    assert_eq!(envelope.neutral().digest(), neutral_digest);
    let payload = envelope
        .require_target_payload(compiler.format())
        .expect("compiler format must be attached");
    assert_eq!(payload.neutral_artifact(), neutral_digest);
    assert_eq!(payload.bytes(), &[4, 2]);
}

/// Regression: packaging target bytes must retain the exact neutral artifact, entry IDs, and bytes.
#[test]
fn neutral_envelope_and_target_payload_round_trip_exactly() {
    let neutral = neutral_artifact([8, 1, 1]);
    let neutral_digest = neutral.digest();
    let payload = TargetPayload::new(
        &neutral,
        format(7),
        profile(7),
        vec![entry_point()],
        vec![0, 3, 7, 255],
    )
    .expect("valid target payload must bind");
    let payload_digest = payload.digest();
    let mut envelope = ArtifactEnvelope::new(neutral);
    envelope
        .attach_target_payload(payload)
        .expect("matching payload must attach");

    let decoded = ArtifactEnvelope::from_bytes(&envelope.to_bytes().expect("envelope must encode"))
        .expect("canonical envelope must decode");
    let decoded_payload = decoded
        .require_target_payload(&format(7))
        .expect("exact format must be compatible");

    assert_eq!(decoded.neutral().digest(), neutral_digest);
    assert_eq!(decoded_payload.neutral_artifact(), neutral_digest);
    assert_eq!(decoded_payload.digest(), payload_digest);
    assert_eq!(decoded_payload.bytes(), &[0, 3, 7, 255]);
    assert_eq!(decoded_payload.entries()[0].node, ArtifactNodeId(0));
    assert_eq!(
        decoded_payload.entries()[0].resource_bindings[0].resource,
        ArtifactValueId(0)
    );
    assert_eq!(decoded.neutral().geometry()[0].workgroup_size, [8, 1, 1]);
}

/// WHY: a target payload implements one selected neutral schedule. Changing
/// that schedule must change target identity even when emitted bytes match.
#[test]
fn target_identity_includes_the_neutral_schedule() {
    let first = neutral_artifact([8, 1, 1]);
    let second = neutral_artifact([16, 1, 1]);
    let first_payload = TargetPayload::new(
        &first,
        format(1),
        profile(1),
        vec![entry_point()],
        vec![9, 8, 7],
    )
    .expect("first target payload must validate");
    let mut second_entry = entry_point();
    second_entry.workgroup_size = [16, 1, 1];
    let second_payload = TargetPayload::new(
        &second,
        format(1),
        profile(1),
        vec![second_entry],
        vec![9, 8, 7],
    )
    .expect("second target payload must validate");

    assert_ne!(first_payload.digest(), second_payload.digest());
}

/// Regression: target bytes materialized from one neutral artifact must not attach to another digest.
#[test]
fn target_payload_rejects_a_different_neutral_artifact_digest() {
    let first = neutral_artifact([8, 1, 1]);
    let second = neutral_artifact([16, 1, 1]);
    assert_ne!(first.digest(), second.digest());
    let payload = TargetPayload::new(
        &first,
        format(1),
        profile(1),
        vec![entry_point()],
        vec![9, 8, 7],
    )
    .expect("payload must bind to its source artifact");
    let mut wrong_envelope = ArtifactEnvelope::new(second);

    let error = wrong_envelope
        .attach_target_payload(payload)
        .expect_err("wrong neutral artifact must fail deterministically");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC020_TARGET_PAYLOAD_ASSOCIATION_MISMATCH"
    );
    assert_eq!(
        diagnostic_path(&error),
        Some("target_payload.neutral_artifact")
    );
}

/// Regression: consumers must reject both unsupported payload schemas and format versions.
#[test]
fn target_payload_schema_and_format_version_skew_are_rejected() {
    let neutral = neutral_artifact([8, 1, 1]);
    let payload = TargetPayload::new(
        &neutral,
        format(2),
        profile(2),
        vec![entry_point()],
        vec![1, 2, 3],
    )
    .expect("payload must construct");
    let mut payload_bytes = payload.to_bytes().expect("payload must encode");
    payload_bytes[4..6].copy_from_slice(&4u16.to_le_bytes());
    let schema_error = TargetPayload::from_bytes(&payload_bytes)
        .expect_err("unsupported attachment schema must fail before body admission");
    assert_eq!(
        schema_error.diagnostic.code.as_str(),
        "MKC018_TARGET_PAYLOAD_VERSION_SKEW"
    );
    assert_eq!(
        diagnostic_path(&schema_error),
        Some("target_payload.schema_version")
    );

    let mut envelope = ArtifactEnvelope::new(neutral);
    envelope
        .attach_target_payload(payload)
        .expect("matching neutral association must attach");
    let format_error = envelope
        .require_target_payload(&format(3))
        .expect_err("wrong target format version must fail");
    assert_eq!(
        format_error.diagnostic.code.as_str(),
        "MKC018_TARGET_PAYLOAD_VERSION_SKEW"
    );
    assert_eq!(
        diagnostic_path(&format_error),
        Some("envelope.target_payloads.format.version")
    );
}

/// Regression: mutation of target bytes must fail the target identity before attachment.
#[test]
fn corrupted_target_payload_identity_is_rejected() {
    let neutral = neutral_artifact([8, 1, 1]);
    let payload = TargetPayload::new(
        &neutral,
        format(1),
        profile(1),
        vec![entry_point()],
        vec![11, 22, 33],
    )
    .expect("payload must construct");
    let mut bytes = payload.to_bytes().expect("payload must encode");
    bytes[10] ^= 1;

    let error = TargetPayload::from_bytes(&bytes)
        .expect_err("payload mutation must fail its domain-separated digest");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC019_TARGET_PAYLOAD_DIGEST_MISMATCH"
    );
    assert_eq!(diagnostic_path(&error), Some("target_payload.digest"));
}

/// WHY: authenticated payload bytes must not admit duplicate or out-of-order selected modules.
#[test]
fn target_module_bundle_rejects_noncanonical_module_order() {
    let program = Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());
    let descriptor = vyre_lower::lower_physical(&program)
        .expect("fixture lowering must succeed")
        .into_descriptor();
    let program = program.to_wire().expect("fixture Program must encode");
    let image = |stage, group| TargetModuleImage {
        group: FusionGroupId(group),
        stage,
        nodes: vec![ArtifactNodeId(group)],
        program: program.clone(),
        descriptor: descriptor.clone(),
        entry_point: format!("group_{group}"),
        bytes: vec![group as u8],
    };
    for modules in [
        vec![image(1, 0), image(0, 1)],
        vec![image(0, 1), image(0, 0)],
        vec![image(0, 0), image(0, 0)],
    ] {
        let bytes = TargetModuleBundle {
            schema_version: vyre_megakernel::TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
            modules,
        }
        .to_bytes()
        .expect("fixture bundle must encode");
        let error = TargetModuleBundle::from_bytes(&bytes)
            .expect_err("noncanonical stage/group order must fail admission");
        assert!(
            error.to_string().contains("canonical stage/group order"),
            "unexpected admission error: {error}"
        );
    }
}
/// WHY: fused multi-node modules where individual node programs declare buffers at colliding local binding slots (e.g. both slot 0) must resolve descriptor bindings by exact buffer name ownership, never cross-node slot index collisions.
#[test]
fn fused_multi_node_binding_resolution_uses_exact_name_ownership_over_colliding_slots() {
    let mut graph = vyre_foundation::ir::ProgramGraph::new();
    let val_a = graph
        .add_external_value(
            "a",
            vyre_foundation::ir::ValueContract {
                dtype: vyre_foundation::ir::DataType::U32,
                shape: vec![vyre_foundation::ir::ShapeDim::Symbol("items".into())],
                access: vyre_foundation::ir::BufferAccess::ReadOnly,
                lifetime: vyre_foundation::ir::ValueLifetime::Invocation,
            },
        )
        .unwrap();
    let val_b = graph
        .add_external_value(
            "b",
            vyre_foundation::ir::ValueContract {
                dtype: vyre_foundation::ir::DataType::U32,
                shape: vec![vyre_foundation::ir::ShapeDim::Symbol("items".into())],
                access: vyre_foundation::ir::BufferAccess::ReadOnly,
                lifetime: vyre_foundation::ir::ValueLifetime::Invocation,
            },
        )
        .unwrap();

    let prog0 = Program::wrapped(
        vec![
            vyre_foundation::ir::BufferDecl::storage(
                "node0_in",
                0,
                vyre_foundation::ir::BufferAccess::ReadOnly,
                vyre_foundation::ir::DataType::U32,
            ),
            vyre_foundation::ir::BufferDecl::storage(
                "mid_out",
                1,
                vyre_foundation::ir::BufferAccess::WriteOnly,
                vyre_foundation::ir::DataType::U32,
            ),
        ],
        [32, 1, 1],
        vec![vyre_foundation::ir::Node::store(
            "mid_out",
            vyre_foundation::ir::Expr::u32(0),
            vyre_foundation::ir::Expr::load("node0_in", vyre_foundation::ir::Expr::u32(0)),
        )],
    );

    let prog1 = Program::wrapped(
        vec![
            vyre_foundation::ir::BufferDecl::storage(
                "node1_in",
                0,
                vyre_foundation::ir::BufferAccess::ReadOnly,
                vyre_foundation::ir::DataType::U32,
            ),
            vyre_foundation::ir::BufferDecl::storage(
                "mid_in",
                1,
                vyre_foundation::ir::BufferAccess::ReadOnly,
                vyre_foundation::ir::DataType::U32,
            ),
            vyre_foundation::ir::BufferDecl::storage(
                "node1_out",
                2,
                vyre_foundation::ir::BufferAccess::WriteOnly,
                vyre_foundation::ir::DataType::U32,
            ),
        ],
        [32, 1, 1],
        vec![vyre_foundation::ir::Node::store(
            "node1_out",
            vyre_foundation::ir::Expr::u32(0),
            vyre_foundation::ir::Expr::add(
                vyre_foundation::ir::Expr::load("node1_in", vyre_foundation::ir::Expr::u32(0)),
                vyre_foundation::ir::Expr::load("mid_in", vyre_foundation::ir::Expr::u32(0)),
            ),
        )],
    );

    let (_node0, outputs0) = graph
        .add_node(
            "node0",
            prog0,
            vec![vyre_foundation::ir::GraphInput {
                buffer: "node0_in".into(),
                value: val_a,
                contract: vyre_foundation::ir::ValueContract {
                    dtype: vyre_foundation::ir::DataType::U32,
                    shape: vec![vyre_foundation::ir::ShapeDim::Symbol("items".into())],
                    access: vyre_foundation::ir::BufferAccess::ReadOnly,
                    lifetime: vyre_foundation::ir::ValueLifetime::Invocation,
                },
            }],
            vec![vyre_foundation::ir::GraphOutput {
                buffer: "mid_out".into(),
                name: "mid".into(),
                contract: vyre_foundation::ir::ValueContract {
                    dtype: vyre_foundation::ir::DataType::U32,
                    shape: vec![vyre_foundation::ir::ShapeDim::Symbol("items".into())],
                    access: vyre_foundation::ir::BufferAccess::WriteOnly,
                    lifetime: vyre_foundation::ir::ValueLifetime::Invocation,
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let mid_val = outputs0[0];

    let (_node1, outputs1) = graph
        .add_node(
            "node1",
            prog1,
            vec![
                vyre_foundation::ir::GraphInput {
                    buffer: "node1_in".into(),
                    value: val_b,
                    contract: vyre_foundation::ir::ValueContract {
                        dtype: vyre_foundation::ir::DataType::U32,
                        shape: vec![vyre_foundation::ir::ShapeDim::Symbol("items".into())],
                        access: vyre_foundation::ir::BufferAccess::ReadOnly,
                        lifetime: vyre_foundation::ir::ValueLifetime::Invocation,
                    },
                },
                vyre_foundation::ir::GraphInput {
                    buffer: "mid_in".into(),
                    value: mid_val,
                    contract: vyre_foundation::ir::ValueContract {
                        dtype: vyre_foundation::ir::DataType::U32,
                        shape: vec![vyre_foundation::ir::ShapeDim::Symbol("items".into())],
                        access: vyre_foundation::ir::BufferAccess::ReadOnly,
                        lifetime: vyre_foundation::ir::ValueLifetime::Invocation,
                    },
                },
            ],
            vec![vyre_foundation::ir::GraphOutput {
                buffer: "node1_out".into(),
                name: "out1".into(),
                contract: vyre_foundation::ir::ValueContract {
                    dtype: vyre_foundation::ir::DataType::U32,
                    shape: vec![vyre_foundation::ir::ShapeDim::Symbol("items".into())],
                    access: vyre_foundation::ir::BufferAccess::WriteOnly,
                    lifetime: vyre_foundation::ir::ValueLifetime::Output,
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let out_val = outputs1[0];

    let mut symbols = std::collections::BTreeMap::new();
    symbols.insert("items".into(), 32);
    let req = vyre_megakernel::CompileRequest::new(
        graph,
        vyre_megakernel::ExternalFacts::new(vyre_megakernel::Digest([0; 32]), symbols),
        vyre_megakernel::DeviceFacts::unknown(),
        vyre_megakernel::SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();

    let artifact = vyre_megakernel::compile(&req).expect("compilation must succeed");

    let mut module_count = 0;
    let mut captured_bindings = Vec::new();
    let _payload = compile_selected_modules(&artifact, format(1), profile(1), |selected, _prof| {
        module_count += 1;
        assert_eq!(
            selected.nodes.len(),
            2,
            "producer and consumer must fuse into a 2-node module"
        );
        captured_bindings = selected.canonical_bindings.clone();
        Ok(EmittedTargetModule {
            entry_point: "fused_entry".into(),
            workgroup_size: [32, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: selected.canonical_bindings.clone(),
            bytes: vec![1, 2, 3],
        })
    })
    .expect("selected module compilation must succeed");

    assert_eq!(module_count, 1, "exactly one fused module must be compiled");

    let node0_in_binding = captured_bindings
        .iter()
        .find(|b| b.resource == ArtifactValueId(val_a.0));
    let node1_in_binding = captured_bindings
        .iter()
        .find(|b| b.resource == ArtifactValueId(val_b.0));
    let out_binding = captured_bindings
        .iter()
        .find(|b| b.resource == ArtifactValueId(out_val.0));

    assert!(
        node0_in_binding.is_some(),
        "node0_in must be bound to val_a"
    );
    assert!(
        node1_in_binding.is_some(),
        "node1_in must be bound to val_b"
    );
    assert!(out_binding.is_some(), "node1_out must be bound to out_val");
    assert_ne!(
        node0_in_binding.unwrap().slot,
        node1_in_binding.unwrap().slot,
        "descriptor slots in fused kernel must be distinct"
    );
}
/// WHY: entry metadata must reject duplicate (group, slot) bindings.
#[test]
fn target_payload_rejects_duplicate_slot_within_entry() {
    let neutral = neutral_artifact([8, 1, 1]);
    let bindings = vec![
        TargetResourceBinding {
            resource: ArtifactValueId(0),
            group: 0,
            slot: 0,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadOnly,
        },
        TargetResourceBinding {
            resource: ArtifactValueId(0),
            group: 0,
            slot: 0,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::WriteOnly,
        },
    ];
    let error = TargetPayload::new(
        &neutral,
        format(1),
        profile(1),
        vec![TargetEntryPoint {
            name: "dup_slot".into(),
            node: ArtifactNodeId(0),
            workgroup_size: [8, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: bindings,
        }],
        vec![1, 2, 3],
    )
    .expect_err("duplicate (group, slot) must fail admission");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC017_MALFORMED_TARGET_PAYLOAD"
    );
}

/// WHY: fused modules where a canonical resource is bound at distinct slots (e.g. producer output and consumer input) must be accepted.
#[test]
fn target_payload_accepts_same_resource_at_distinct_slots() {
    let neutral = neutral_artifact([8, 1, 1]);
    let bindings = vec![
        TargetResourceBinding {
            resource: ArtifactValueId(0),
            group: 0,
            slot: 0,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadOnly,
        },
        TargetResourceBinding {
            resource: ArtifactValueId(0),
            group: 0,
            slot: 1,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::WriteOnly,
        },
    ];
    let payload = TargetPayload::new(
        &neutral,
        format(1),
        profile(1),
        vec![TargetEntryPoint {
            name: "distinct_slots_same_res".into(),
            node: ArtifactNodeId(0),
            workgroup_size: [8, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: bindings,
        }],
        vec![1, 2, 3],
    )
    .expect("same resource at distinct slots must be admitted");
    assert_eq!(payload.entries()[0].resource_bindings.len(), 2);
}

/// WHY: target payload must fail closed when referencing an unknown canonical resource.
#[test]
fn target_payload_rejects_unknown_canonical_resource() {
    let neutral = neutral_artifact([8, 1, 1]);
    let bindings = vec![TargetResourceBinding {
        resource: ArtifactValueId(999),
        group: 0,
        slot: 0,
        memory: TargetResourceMemory::Global,
        access: TargetResourceAccess::ReadOnly,
    }];
    let error = TargetPayload::new(
        &neutral,
        format(1),
        profile(1),
        vec![TargetEntryPoint {
            name: "unknown_res".into(),
            node: ArtifactNodeId(0),
            workgroup_size: [8, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: bindings,
        }],
        vec![1, 2, 3],
    )
    .expect_err("unknown resource must fail admission");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC020_TARGET_PAYLOAD_ASSOCIATION_MISMATCH"
    );
}

/// Build a one-node graph whose program writes `destination` under an optional
/// constant guard on axis-0 logical index, and return its selected launch span.
///
/// `source` is `chunk` elements and `destination` is `cache` elements, so the
/// buffer-derived span and the guarded span differ by construction.
fn selected_span(chunk: u32, cache: u32, guard: Option<u32>) -> u32 {
    use vyre_foundation::ir::{
        BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, ProgramGraph,
        ShapeDim, ValueContract, ValueLifetime,
    };

    let contract = |access, lifetime, symbol: &str| ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol(symbol.into())],
        access,
        lifetime,
    };

    let mut graph = ProgramGraph::new();
    let source_value = graph
        .add_external_value(
            "chunk",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, "chunk"),
        )
        .expect("external chunk value must be accepted");

    let index = Expr::var("index");
    let store = Node::store(
        "cache_out",
        Expr::add(index.clone(), Expr::u32(1)),
        Expr::load("chunk_in", index.clone()),
    );
    let body = match guard {
        Some(limit) => vec![
            Node::let_bind("index", Expr::LogicalIndex { axis: 0 }),
            Node::if_then(Expr::lt(index, Expr::u32(limit)), vec![store]),
        ],
        None => vec![
            Node::let_bind("index", Expr::LogicalIndex { axis: 0 }),
            store,
        ],
    };
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("chunk_in", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("cache_out", 1, BufferAccess::WriteOnly, DataType::U32),
        ],
        [32, 1, 1],
        body,
    );

    graph
        .add_node(
            "scatter",
            program,
            vec![GraphInput {
                buffer: "chunk_in".into(),
                value: source_value,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, "chunk"),
            }],
            vec![GraphOutput {
                buffer: "cache_out".into(),
                name: "cache".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, "cache"),
                retained_successor_of: None,
            }],
        )
        .expect("scatter node must be accepted");

    let mut symbols = std::collections::BTreeMap::new();
    symbols.insert("chunk".into(), u64::from(chunk));
    symbols.insert("cache".into(), u64::from(cache));
    let request = vyre_megakernel::CompileRequest::new(
        graph,
        vyre_megakernel::ExternalFacts::new(vyre_megakernel::Digest([0; 32]), symbols),
        vyre_megakernel::DeviceFacts::unknown(),
        vyre_megakernel::SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .expect("scatter request must validate");
    let artifact = vyre_megakernel::compile(&request).expect("scatter must compile");

    let mut span = None;
    compile_selected_modules(&artifact, format(1), profile(1), |selected, _prof| {
        span = Some(selected.logical_element_count);
        Ok(EmittedTargetModule {
            entry_point: "scatter_entry".into(),
            workgroup_size: [32, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: selected.canonical_bindings.clone(),
            bytes: vec![1],
        })
    })
    .expect("selected module compilation must succeed");
    span.expect("one module must be compiled")
}

/// WHY: a launch span taken from the widest declared resource is right for a
/// gather and wrong for a scatter, which guards on a small source and declares
/// a much larger destination. The guard is in the IR, so the compiler reads it
/// instead of firing one lane per destination element and letting the guard
/// discard the rest. Without the guard the destination is the only bound left,
/// which is the case the cap must not touch.
#[test]
fn a_constant_guard_caps_the_launch_span_at_the_domain_it_admits() {
    assert_eq!(
        selected_span(8, 1024, Some(8)),
        8,
        "a guarded scatter must launch over its guarded domain, not its destination"
    );
    assert_eq!(
        selected_span(8, 1024, None),
        1024,
        "an unguarded write has no domain but its destination"
    );
}

/// WHY: the cap is a minimum, not a replacement. A guard wider than anything
/// the program can address must not raise the launch above the resources, and a
/// guard of zero must not produce a launch of zero groups, which a driver
/// rejects outright.
#[test]
fn a_guard_never_raises_the_span_above_the_resources_or_lowers_it_below_one() {
    assert_eq!(
        selected_span(8, 64, Some(4_000_000)),
        64,
        "a guard wider than the resources must not widen the launch"
    );
    assert_eq!(
        selected_span(8, 64, Some(0)),
        1,
        "a zero-domain guard must still leave a launchable span"
    );
}
