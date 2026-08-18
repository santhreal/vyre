use std::collections::BTreeMap;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    compile, Artifact, ArtifactNodeId, ArtifactValueId, CompileRequest, DeviceFacts, Digest,
    ExternalFacts, SearchBudget, TargetEntryPoint, TargetPayload, TargetPayloadFormat,
    TargetProfile, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};

use crate::materialize::{unbound_input, InstanceCore, NEUTRAL_MESSAGES};
use crate::{BackendError, BindingPlan, DeviceIdentity};

fn test_format() -> TargetPayloadFormat {
    TargetPayloadFormat::new("test.target-binary", 1).unwrap()
}

fn test_profile() -> TargetProfile {
    TargetProfile::new("test.target-binary", 1, [32, 1, 1], 32, 1024, 0).unwrap()
}

fn test_device() -> DeviceIdentity {
    DeviceIdentity {
        backend: "test",
        device: "test-device".into(),
        generation: 1,
    }
}

fn test_instance_core(
    artifact: &Artifact,
    payload: &TargetPayload,
) -> Result<InstanceCore, BackendError> {
    InstanceCore::new_with_module_slots(
        artifact,
        payload,
        test_device(),
        NEUTRAL_MESSAGES,
        vec![BTreeMap::new(); artifact.fusion().len()],
    )
}

fn contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(32)],
        access,
        lifetime,
    }
}

#[test]
fn read_write_retained_preserves_input_output_and_order() {
    let mut graph = ProgramGraph::new();
    let state_in = graph
        .add_external_value(
            "state_in",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
        )
        .unwrap();

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("s_in", 0, BufferAccess::ReadWrite, DataType::U32).with_count(32),
            BufferDecl::storage("s_out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "s_out",
            Expr::u32(0),
            Expr::add(Expr::load("s_in", Expr::u32(0)), Expr::u32(1)),
        )],
    );

    let (_, outputs) = graph
        .add_node(
            "node0",
            prog.clone(),
            vec![GraphInput {
                buffer: "s_in".into(),
                value: state_in,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![GraphOutput {
                buffer: "s_out".into(),
                name: "state_out".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                retained_successor_of: Some(state_in),
            }],
        )
        .unwrap();
    let state_out = outputs[0];

    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();

    let artifact = compile(&req).expect("compilation must succeed");

    let payload = TargetPayload::new(
        &artifact,
        test_format(),
        test_profile(),
        vec![TargetEntryPoint {
            name: "entry0".into(),
            node: ArtifactNodeId(0),
            workgroup_size: [32, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: vec![
                TargetResourceBinding {
                    resource: ArtifactValueId(state_in.0),
                    group: 0,
                    slot: 0,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadWrite,
                },
                TargetResourceBinding {
                    resource: ArtifactValueId(state_out.0),
                    group: 0,
                    slot: 1,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadWrite,
                },
            ],
        }],
        vec![1, 2, 3],
    )
    .unwrap();

    let core = test_instance_core(&artifact, &payload).unwrap();
    assert_eq!(
        core.module_inputs[0],
        vec![ArtifactValueId(state_in.0), ArtifactValueId(state_in.0)]
    );
    assert_eq!(
        core.module_outputs[0],
        vec![ArtifactValueId(state_in.0), ArtifactValueId(state_out.0)]
    );
}

#[test]
fn write_only_output_and_read_only_input_separation() {
    let mut graph = ProgramGraph::new();
    let val_const = graph
        .add_external_value(
            "const_in",
            contract(BufferAccess::ReadOnly, ValueLifetime::Constant),
        )
        .unwrap();

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("c_in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(32),
            BufferDecl::storage("w_out", 1, BufferAccess::WriteOnly, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "w_out",
            Expr::u32(0),
            Expr::load("c_in", Expr::u32(0)),
        )],
    );

    let (_, outputs) = graph
        .add_node(
            "node0",
            prog,
            vec![GraphInput {
                buffer: "c_in".into(),
                value: val_const,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Constant),
            }],
            vec![GraphOutput {
                buffer: "w_out".into(),
                name: "res_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let res_out = outputs[0];

    let mut facts = ExternalFacts::new(Digest([0; 32]), BTreeMap::new());
    facts.constant_identities.insert(val_const, Digest([1; 32]));
    let req = CompileRequest::new(
        graph,
        facts,
        DeviceFacts::unknown(),
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();

    let artifact = compile(&req).expect("compilation must succeed");

    let payload = TargetPayload::new(
        &artifact,
        test_format(),
        test_profile(),
        vec![TargetEntryPoint {
            name: "entry0".into(),
            node: ArtifactNodeId(0),
            workgroup_size: [32, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: vec![
                TargetResourceBinding {
                    resource: ArtifactValueId(val_const.0),
                    group: 0,
                    slot: 0,
                    memory: TargetResourceMemory::Constant,
                    access: TargetResourceAccess::ReadOnly,
                },
                TargetResourceBinding {
                    resource: ArtifactValueId(res_out.0),
                    group: 0,
                    slot: 1,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::WriteOnly,
                },
            ],
        }],
        vec![1, 2, 3],
    )
    .unwrap();

    let core = test_instance_core(&artifact, &payload).unwrap();
    assert_eq!(core.module_inputs[0], vec![ArtifactValueId(val_const.0)]);
    assert_eq!(core.module_outputs[0], vec![ArtifactValueId(res_out.0)]);
}

#[test]
fn read_write_invocation_lifetime_is_input_and_output() {
    let mut graph = ProgramGraph::new();
    let val_inv = graph
        .add_external_value(
            "inv_rw",
            contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
        )
        .unwrap();

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_rw", 0, BufferAccess::ReadWrite, DataType::U32).with_count(32),
            BufferDecl::storage("out_rw", 1, BufferAccess::ReadWrite, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "out_rw",
            Expr::u32(0),
            Expr::add(Expr::load("in_rw", Expr::u32(0)), Expr::u32(1)),
        )],
    );

    let (_, outputs) = graph
        .add_node(
            "node0",
            prog,
            vec![GraphInput {
                buffer: "in_rw".into(),
                value: val_inv,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "out_rw".into(),
                name: "inv_out".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let out_id = outputs[0];

    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();

    let artifact = compile(&req).expect("compilation must succeed");

    let payload = TargetPayload::new(
        &artifact,
        test_format(),
        test_profile(),
        vec![TargetEntryPoint {
            name: "entry0".into(),
            node: ArtifactNodeId(0),
            workgroup_size: [32, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: vec![TargetResourceBinding {
                resource: ArtifactValueId(val_inv.0),
                group: 0,
                slot: 0,
                memory: TargetResourceMemory::Global,
                access: TargetResourceAccess::ReadWrite,
            }],
        }],
        vec![1, 2, 3],
    )
    .unwrap();

    let core = test_instance_core(&artifact, &payload).unwrap();
    assert_eq!(core.module_inputs[0], vec![ArtifactValueId(val_inv.0)]);
    assert_eq!(core.module_outputs[0], vec![ArtifactValueId(val_inv.0)]);
}

#[test]
fn read_write_constant_lifetime_is_input_and_output() {
    let mut graph = ProgramGraph::new();
    let val_const = graph
        .add_external_value(
            "const_rw",
            contract(BufferAccess::ReadWrite, ValueLifetime::Constant),
        )
        .unwrap();

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_rw", 0, BufferAccess::ReadWrite, DataType::U32).with_count(32),
            BufferDecl::storage("out_rw", 1, BufferAccess::ReadWrite, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "out_rw",
            Expr::u32(0),
            Expr::add(Expr::load("in_rw", Expr::u32(0)), Expr::u32(1)),
        )],
    );

    let (_, outputs) = graph
        .add_node(
            "node0",
            prog,
            vec![GraphInput {
                buffer: "in_rw".into(),
                value: val_const,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Constant),
            }],
            vec![GraphOutput {
                buffer: "out_rw".into(),
                name: "const_out".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let _out_id = outputs[0];

    let mut facts = ExternalFacts::new(Digest([0; 32]), BTreeMap::new());
    facts.constant_identities.insert(val_const, Digest([1; 32]));
    let req = CompileRequest::new(
        graph,
        facts,
        DeviceFacts::unknown(),
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();

    let artifact = compile(&req).expect("compilation must succeed");

    let payload = TargetPayload::new(
        &artifact,
        test_format(),
        test_profile(),
        vec![TargetEntryPoint {
            name: "entry0".into(),
            node: ArtifactNodeId(0),
            workgroup_size: [32, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: vec![TargetResourceBinding {
                resource: ArtifactValueId(val_const.0),
                group: 0,
                slot: 0,
                memory: TargetResourceMemory::Global,
                access: TargetResourceAccess::ReadWrite,
            }],
        }],
        vec![1, 2, 3],
    )
    .unwrap();

    let core = test_instance_core(&artifact, &payload).unwrap();
    assert_eq!(core.module_inputs[0], vec![ArtifactValueId(val_const.0)]);
    assert_eq!(core.module_outputs[0], vec![ArtifactValueId(val_const.0)]);
}

#[test]
fn module_aliases_with_mixed_access_and_lifetimes_contract() {
    let mut graph = ProgramGraph::new();
    let val_in = graph
        .add_external_value(
            "in_inv",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .unwrap();
    let state_init = graph
        .add_external_value(
            "state_init",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
        )
        .unwrap();

    let prog0 = Program::wrapped(
        vec![
            BufferDecl::storage("i0", 0, BufferAccess::ReadOnly, DataType::U32).with_count(32),
            BufferDecl::output("o0", 1, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "o0",
            Expr::u32(0),
            Expr::load("i0", Expr::u32(0)),
        )],
    );

    let prog1 = Program::wrapped(
        vec![
            BufferDecl::storage("s_in", 0, BufferAccess::ReadWrite, DataType::U32).with_count(32),
            BufferDecl::storage("s_out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(32),
            BufferDecl::storage("o1", 2, BufferAccess::WriteOnly, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![
            Node::store(
                "s_out",
                Expr::u32(0),
                Expr::add(Expr::load("s_in", Expr::u32(0)), Expr::u32(1)),
            ),
            Node::store("o1", Expr::u32(0), Expr::load("s_in", Expr::u32(0))),
        ],
    );

    let (node0, out0_vec) = graph
        .add_node(
            "node0",
            prog0,
            vec![GraphInput {
                buffer: "i0".into(),
                value: val_in,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "o0".into(),
                name: "out0".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let out0 = out0_vec[0];

    let (node1, out1_vec) = graph
        .add_node(
            "node1",
            prog1,
            vec![GraphInput {
                buffer: "s_in".into(),
                value: state_init,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![
                GraphOutput {
                    buffer: "s_out".into(),
                    name: "state_next".into(),
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                    retained_successor_of: Some(state_init),
                },
                GraphOutput {
                    buffer: "o1".into(),
                    name: "out1".into(),
                    contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                    retained_successor_of: None,
                },
            ],
        )
        .unwrap();
    let state_next = out1_vec[0];
    let out1 = out1_vec[1];

    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();

    let artifact = compile(&req).expect("compilation must succeed");
    let payload = TargetPayload::new(
        &artifact,
        test_format(),
        test_profile(),
        vec![
            TargetEntryPoint {
                name: "entry0".into(),
                node: ArtifactNodeId(node0.0),
                workgroup_size: [32, 1, 1],
                grid_size: [1, 1, 1],
                dynamic_shared_bytes: 0,
                resource_bindings: vec![
                    TargetResourceBinding {
                        resource: ArtifactValueId(val_in.0),
                        group: 0,
                        slot: 0,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::ReadOnly,
                    },
                    TargetResourceBinding {
                        resource: ArtifactValueId(out0.0),
                        group: 0,
                        slot: 1,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::WriteOnly,
                    },
                ],
            },
            TargetEntryPoint {
                name: "entry1".into(),
                node: ArtifactNodeId(node1.0),
                workgroup_size: [32, 1, 1],
                grid_size: [1, 1, 1],
                dynamic_shared_bytes: 0,
                resource_bindings: vec![
                    TargetResourceBinding {
                        resource: ArtifactValueId(state_init.0),
                        group: 0,
                        slot: 0,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::ReadWrite,
                    },
                    TargetResourceBinding {
                        resource: ArtifactValueId(state_next.0),
                        group: 0,
                        slot: 1,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::WriteOnly,
                    },
                    TargetResourceBinding {
                        resource: ArtifactValueId(out1.0),
                        group: 0,
                        slot: 2,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::WriteOnly,
                    },
                ],
            },
        ],
        vec![1, 2, 3],
    )
    .unwrap();

    let core = test_instance_core(&artifact, &payload).unwrap();
    let module0 = core
        .module_named_resources
        .iter()
        .position(|resources| resources.contains_key("i0"))
        .unwrap();
    let module1 = core
        .module_named_resources
        .iter()
        .position(|resources| resources.contains_key("s_in"))
        .unwrap();
    assert_eq!(core.module_inputs[module0], vec![ArtifactValueId(val_in.0)]);
    assert_eq!(core.module_outputs[module0], vec![ArtifactValueId(out0.0)]);
    assert_eq!(
        core.module_inputs[module1],
        vec![ArtifactValueId(state_init.0)]
    );
    assert_eq!(
        core.module_outputs[module1],
        vec![
            ArtifactValueId(state_init.0),
            ArtifactValueId(state_next.0),
            ArtifactValueId(out1.0),
        ]
    );
}

/// WHY: closes the class "target descriptor order is mistaken for Program
/// host-input order". Target lowering may place read-write storage before
/// read-only storage even when the Program declares the read-only buffer
/// first. Positional lookup then uploads the output-sized bytes into the
/// input binding. Identity lookup must preserve each declaration's bytes.
///
/// This does not cover an artifact whose canonical resource names no longer
/// match its Program buffer names; artifact validation owns that refusal.
#[test]
fn target_descriptor_order_does_not_reorder_program_inputs() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("nodes", 0, BufferAccess::ReadOnly, DataType::U32).with_count(60),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(8),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("nodes", Expr::u32(0)),
        )],
    );
    let graph = ProgramGraph::from_program("walk", program.clone())
        .expect("the two-buffer Program must lift to a graph");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .expect("the graph must validate");
    let artifact = compile(&request).expect("the graph must compile");
    let nodes = artifact
        .resources()
        .iter()
        .find(|resource| resource.name == "nodes")
        .expect("the artifact must carry `nodes`")
        .value;
    let out = artifact
        .resources()
        .iter()
        .find(|resource| resource.name == "out")
        .expect("the artifact must carry `out`")
        .value;
    let payload = TargetPayload::new(
        &artifact,
        test_format(),
        test_profile(),
        vec![TargetEntryPoint {
            name: "walk".into(),
            node: ArtifactNodeId(0),
            workgroup_size: [1, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: vec![
                TargetResourceBinding {
                    resource: out,
                    group: 0,
                    slot: 0,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadWrite,
                },
                TargetResourceBinding {
                    resource: nodes,
                    group: 0,
                    slot: 1,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadOnly,
                },
            ],
        }],
        vec![1, 2, 3],
    )
    .expect("descriptor order is independent from Program declaration order");
    let core = test_instance_core(&artifact, &payload).unwrap();
    assert_eq!(
        core.module_inputs[0],
        vec![out, nodes],
        "the fixture must preserve the target descriptor order that triggered the defect"
    );
    let plan = BindingPlan::build(&program).expect("the Program binding plan must be valid");
    let node_bytes = vec![0x11; 240];
    let out_bytes = vec![0x22; 32];
    let state = BTreeMap::from([(nodes, node_bytes.clone()), (out, out_bytes.clone())]);

    let gathered = core
        .gather_inputs_for_module(0, &plan, &program, &state, unbound_input)
        .expect("identity lookup must gather both declared inputs");

    assert_eq!(gathered, vec![node_bytes.as_slice(), out_bytes.as_slice()]);
}

/// WHY: Section 169.5 / Row 174.5 requires proving exact group/slot lookup, bidirectional
/// predecessor lineage resolution, and fail-closed missing identities across multi-stage execution.
#[test]
fn bidirectional_retained_lineage_and_exact_slot_resolution_fails_closed() {
    let mut graph = ProgramGraph::new();
    let state_val = graph
        .add_external_value(
            "state_root",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
        )
        .unwrap();

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("state_buf", 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::output("out_buf", 1, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "out_buf",
            Expr::u32(0),
            Expr::load("state_buf", Expr::u32(0)),
        )],
    );

    let (node_id, outputs) = graph
        .add_node(
            "stage_node",
            prog.clone(),
            vec![GraphInput {
                buffer: "state_buf".into(),
                value: state_val,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![GraphOutput {
                buffer: "out_buf".into(),
                name: "output_final".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: Some(state_val),
            }],
        )
        .unwrap();
    let out_final = outputs[0];

    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(32, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();
    let artifact = compile(&req).unwrap();

    let payload = TargetPayload::new(
        &artifact,
        test_format(),
        test_profile(),
        vec![TargetEntryPoint {
            name: "stage_entry".into(),
            node: ArtifactNodeId(node_id.0),
            workgroup_size: [32, 1, 1],
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: vec![
                TargetResourceBinding {
                    resource: ArtifactValueId(state_val.0),
                    group: 0,
                    slot: 0,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadWrite,
                },
                TargetResourceBinding {
                    resource: ArtifactValueId(out_final.0),
                    group: 0,
                    slot: 1,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::WriteOnly,
                },
            ],
        }],
        vec![1, 2, 3],
    )
    .unwrap();

    let core = test_instance_core(&artifact, &payload).unwrap();
    let plan = BindingPlan::build(&prog).unwrap();

    let in_val = core
        .value_for_module_binding(&core.module_inputs, 0, &plan.bindings[0])
        .expect("retained input must resolve");
    assert_eq!(in_val, ArtifactValueId(state_val.0));

    let out_val = core
        .value_for_module_binding(&core.module_outputs, 0, &plan.bindings[1])
        .expect("output binding must resolve");
    assert_eq!(out_val, ArtifactValueId(out_final.0));

    let mut missing_binding = plan.bindings[0].clone();
    missing_binding.name = "unmapped_buffer".into();
    missing_binding.binding = 99;
    assert!(
        core.value_for_module_binding(&core.module_inputs, 0, &missing_binding)
            .is_err(),
        "unmapped buffer must fail closed with unmapped_buffer"
    );
    assert!(
        core.value_for_module_slot(&core.module_inputs, 0, 0, 99, "unmapped_slot")
            .is_err(),
        "unmapped slot must fail closed with invalid_module"
    );
}
