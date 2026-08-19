use std::collections::BTreeMap;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ValueLifetime,
};
use vyre_megakernel::{ArtifactNodeId, ArtifactValueId, TargetResourceAccess};

use crate::materialize::materialize_test_fixtures::{
    compile_graph, contract, entry_point, global_bindings, test_instance_core, test_payload,
};
use crate::materialize::{retained_chain_relates, unbound_input, unbound_resident_buffer};
use crate::{BackendError, BindingPlan, Resource};

#[test]
fn sparse_and_reordered_module_binding_identities() {
    let mut graph = ProgramGraph::new();
    let val_x = graph
        .add_external_value(
            "x",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .unwrap();
    let val_y = graph
        .add_external_value(
            "y",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .unwrap();

    let program0 = Program::wrapped(
        vec![
            BufferDecl::storage("in_b", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("in_a", 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("out_0", 2, BufferAccess::WriteOnly, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "out_0",
            Expr::u32(0),
            Expr::add(
                Expr::load("in_b", Expr::u32(0)),
                Expr::load("in_a", Expr::u32(0)),
            ),
        )],
    );

    let (_, outputs) = graph
        .add_node(
            "node0",
            program0.clone(),
            vec![
                GraphInput {
                    buffer: "in_b".into(),
                    value: val_y,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "in_a".into(),
                    value: val_x,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                },
            ],
            vec![GraphOutput {
                buffer: "out_0".into(),
                name: "res".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let res_id = outputs[0];

    let artifact = compile_graph(graph);

    let bindings = global_bindings(&[
        (ArtifactValueId(val_y.0), TargetResourceAccess::ReadOnly),
        (ArtifactValueId(val_x.0), TargetResourceAccess::ReadOnly),
        (ArtifactValueId(res_id.0), TargetResourceAccess::WriteOnly),
    ]);

    let payload = test_payload(
        &artifact,
        vec![entry_point("entry0", ArtifactNodeId(0), bindings)],
    );

    let core = test_instance_core(&artifact, &payload).unwrap();
    let plan0 = BindingPlan::build(&program0).unwrap();

    let mut state = BTreeMap::new();
    state.insert(ArtifactValueId(val_y.0), vec![2, 0, 0, 0]);
    state.insert(ArtifactValueId(val_x.0), vec![1, 0, 0, 0]);

    let gathered = core
        .gather_inputs_for_module(0, &plan0, &program0, &state, unbound_input)
        .unwrap();

    assert_eq!(gathered[0], &[2, 0, 0, 0]);
    assert_eq!(gathered[1], &[1, 0, 0, 0]);

    core.absorb_outputs_for_module(
        0,
        &plan0,
        &program0,
        vec![vec![3, 0, 0, 0]],
        &mut state,
        |idx, name| BackendError::InvalidProgram {
            fix: format!("missing output {idx} {name}"),
        },
    )
    .unwrap();

    assert_eq!(
        state.get(&ArtifactValueId(res_id.0)).unwrap(),
        &[3, 0, 0, 0]
    );

    let completion = core.completion(&state, None).unwrap();
    assert_eq!(
        completion.outputs.get(&ArtifactValueId(res_id.0)).unwrap(),
        &[3, 0, 0, 0]
    );

    let owner = crate::ResidentOwner::new().expect("resident owner identity");
    let res_x = Resource::Resident(owner.handle(10));
    let res_y = Resource::Resident(owner.handle(20));
    let res_out = Resource::Resident(owner.handle(30));

    let mut resident_map = BTreeMap::new();
    resident_map.insert(ArtifactValueId(val_x.0), res_x.clone());
    resident_map.insert(ArtifactValueId(val_y.0), res_y.clone());
    resident_map.insert(ArtifactValueId(res_id.0), res_out.clone());

    let ordered = core
        .ordered_resident_resources_for_module(
            0,
            &plan0,
            &program0,
            &resident_map,
            unbound_resident_buffer,
        )
        .expect("ordered resident resources must resolve by authenticated identity");

    // Binding plan order: in_b (0), in_a (1), out_0 (2)
    assert_eq!(ordered[0], res_y);
    assert_eq!(ordered[1], res_x);
    assert_eq!(ordered[2], res_out);

    // Unbound resource fails closed
    let mut missing_map = resident_map.clone();
    missing_map.remove(&ArtifactValueId(val_x.0));
    let unbound_err = core
        .ordered_resident_resources_for_module(
            0,
            &plan0,
            &program0,
            &missing_map,
            unbound_resident_buffer,
        )
        .expect_err("missing resident resource must fail closed");
    assert!(unbound_err.to_string().contains("in_a"));
}

/// WHY: multi-segment dispatches with whole-grid fences or fused pipelines
/// link successive retained values via `retained_successor_of`. Materialization
/// must preserve transitive retained value lineage in both directions:
/// when gathering module inputs (resolving the successor value from the
/// root canonical value or previous segment output) and when absorbing outputs
/// (propagating the produced bytes back to all transitive predecessors).
///
/// Does not catch: hardware device timeouts during multi-segment dispatch,
/// which is covered by backend-specific resident grid sync suites.
#[test]
fn transitive_retained_predecessor_lineage_preservation() {
    let mut graph = ProgramGraph::new();
    let state_init = graph
        .add_external_value(
            "canonical_state",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
        )
        .unwrap();

    let seg0_prog = Program::wrapped(
        vec![BufferDecl::storage(
            "state",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [32, 1, 1],
        vec![Node::store(
            "state",
            Expr::u32(0),
            Expr::add(Expr::load("state", Expr::u32(0)), Expr::u32(1)),
        )],
    );

    let (node0, seg0_outputs) = graph
        .add_node(
            "seg0",
            seg0_prog.clone(),
            vec![GraphInput {
                buffer: "state".into(),
                value: state_init,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![GraphOutput {
                buffer: "state".into(),
                name: "state_mid".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                retained_successor_of: Some(state_init),
            }],
        )
        .unwrap();
    let state_mid = seg0_outputs[0];

    let seg1_prog = Program::wrapped(
        vec![
            BufferDecl::storage("state", 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage("res", 1, BufferAccess::WriteOnly, DataType::U32),
        ],
        [32, 1, 1],
        vec![
            Node::store(
                "state",
                Expr::u32(0),
                Expr::add(Expr::load("state", Expr::u32(0)), Expr::u32(2)),
            ),
            Node::store("res", Expr::u32(0), Expr::load("state", Expr::u32(0))),
        ],
    );

    let (node1, seg1_outputs) = graph
        .add_node(
            "seg1",
            seg1_prog.clone(),
            vec![GraphInput {
                buffer: "state".into(),
                value: state_mid,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![
                GraphOutput {
                    buffer: "state".into(),
                    name: "state_final".into(),
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                    retained_successor_of: Some(state_mid),
                },
                GraphOutput {
                    buffer: "res".into(),
                    name: "res".into(),
                    contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                    retained_successor_of: None,
                },
            ],
        )
        .unwrap();
    let state_final = seg1_outputs[0];
    let out_id = seg1_outputs[1];

    let artifact = compile_graph(graph);

    let payload = test_payload(
        &artifact,
        vec![
            entry_point(
                "seg0_entry",
                ArtifactNodeId(node0.0),
                global_bindings(&[(
                    ArtifactValueId(state_mid.0),
                    TargetResourceAccess::ReadWrite,
                )]),
            ),
            entry_point(
                "seg1_entry",
                ArtifactNodeId(node1.0),
                global_bindings(&[
                    (
                        ArtifactValueId(state_final.0),
                        TargetResourceAccess::ReadWrite,
                    ),
                    (ArtifactValueId(out_id.0), TargetResourceAccess::WriteOnly),
                ]),
            ),
        ],
    );

    let core = test_instance_core(&artifact, &payload).unwrap();
    assert_eq!(
        core.module_inputs,
        vec![
            vec![ArtifactValueId(state_init.0)],
            vec![ArtifactValueId(state_init.0)]
        ],
        "every retained successor module must read the bound root predecessor",
    );
    assert!(
        core.value_for_buffer("state").is_err(),
        "the fused Program-local name must not be present in the canonical artifact ABI",
    );
    let seg0_plan = BindingPlan::build(&seg0_prog).unwrap();
    assert!(
        core.value_for_module_binding(&core.module_inputs, 2, &seg0_plan.bindings[0])
            .is_err(),
        "a missing module identity must fail closed instead of falling back to module zero",
    );
    let seg1_plan = BindingPlan::build(&seg1_prog).unwrap();

    let mut state = BTreeMap::new();
    state.insert(ArtifactValueId(state_init.0), vec![0, 0, 0, 0]);

    let gathered0 = core
        .gather_inputs_for_module(0, &seg0_plan, &seg0_prog, &state, unbound_input)
        .unwrap();
    assert_eq!(gathered0[0], &[0, 0, 0, 0]);

    core.absorb_outputs_for_module(
        0,
        &seg0_plan,
        &seg0_prog,
        vec![vec![42, 0, 0, 0]],
        &mut state,
        |idx, name| BackendError::InvalidProgram {
            fix: format!("missing output {idx} {name}"),
        },
    )
    .unwrap();

    assert_eq!(
        state.get(&ArtifactValueId(state_mid.0)).unwrap(),
        &[42, 0, 0, 0]
    );
    assert_eq!(
        state.get(&ArtifactValueId(state_init.0)).unwrap(),
        &[42, 0, 0, 0]
    );

    let gathered1 = core
        .gather_inputs_for_module(1, &seg1_plan, &seg1_prog, &state, unbound_input)
        .unwrap();
    assert_eq!(gathered1[0], &[42, 0, 0, 0]);

    core.absorb_outputs_for_module(
        1,
        &seg1_plan,
        &seg1_prog,
        vec![vec![99, 0, 0, 0], vec![1, 2, 3, 4]],
        &mut state,
        |idx, name| BackendError::InvalidProgram {
            fix: format!("missing output {idx} {name}"),
        },
    )
    .unwrap();

    assert_eq!(
        state.get(&ArtifactValueId(state_final.0)).unwrap(),
        &[99, 0, 0, 0]
    );
    assert_eq!(
        state.get(&ArtifactValueId(state_mid.0)).unwrap(),
        &[99, 0, 0, 0]
    );
    assert_eq!(
        state.get(&ArtifactValueId(state_init.0)).unwrap(),
        &[99, 0, 0, 0]
    );
    assert_eq!(
        state.get(&ArtifactValueId(out_id.0)).unwrap(),
        &[1, 2, 3, 4]
    );

    let completion = core.completion(&state, Some(1000)).unwrap();
    assert_eq!(
        completion
            .retained
            .get(&ArtifactValueId(state_init.0))
            .unwrap(),
        &[99, 0, 0, 0]
    );
    assert_eq!(
        completion
            .retained
            .get(&ArtifactValueId(state_mid.0))
            .unwrap(),
        &[99, 0, 0, 0]
    );
    assert_eq!(
        completion
            .retained
            .get(&ArtifactValueId(state_final.0))
            .unwrap(),
        &[99, 0, 0, 0]
    );
    assert_eq!(
        completion.outputs.get(&ArtifactValueId(out_id.0)).unwrap(),
        &[1, 2, 3, 4]
    );
}

#[test]
fn later_module_inputs_and_outputs_resolve_by_named_entry_identity() {
    let mut graph = ProgramGraph::new();
    let val_a = graph
        .add_external_value(
            "a",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .unwrap();
    let val_b = graph
        .add_external_value(
            "b",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .unwrap();
    let program = |input: &str, output: &str| {
        Program::wrapped(
            vec![
                BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::storage(output, 1, BufferAccess::WriteOnly, DataType::U32),
            ],
            [32, 1, 1],
            vec![Node::store(
                output,
                Expr::u32(0),
                Expr::load(input, Expr::u32(0)),
            )],
        )
    };
    let prog0 = program("node0_in", "node0_out");
    let prog1 = program("node1_in", "node1_out");
    let (node0, outputs0) = graph
        .add_node(
            "node0",
            prog0.clone(),
            vec![GraphInput {
                buffer: "node0_in".into(),
                value: val_a,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "node0_out".into(),
                name: "out0".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let out0 = outputs0[0];
    let (node1, outputs1) = graph
        .add_node(
            "node1",
            prog1.clone(),
            vec![GraphInput {
                buffer: "node1_in".into(),
                value: val_b,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "node1_out".into(),
                name: "out1".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let out1 = outputs1[0];
    let artifact = compile_graph(graph);
    let entry = |node, input, output| {
        entry_point(
            "main",
            node,
            global_bindings(&[
                (input, TargetResourceAccess::ReadOnly),
                (output, TargetResourceAccess::WriteOnly),
            ]),
        )
    };
    let payload = test_payload(
        &artifact,
        vec![
            entry(
                ArtifactNodeId(node0.0),
                ArtifactValueId(val_a.0),
                ArtifactValueId(out0.0),
            ),
            entry(
                ArtifactNodeId(node1.0),
                ArtifactValueId(val_b.0),
                ArtifactValueId(out1.0),
            ),
        ],
    );
    let core = test_instance_core(&artifact, &payload).unwrap();
    let module0 = core
        .module_named_resources
        .iter()
        .position(|resources| resources.contains_key("node0_in"))
        .unwrap();
    let module1 = core
        .module_named_resources
        .iter()
        .position(|resources| resources.contains_key("node1_in"))
        .unwrap();
    let mut state = BTreeMap::from([
        (ArtifactValueId(val_a.0), vec![10, 0, 0, 0]),
        (ArtifactValueId(val_b.0), vec![30, 0, 0, 0]),
    ]);
    let gathered0 = core
        .gather_inputs_for_module(
            module0,
            &BindingPlan::build(&prog0).unwrap(),
            &prog0,
            &state,
            unbound_input,
        )
        .unwrap();
    let gathered1 = core
        .gather_inputs_for_module(
            module1,
            &BindingPlan::build(&prog1).unwrap(),
            &prog1,
            &state,
            unbound_input,
        )
        .unwrap();
    assert_eq!(gathered0, vec![&[10, 0, 0, 0][..]]);
    assert_eq!(gathered1, vec![&[30, 0, 0, 0][..]]);
    core.absorb_outputs_for_module(
        module0,
        &BindingPlan::build(&prog0).unwrap(),
        &prog0,
        vec![vec![20, 0, 0, 0]],
        &mut state,
        |idx, name| BackendError::InvalidProgram {
            fix: format!("missing output {idx} {name}"),
        },
    )
    .unwrap();
    core.absorb_outputs_for_module(
        module1,
        &BindingPlan::build(&prog1).unwrap(),
        &prog1,
        vec![vec![40, 0, 0, 0]],
        &mut state,
        |idx, name| BackendError::InvalidProgram {
            fix: format!("missing output {idx} {name}"),
        },
    )
    .unwrap();
    assert_eq!(state[&ArtifactValueId(out0.0)], [20, 0, 0, 0]);
    assert_eq!(state[&ArtifactValueId(out1.0)], [40, 0, 0, 0]);
}

/// WHY: a payload that binds an `Output`-lifetime value `WriteOnly` names no
/// input identity for it, because nothing reads it. The access on the payload
/// binding decides that, not the lifetime.
#[test]
fn write_only_binding_of_an_output_value_is_output_only() {
    let mut graph = ProgramGraph::new();
    let val_in = graph
        .add_external_value(
            "input_val",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .unwrap();

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_buf", 0, BufferAccess::ReadOnly, DataType::U32).with_count(32),
            BufferDecl::output("out_buf", 1, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "out_buf",
            Expr::u32(0),
            Expr::load("in_buf", Expr::u32(0)),
        )],
    );

    let (_, outputs) = graph
        .add_node(
            "node0",
            prog.clone(),
            vec![GraphInput {
                buffer: "in_buf".into(),
                value: val_in,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "out_buf".into(),
                name: "output_val".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let out_id = outputs[0];

    let artifact = compile_graph(graph);

    let payload = test_payload(
        &artifact,
        vec![entry_point(
            "entry0",
            ArtifactNodeId(0),
            global_bindings(&[
                (ArtifactValueId(val_in.0), TargetResourceAccess::ReadOnly),
                (ArtifactValueId(out_id.0), TargetResourceAccess::WriteOnly),
            ]),
        )],
    );

    let core = test_instance_core(&artifact, &payload).unwrap();
    assert_eq!(core.module_inputs[0], vec![ArtifactValueId(val_in.0)]);
    assert_eq!(core.module_outputs[0], vec![ArtifactValueId(out_id.0)]);

    let plan = BindingPlan::build(&prog).unwrap();
    let mut state = BTreeMap::new();
    state.insert(ArtifactValueId(val_in.0), vec![42, 0, 0, 0]);

    let gathered = core
        .gather_inputs_for_module(0, &plan, &prog, &state, unbound_input)
        .unwrap();
    assert_eq!(gathered.len(), 1);
    assert_eq!(gathered[0], &[42, 0, 0, 0]);

    core.absorb_outputs_for_module(
        0,
        &plan,
        &prog,
        vec![vec![42, 0, 0, 0]],
        &mut state,
        |idx, name| BackendError::InvalidProgram {
            fix: format!("missing output {idx} {name}"),
        },
    )
    .unwrap();

    let completion = core.completion(&state, None).unwrap();
    assert_eq!(
        completion.outputs.get(&ArtifactValueId(out_id.0)).unwrap(),
        &[42, 0, 0, 0]
    );
}

/// WHY: one Program buffer name carries two artifact identities whenever the
/// buffer is `ReadWrite`: it reads the resource it was wired to and writes the
/// renamed successor. A guard that rejected any two identities under one name
/// therefore rejected every fixpoint node in the tree, and it shipped with no
/// test, so nothing said which of the two readings was the contract. The
/// contract is that the pair is accepted when the retained chain relates the two
/// identities, because that is the same buffer at two points of its lineage, and
/// resolution walks that chain in either direction.
///
/// It does not cover a group whose members belong to different Programs: fusion
/// groups fuse nodes of one graph, so a name is only ever ambiguous through the
/// read/write pair this asserts.
#[test]
fn a_read_write_buffer_keeps_both_ends_of_its_retained_chain_under_one_name() {
    let mut graph = ProgramGraph::new();
    let state_init = graph
        .add_external_value(
            "canonical_state",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
        )
        .unwrap();

    let prog = Program::wrapped(
        vec![BufferDecl::storage(
            "state",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [32, 1, 1],
        vec![Node::store(
            "state",
            Expr::u32(0),
            Expr::add(Expr::load("state", Expr::u32(0)), Expr::u32(1)),
        )],
    );

    let (node, outputs) = graph
        .add_node(
            "step",
            prog.clone(),
            vec![GraphInput {
                buffer: "state".into(),
                value: state_init,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![GraphOutput {
                buffer: "state".into(),
                name: "state_next".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                retained_successor_of: Some(state_init),
            }],
        )
        .unwrap();
    let state_next = outputs[0];

    let artifact = compile_graph(graph);

    // The ABI is what the guard reads: one name, two identities, one chain.
    let abi_entry = artifact
        .abi()
        .entries
        .iter()
        .find(|entry| entry.node == ArtifactNodeId(node.0))
        .expect("the node must have an ABI entry");
    let read = abi_entry
        .input_bindings
        .iter()
        .find(|binding| binding.buffer == "state")
        .expect("the ReadWrite buffer must bind an input identity")
        .value;
    let written = abi_entry
        .output_bindings
        .iter()
        .find(|binding| binding.buffer == "state")
        .expect("the ReadWrite buffer must bind an output identity")
        .value;
    assert_ne!(
        read, written,
        "the fixture must exercise two identities under one name"
    );

    let payload = test_payload(
        &artifact,
        vec![entry_point(
            "step_entry",
            ArtifactNodeId(node.0),
            global_bindings(&[(
                ArtifactValueId(state_next.0),
                TargetResourceAccess::ReadWrite,
            )]),
        )],
    );

    let core = test_instance_core(&artifact, &payload)
        .expect("a ReadWrite buffer's own read and write identities are not a name collision");

    // The write identity is the one the name resolves to, and the read identity
    // remains reachable because the two share a retained chain.
    assert_eq!(
        core.module_named_resources[0].get("state").copied(),
        Some(ArtifactValueId(state_next.0))
    );
    assert!(
        core.retained_predecessors
            .get(&ArtifactValueId(state_next.0))
            .is_some_and(|priors| priors.contains(&ArtifactValueId(state_init.0))),
        "the two identities must be related by the retained chain the guard consults"
    );
}

/// WHY: accepting the read/write pair must not weaken collision rejection for
/// identities outside the same retained lineage. This covers both map
/// directions, transitive ancestry, and the fail-closed unrelated case.
#[test]
fn retained_chain_identity_guard_accepts_only_one_lineage() {
    let root = ArtifactValueId(1);
    let child = ArtifactValueId(2);
    let grandchild = ArtifactValueId(3);
    let unrelated = ArtifactValueId(4);
    let retained_predecessors =
        BTreeMap::from([(child, vec![root]), (grandchild, vec![child, root])]);

    assert!(retained_chain_relates(&retained_predecessors, root, child));
    assert!(retained_chain_relates(&retained_predecessors, child, root));
    assert!(retained_chain_relates(
        &retained_predecessors,
        root,
        grandchild
    ));
    assert!(!retained_chain_relates(
        &retained_predecessors,
        root,
        unrelated
    ));
    assert!(!retained_chain_relates(
        &retained_predecessors,
        child,
        unrelated
    ));
}

/// Read and write halves of one payload access, matched exhaustively so a new
/// `TargetResourceAccess` member stops compiling instead of silently inheriting
/// one of these answers.
fn reads_and_writes(access: TargetResourceAccess) -> (bool, bool) {
    match access {
        TargetResourceAccess::ReadOnly => (true, false),
        TargetResourceAccess::WriteOnly => (false, true),
        TargetResourceAccess::ReadWrite => (true, true),
    }
}

/// WHY: the directional projection is what tells a module which canonical value
/// carries the bytes of a buffer it reads. Classifying a `ReadWrite` binding by
/// its resource lifetime instead of its access left an `Output`-lifetime
/// resource out of the input projection, and every wgpu case whose witness binds
/// its output buffer and reads it in the same pass
/// (`vyre-libs::security::aliases_dataflow`, `flows_to_to_sink`,
/// `flows_to_with_sanitizer`, `sink_intersection`, `taint_pollution`,
/// `vyre-libs::llm::sample_token`) failed materialization before its first case
/// ran. The contract is that the payload access decides membership and the
/// lifetime decides nothing: a resource that is read names an input identity at
/// every lifetime.
///
/// The loop covers every access against every lifetime a graph output can carry,
/// so a rule reintroduced for one pair fails here. It does not cover
/// `ResourceLifetime::Constant`, which no graph output can hold.
#[test]
fn payload_access_decides_projection_membership_at_every_lifetime() {
    for access in [
        TargetResourceAccess::ReadOnly,
        TargetResourceAccess::WriteOnly,
        TargetResourceAccess::ReadWrite,
    ] {
        for lifetime in [
            ValueLifetime::Output,
            ValueLifetime::Retained,
            ValueLifetime::Invocation,
        ] {
            let mut graph = ProgramGraph::new();
            let val_in = graph
                .add_external_value(
                    "input_val",
                    contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                )
                .unwrap();
            let prog = Program::wrapped(
                vec![
                    BufferDecl::storage("in_buf", 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(32),
                    BufferDecl::storage("acc", 1, BufferAccess::ReadWrite, DataType::U32)
                        .with_count(32),
                ],
                [32, 1, 1],
                vec![Node::store(
                    "acc",
                    Expr::u32(0),
                    Expr::load("acc", Expr::u32(0)),
                )],
            );
            let (_, outputs) = graph
                .add_node(
                    "node0",
                    prog.clone(),
                    vec![GraphInput {
                        buffer: "in_buf".into(),
                        value: val_in,
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                    }],
                    vec![GraphOutput {
                        buffer: "acc".into(),
                        name: "acc_val".into(),
                        contract: contract(BufferAccess::ReadWrite, lifetime),
                        retained_successor_of: None,
                    }],
                )
                .unwrap();
            let acc_id = ArtifactValueId(outputs[0].0);

            let artifact = compile_graph(graph);
            let payload = test_payload(
                &artifact,
                vec![entry_point(
                    "entry0",
                    ArtifactNodeId(0),
                    global_bindings(&[
                        (ArtifactValueId(val_in.0), TargetResourceAccess::ReadOnly),
                        (acc_id, access),
                    ]),
                )],
            );

            let core = test_instance_core(&artifact, &payload).unwrap();
            let (reads, writes) = reads_and_writes(access);
            assert_eq!(
                core.module_inputs[0].contains(&acc_id),
                reads,
                "Fix: a {access:?} binding at lifetime {lifetime:?} must appear in the input projection exactly when it reads"
            );
            assert_eq!(
                core.module_outputs[0].contains(&acc_id),
                writes,
                "Fix: a {access:?} binding at lifetime {lifetime:?} must appear in the output projection exactly when it writes"
            );
            if !reads {
                continue;
            }

            let plan = BindingPlan::build(&prog).unwrap();
            let mut state = BTreeMap::new();
            state.insert(ArtifactValueId(val_in.0), vec![42, 0, 0, 0]);
            state.insert(acc_id, vec![7, 0, 0, 0]);
            let gathered = core
                .gather_inputs_for_module(0, &plan, &prog, &state, unbound_input)
                .expect("a read binding must resolve to the bytes bound for it");
            assert!(
                gathered.contains(&&[7, 0, 0, 0][..]),
                "Fix: the read-write buffer must gather the bytes bound to its canonical value"
            );
        }
    }
}
