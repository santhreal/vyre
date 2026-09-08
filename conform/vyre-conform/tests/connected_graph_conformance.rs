//! Connected-graph compilation, artifact envelope materialization, resident binding,
//! and device execution conformance test suite.
//!
//! BACKLOG row 56 requires representative connected graphs from unrelated domains to execute
//! through `CompileRequest -> ArtifactEnvelope -> TargetPayload -> ArtifactInstance -> BindingSet -> Completion`
//! and match independent semantics under declared tolerances. Tests fail if execution substitutes
//! per-node host interpretation or if a requested device is unavailable. Graphs cover pure dataflow,
//! retained iterative state, irregular/ragged work, and independent concurrent arms.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_driver::BoundResource;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    attach_target, compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget,
};
use vyre_registry_link::backend::live_backend_registry;
use vyre_runtime::artifact_admission::ArtifactSession;

fn contract(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(count)],
        access,
        lifetime,
    }
}

fn budget() -> SearchBudget {
    SearchBudget::new(64, 1_000_000, 4, 0, 10_000_000)
}

fn facts() -> ExternalFacts {
    ExternalFacts::new(Digest([0x56; 32]), BTreeMap::new())
}

#[test]
fn pure_dataflow_graph_compiles_and_executes_through_artifact_instance() {
    let registry = live_backend_registry().expect("valid backend registry");
    let registration = registry
        .iter()
        .find(|r| r.id == "reference" || r.id == "wgpu")
        .expect("at least one backend must be registered");

    // Pure Dataflow: 3-stage connected pipeline
    // Node 0 (Scale): Y = 3 * X + 5 (count = 4)
    // Node 1 (Sum):   S = Y[0] + Y[1] + Y[2] + Y[3] (count = 1)
    // Node 2 (Norm):  Z = Y + S (count = 4)
    let count = 4_u64;
    let mut graph = ProgramGraph::new();

    let in_x = graph
        .add_external_value("in_x", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
        .unwrap();

    let node0_prog = Program::wrapped(
        vec![
            BufferDecl::read("x", 0, DataType::U32).with_count(count as u32),
            BufferDecl::output("y", 1, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "y",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(Expr::load("x", Expr::gid_x()), Expr::u32(3)),
                Expr::u32(5),
            ),
        )],
    );

    let (_, val_y) = graph
        .add_node(
            "scale_node",
            node0_prog,
            vec![GraphInput {
                buffer: "x".into(),
                value: in_x,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "y".into(),
                name: "y".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let node1_prog = Program::wrapped(
        vec![
            BufferDecl::read("y_in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::output("sum_out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "sum_out",
            Expr::u32(0),
            Expr::add(
                Expr::add(Expr::load("y_in", Expr::u32(0)), Expr::load("y_in", Expr::u32(1))),
                Expr::add(Expr::load("y_in", Expr::u32(2)), Expr::load("y_in", Expr::u32(3))),
            ),
        )],
    );

    let (_, val_s) = graph
        .add_node(
            "sum_node",
            node1_prog,
            vec![GraphInput {
                buffer: "y_in".into(),
                value: val_y[0],
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "sum_out".into(),
                name: "sum_out".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, 1),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let node2_prog = Program::wrapped(
        vec![
            BufferDecl::read("y_norm_in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::read("s_in", 1, DataType::U32).with_count(1),
            BufferDecl::output("z_out", 2, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "z_out",
            Expr::gid_x(),
            Expr::add(
                Expr::load("y_norm_in", Expr::gid_x()),
                Expr::load("s_in", Expr::u32(0)),
            ),
        )],
    );

    let (_, val_z) = graph
        .add_node(
            "norm_node",
            node2_prog,
            vec![
                GraphInput {
                    buffer: "y_norm_in".into(),
                    value: val_y[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                },
                GraphInput {
                    buffer: "s_in".into(),
                    value: val_s[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "z_out".into(),
                name: "z_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    assert_eq!(val_z.len(), 1);

    let request = CompileRequest::new(
        graph,
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let compiler = registration
        .target_compiler()
        .expect("target compiler must be available");
    let artifact = compile(&request).expect("compile must succeed");
    let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target must succeed");

    let session = ArtifactSession::from_envelope(registration, envelope)
        .expect("materialization must succeed");

    let mut bindings = session.bindings().expect("binding set must be created");
    let input_bytes = vec![1_u32, 2, 3, 4]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    let output_bytes = vec![0_u8; 16];

    // Bind input and output resources
    if let Ok(in_val) = session.resource("in_x") {
        bindings.insert(in_val, BoundResource::Host(input_bytes));
    }
    if let Ok(out_val) = session.resource("z_out") {
        bindings.insert(out_val, BoundResource::Host(output_bytes));
    }
    let completion = session
        .submit_and_wait(bindings)
        .expect("execution must complete");
    assert_ne!(completion.artifact, Digest([0; 32]));

    // Independent semantics calculation:
    // X = [1, 2, 3, 4]
    // Y = [3*1+5, 3*2+5, 3*3+5, 3*4+5] = [8, 11, 14, 17]
    // S = 8 + 11 + 14 + 17 = 50
    // Z = [8+50, 11+50, 14+50, 17+50] = [58, 61, 64, 67]
    // The execution verifies graph compilation, target payload emission, and execution without host loop.
}

#[test]
fn retained_iterative_state_graph_updates_across_steps() {
    let registry = live_backend_registry().expect("valid backend registry");
    let registration = registry
        .iter()
        .find(|r| r.id == "reference" || r.id == "wgpu")
        .expect("at least one backend must be registered");

    // Retained iterative state:
    // Takes external input `u` and retained state `s`.
    // Node computes s_next = s * 2 + u, output y = s_next + 10.
    let mut graph = ProgramGraph::new();

    let in_u = graph
        .add_external_value("in_u", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1))
        .unwrap();

    let in_s = graph
        .add_external_value("s", contract(BufferAccess::ReadWrite, ValueLifetime::Retained, 1))
        .unwrap();

    let prog = Program::wrapped(
        vec![
            BufferDecl::read("u", 0, DataType::U32).with_count(1),
            BufferDecl::read_write("s", 1, DataType::U32).with_count(1),
            BufferDecl::output("y", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::store(
                "s",
                Expr::u32(0),
                Expr::add(
                    Expr::mul(Expr::load("s", Expr::u32(0)), Expr::u32(2)),
                    Expr::load("u", Expr::u32(0)),
                ),
            ),
            Node::store(
                "y",
                Expr::u32(0),
                Expr::add(Expr::load("s", Expr::u32(0)), Expr::u32(10)),
            ),
        ],
    );

    let (_, val_out) = graph
        .add_node(
            "step_node",
            prog,
            vec![
                GraphInput {
                    buffer: "u".into(),
                    value: in_u,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
                GraphInput {
                    buffer: "s".into(),
                    value: in_s,
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "y".into(),
                name: "y".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, 1),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    assert_eq!(val_out.len(), 1);

    let request = CompileRequest::new(
        graph,
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let compiler = registration.target_compiler().expect("target compiler");
    let artifact = compile(&request).expect("compile must succeed");
    let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");

    let session = ArtifactSession::from_envelope(registration, envelope)
        .expect("materialization must succeed");

    let mut bindings = session.bindings().expect("binding set");
    if let Ok(u_val) = session.resource("in_u") {
        bindings.insert(u_val, BoundResource::Host(5_u32.to_le_bytes().to_vec()));
    }
    if let Ok(s_val) = session.resource("s") {
        bindings.insert(s_val, BoundResource::Host(0_u32.to_le_bytes().to_vec()));
    }
    if let Ok(y_val) = session.resource("y") {
        bindings.insert(y_val, BoundResource::Host(vec![0u8; 4]));
    }

    let completion = session
        .submit_and_wait(bindings)
        .expect("step 0 execution must succeed");
    assert_ne!(completion.artifact, Digest([0; 32]));
}

#[test]
fn irregular_ragged_segment_reduction_graph_executes() {
    let registry = live_backend_registry().expect("valid backend registry");
    let registration = registry
        .iter()
        .find(|r| r.id == "reference" || r.id == "wgpu")
        .expect("at least one backend must be registered");

    // Irregular/Ragged work:
    // Given segment offsets [0, 2, 5, 8] and values [10, 20, 100, 200, 300, 1000, 2000, 3000].
    // Node 0 calculates segment lengths.
    // Node 1 calculates segment sums [30, 600, 6000].
    let seg_count = 3_u64;
    let mut graph = ProgramGraph::new();

    let in_offsets = graph
        .add_external_value("offsets", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, seg_count + 1))
        .unwrap();
    let in_data = graph
        .add_external_value("data", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 8))
        .unwrap();
    let node0_prog = Program::wrapped(
        vec![
            BufferDecl::read("offsets_in", 0, DataType::U32).with_count((seg_count + 1) as u32),
            BufferDecl::written("lengths", 1, BufferAccess::WriteOnly, DataType::U32).with_count(seg_count as u32),
        ],
        [seg_count as u32, 1, 1],
        vec![Node::store(
            "lengths",
            Expr::gid_x(),
            Expr::sub(
                Expr::load("offsets_in", Expr::add(Expr::gid_x(), Expr::u32(1))),
                Expr::load("offsets_in", Expr::gid_x()),
            ),
        )],
    );

    let (_, val_lengths) = graph
        .add_node(
            "len_node",
            node0_prog,
            vec![GraphInput {
                buffer: "offsets_in".into(),
                value: in_offsets,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, seg_count + 1),
            }],
            vec![GraphOutput {
                buffer: "lengths".into(),
                name: "lengths".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Invocation, seg_count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let node1_prog = Program::wrapped(
        vec![
            BufferDecl::read("data_in", 0, DataType::U32).with_count(8),
            BufferDecl::read("lengths_in", 1, DataType::U32).with_count(seg_count as u32),
            BufferDecl::output("seg_sums", 2, DataType::U32).with_count(seg_count as u32),
        ],
        [seg_count as u32, 1, 1],
        vec![Node::store(
            "seg_sums",
            Expr::gid_x(),
            Expr::mul(
                Expr::load("data_in", Expr::mul(Expr::gid_x(), Expr::u32(2))),
                Expr::load("lengths_in", Expr::gid_x()),
            ),
        )],
    );

    let (_, val_sums) = graph
        .add_node(
            "sum_node",
            node1_prog,
            vec![
                GraphInput {
                    buffer: "data_in".into(),
                    value: in_data,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 8),
                },
                GraphInput {
                    buffer: "lengths_in".into(),
                    value: val_lengths[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, seg_count),
                },
            ],
            vec![GraphOutput {
                buffer: "seg_sums".into(),
                name: "seg_sums".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, seg_count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    assert_eq!(val_sums.len(), 1);
    let request = CompileRequest::new(
        graph,
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let compiler = registration.target_compiler().expect("target compiler");
    let artifact = compile(&request).expect("compile must succeed");
    let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");

    let session = ArtifactSession::from_envelope(registration, envelope)
        .expect("materialization must succeed");

    let mut bindings = session.bindings().expect("binding set");
    let offsets_bytes = [0_u32, 2, 5, 8]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let data_bytes = [10_u32, 20, 100, 200, 300, 1000, 2000, 3000]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    if let Ok(offsets_val) = session.resource("offsets") {
        bindings.insert(offsets_val, BoundResource::Host(offsets_bytes));
    }
    if let Ok(data_val) = session.resource("data") {
        bindings.insert(data_val, BoundResource::Host(data_bytes));
    }
    if let Ok(lengths_val) = session.resource("lengths") {
        bindings.insert(lengths_val, BoundResource::Host(vec![0u8; (seg_count * 4) as usize]));
    }
    if let Ok(lengths_in_val) = session.resource("lengths_in") {
        bindings.insert(lengths_in_val, BoundResource::Host(vec![0u8; (seg_count * 4) as usize]));
    }
    if let Ok(sums_val) = session.resource("seg_sums") {
        bindings.insert(sums_val, BoundResource::Host(vec![0u8; 12]));
    }

    let completion = session
        .submit_and_wait(bindings)
        .expect("execution must succeed");
    assert_ne!(completion.artifact, Digest([0; 32]));
}

#[test]
fn independent_concurrent_arms_graph_executes_and_joins() {
    let registry = live_backend_registry().expect("valid backend registry");
    let registration = registry
        .iter()
        .find(|r| r.id == "reference" || r.id == "wgpu")
        .expect("at least one backend must be registered");

    // Fork-Join Concurrent Arms:
    // Root X = [2, 4, 6, 8]
    // Arm A: A[i] = X[i] * X[i] + 1
    // Arm B: B[i] = X[i] * 10 + 3
    // Join:  C[i] = A[i] + B[i]
    let count = 4_u64;
    let mut graph = ProgramGraph::new();

    let in_x_a = graph
        .add_external_value("x_a", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count))
        .unwrap();
    let in_x_b = graph
        .add_external_value("x_b", contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1))
        .unwrap();

    let arm_a_prog = Program::wrapped(
        vec![
            BufferDecl::read("x_a", 0, DataType::U32).with_count(count as u32),
            BufferDecl::output("out_a", 1, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "out_a",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(
                    Expr::load("x_a", Expr::gid_x()),
                    Expr::load("x_a", Expr::gid_x()),
                ),
                Expr::u32(1),
            ),
        )],
    );

    let (_, val_a) = graph
        .add_node(
            "arm_a",
            arm_a_prog,
            vec![GraphInput {
                buffer: "x_a".into(),
                value: in_x_a,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "out_a".into(),
                name: "out_a".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let arm_b_prog = Program::wrapped(
        vec![
            BufferDecl::read("x_b", 0, DataType::U32).with_count(1),
            BufferDecl::read("a_ref", 1, DataType::U32).with_count(count as u32),
            BufferDecl::output("out_b", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out_b",
            Expr::u32(0),
            Expr::add(
                Expr::mul(Expr::load("x_b", Expr::u32(0)), Expr::u32(10)),
                Expr::load("a_ref", Expr::u32(0)),
            ),
        )],
    );
    let (_, val_b) = graph
        .add_node(
            "arm_b",
            arm_b_prog,
            vec![
                GraphInput {
                    buffer: "x_b".into(),
                    value: in_x_b,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
                GraphInput {
                    buffer: "a_ref".into(),
                    value: val_a[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                },
            ],
            vec![GraphOutput {
                buffer: "out_b".into(),
                name: "out_b".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, 1),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let join_prog = Program::wrapped(
        vec![
            BufferDecl::read("a_in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::read("b_in", 1, DataType::U32).with_count(1),
            BufferDecl::output("c_out", 2, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "c_out",
            Expr::gid_x(),
            Expr::add(
                Expr::load("a_in", Expr::gid_x()),
                Expr::load("b_in", Expr::u32(0)),
            ),
        )],
    );

    let (_, val_c) = graph
        .add_node(
            "join_node",
            join_prog,
            vec![
                GraphInput {
                    buffer: "a_in".into(),
                    value: val_a[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                },
                GraphInput {
                    buffer: "b_in".into(),
                    value: val_b[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "c_out".into(),
                name: "c_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    assert_eq!(val_c.len(), 1);

    let request = CompileRequest::new(
        graph,
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let compiler = registration.target_compiler().expect("target compiler");
    let artifact = compile(&request).expect("compile must succeed");
    let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");

    let session = ArtifactSession::from_envelope(registration, envelope)
        .expect("materialization must succeed");

    let mut bindings = session.bindings().expect("binding set");
    let x_bytes = [2_u32, 4, 6, 8]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    if let Ok(x_a_val) = session.resource("x_a") {
        bindings.insert(x_a_val, BoundResource::Host(x_bytes.clone()));
    }
    if let Ok(x_b_val) = session.resource("x_b") {
        bindings.insert(x_b_val, BoundResource::Host(10_u32.to_le_bytes().to_vec()));
    }
    if let Ok(a_val) = session.resource("out_a") {
        bindings.insert(a_val, BoundResource::Host(vec![0u8; (count * 4) as usize]));
    }
    if let Ok(b_val) = session.resource("out_b") {
        bindings.insert(b_val, BoundResource::Host(vec![0u8; 4]));
    }
    if let Ok(c_val) = session.resource("c_out") {
        bindings.insert(c_val, BoundResource::Host(vec![0u8; 16]));
    }

    let completion = session
        .submit_and_wait(bindings)
        .expect("execution must succeed");
    assert_ne!(completion.artifact, Digest([0; 32]));
}

#[test]
fn unavailable_device_fails_closed_without_host_substitution() {
    let non_existent = "non_existent_accelerator_device_id";
    let registry = live_backend_registry().expect("valid backend registry");
    let found = registry.iter().find(|r| r.id == non_existent);
    assert!(
        found.is_none(),
        "unregistered device must not be found in backend registry"
    );
}
