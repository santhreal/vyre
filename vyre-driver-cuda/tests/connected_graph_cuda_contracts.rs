//! CUDA connected-graph production route contracts.
//!
//! Four connected graphs from unrelated domains execute through
//! `CompileRequest -> ArtifactEnvelope -> TargetPayload -> ArtifactInstance
//! -> BindingSet -> Completion` on the acquired device and are compared
//! against an independent host implementation of the same mathematics under a
//! tolerance declared per graph.
//!
//! The four shapes are covered once each: pure dataflow, retained iterative
//! state, irregular ragged work, and independent concurrent arms that join.
//! Every graph is multi-node with at least one intra-graph value edge, so a
//! single-node fast path proves none of them.
//!
//! Output bytes cannot separate the artifact route from a per-node host
//! interpretation of the same graph, because both compute the same answer.
//! [`DeviceExecutionEvidence`] is what separates them, and every graph test
//! asserts it through the same choke point.

#![cfg(all(test, feature = "device-tests"))]

use std::collections::{BTreeMap, BTreeSet};

use vyre_driver::{BackendRegistration, BoundResource, Completion, DeviceIdentity};
use vyre_driver_cuda::{registered_backend_id, CUDA_BACKEND_ID};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    attach_target, compile, ArtifactValueId, CompileObjective, CompileRequest, Digest,
    ExternalFacts, ObjectiveMetric, SearchBudget,
};
use vyre_runtime::artifact_admission::ArtifactSession;
use vyre_test_support::backend_execution_domain::assert_dispatch_leaves_the_host;
// The pure-dataflow graph and its host oracle: every concrete driver is asked
// the same question, so an answer that differs by which suite built the graph
// proves nothing about the device.
use vyre_test_support::graph_fixtures::{
    pure_dataflow_graph, pure_dataflow_oracle, PURE_DATAFLOW_LANES,
};

fn contract(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract::dense_1d(DataType::U32, count, access, lifetime)
}

fn f32_contract(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract::dense_1d(DataType::F32, count, access, lifetime)
}

fn budget() -> SearchBudget {
    SearchBudget::new(64, 1_000_000, 4, 0, 10_000_000)
}

fn facts() -> ExternalFacts {
    ExternalFacts::new(Digest([0x56; 32]), BTreeMap::new())
}

/// The registered CUDA backend, or a configuration failure naming the device.
///
/// A missing device is never a skip. `device-tests` selects whether this file
/// is compiled at all; once compiled, the device is required and its absence
/// fails the suite here with the driver's own diagnosis.
fn required_cuda_registration() -> &'static BackendRegistration {
    vyre_driver::backend_registration(CUDA_BACKEND_ID).unwrap_or_else(|error| {
        panic!(
            "requested device `{CUDA_BACKEND_ID}` is unavailable. A requested device that cannot \
             be resolved fails the run; no host route substitutes for it. Fix: link and install \
             the CUDA driver crate for this host: {error}"
        )
    })
}

/// One compiled connected graph admitted onto the acquired device.
struct DeviceGraphRun {
    session: ArtifactSession,
    artifact_digest: Digest,
    payload_digest: Digest,
    node_count: usize,
}

/// Records that only the artifact route produces.
///
/// The payload digest is the target compiler's emitted module identity, the
/// device identity is the acquired generation that loaded that module, and
/// `device_ns` is the CUDA event measurement taken around the launch. A host
/// interpretation of the graph's nodes has no compiled module, no device
/// generation, and no event pair, so it cannot present any of them.
struct DeviceExecutionEvidence {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
    device_ns: Option<u64>,
    outputs: BTreeSet<ArtifactValueId>,
    retained: BTreeSet<ArtifactValueId>,
}

impl DeviceGraphRun {
    /// Compile `graph` for the acquired device and admit it for submission.
    fn admit(graph: ProgramGraph) -> Self {
        let registration = required_cuda_registration();
        let materializer = registration
            .materializer()
            .expect("CUDA materializer acquisition must succeed");
        assert!(
            materializer.device().is_healthy(),
            "acquired CUDA device must be healthy before admission"
        );
        let device = registration
            .acquire()
            .expect("CUDA device acquisition must succeed");
        let device_facts = device.device_profile().compile_facts();

        let node_count = graph.nodes().len();
        assert!(
            node_count > 1,
            "a connected-graph contract needs more than one node; this graph has {node_count}"
        );

        let request = CompileRequest::new(
            graph,
            facts(),
            device_facts,
            budget(),
            CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
        )
        .validate()
        .expect("compile request must validate");

        let compiler = registration
            .target_compiler()
            .expect("CUDA target compiler must be registered");
        let artifact = compile(&request).expect("compile must succeed");
        assert_eq!(
            artifact.nodes().len(),
            node_count,
            "the artifact must retain every graph node"
        );
        let artifact_digest = artifact.digest();
        assert_ne!(artifact_digest, Digest([0; 32]));

        let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");
        let payloads = envelope.target_payloads();
        assert_eq!(
            payloads.len(),
            1,
            "one registered target compiler emits one payload"
        );
        let payload_digest = payloads[0].digest();
        assert_ne!(payload_digest, Digest([0; 32]));

        let session =
            ArtifactSession::from_envelope(registration, envelope).expect("materialization");
        Self {
            session,
            artifact_digest,
            payload_digest,
            node_count,
        }
    }

    /// Bind one host resource by artifact resource name.
    ///
    /// A name the artifact does not expose is a compile defect, not a reason
    /// to submit an incomplete binding set.
    fn resource(&self, name: &str) -> ArtifactValueId {
        self.session
            .resource(name)
            .unwrap_or_else(|error| panic!("artifact must expose resource `{name}`: {error}"))
    }

    /// Submit `bindings` and return the completion plus its device evidence.
    fn submit(&self, bindings: vyre_driver::BindingSet) -> (Completion, DeviceExecutionEvidence) {
        let completion = self
            .session
            .submit_and_wait(bindings)
            .expect("submission must complete on the device");
        let evidence = DeviceExecutionEvidence {
            artifact: completion.artifact,
            payload: self.session.payload().expect("session payload identity"),
            device: self.session.device().expect("session device identity"),
            device_ns: completion.device_ns,
            outputs: completion.outputs.keys().copied().collect(),
            retained: completion.retained.keys().copied().collect(),
        };
        (completion, evidence)
    }

    /// Assert the submission ran as this artifact on this device.
    ///
    /// `domain` names the problem domain the graph came from, so a failure
    /// states which of the four shapes lost its device route.
    fn assert_device_executed(
        &self,
        evidence: &DeviceExecutionEvidence,
        domain: &str,
        expected_outputs: &[ArtifactValueId],
        expected_retained: &[ArtifactValueId],
    ) {
        assert_eq!(
            evidence.artifact, self.artifact_digest,
            "{domain}: the completion must name the compiled artifact"
        );
        assert_eq!(
            evidence.artifact,
            self.session.artifact().expect("session artifact identity"),
            "{domain}: the session and the completion must agree on artifact identity"
        );
        assert_eq!(
            evidence.payload, self.payload_digest,
            "{domain}: the admitted payload must be the one the target compiler emitted"
        );
        assert_eq!(
            evidence.device.backend, CUDA_BACKEND_ID,
            "{domain}: the executing device must be the requested backend"
        );
        assert!(
            !evidence.device.device.is_empty(),
            "{domain}: the acquired device generation must identify its physical device"
        );
        let device_ns = evidence.device_ns.unwrap_or_else(|| {
            panic!(
                "{domain}: the completion carries no device-measured duration. The artifact route \
                 measures CUDA events around every launch, so an absent measurement means the \
                 outputs did not come from a device launch."
            )
        });
        assert!(
            device_ns > 0,
            "{domain}: device-measured duration must be positive, got {device_ns} ns"
        );
        assert_eq!(
            evidence.outputs,
            expected_outputs.iter().copied().collect::<BTreeSet<_>>(),
            "{domain}: the completion must return exactly the declared output values"
        );
        assert_eq!(
            evidence.retained,
            expected_retained.iter().copied().collect::<BTreeSet<_>>(),
            "{domain}: the completion must return exactly the declared retained values"
        );
        assert!(
            self.node_count > 1,
            "{domain}: a single-node graph proves nothing about connected execution"
        );
    }
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_u32(bytes: &[u8], label: &str) -> Vec<u32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "{label}: readback length {} is not a whole number of u32 words",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn read_f32(bytes: &[u8], label: &str) -> Vec<f32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "{label}: readback length {} is not a whole number of f32 words",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

/// Distance between two finite `f32` values counted in representable steps.
///
/// Both operands are same-signed and finite in every case that calls this, so
/// the ordered bit patterns of the magnitudes are monotonic and their
/// difference is the number of representable values between them.
fn ulp_distance(left: f32, right: f32) -> u32 {
    assert!(
        left.is_finite() && right.is_finite(),
        "ULP distance is defined for finite values only, got {left} and {right}"
    );
    assert!(
        left.is_sign_positive() == right.is_sign_positive(),
        "ULP distance across zero is not a rounding difference: {left} against {right}"
    );
    let a = left.to_bits() & 0x7fff_ffff;
    let b = right.to_bits() & 0x7fff_ffff;
    a.abs_diff(b)
}

// -------------------------------------------------------------------------
// Domain 1: image tone mapping. Pure dataflow.
//
// Declared tolerance: exact, zero representable steps. Every value is U32 and
// every operation is integer add and multiply, which the IR defines without
// rounding freedom, so any nonzero tolerance would accept a wrong answer.
// -------------------------------------------------------------------------

#[test]
fn cuda_executes_pure_dataflow_connected_graph() {
    let run = DeviceGraphRun::admit(pure_dataflow_graph());
    let in_x = run.resource("in_x");
    let z_out = run.resource("z_out");

    let x = [1_u32, 2, 3, 4];
    let output_bytes = usize::try_from(PURE_DATAFLOW_LANES).expect("lane count fits a length") * 4;
    let mut bindings = run.session.bindings().expect("binding set");
    bindings.insert(in_x, BoundResource::Host(u32_bytes(&x)));
    bindings.insert(z_out, BoundResource::Host(vec![0u8; output_bytes]));

    let (completion, evidence) = run.submit(bindings);
    run.assert_device_executed(&evidence, "image tone mapping", &[z_out], &[]);

    let actual = read_u32(
        completion.outputs.get(&z_out).expect("z_out output bytes"),
        "z_out",
    );
    assert_eq!(
        actual,
        pure_dataflow_oracle(x).to_vec(),
        "image tone mapping: device answer must equal the host oracle exactly"
    );
}

// -------------------------------------------------------------------------
// Domain 2: streaming telemetry accumulation. Retained iterative state.
//
// Declared tolerance: exact, zero representable steps. The recurrence is
// integer, and a retained accumulator that drifts by one step compounds on
// every later step, so exactness is the only defensible bound.
// -------------------------------------------------------------------------

/// state' = 2*state + u, snapshot = state', y = snapshot + 10.
///
/// Two nodes joined by an invocation-lifetime snapshot value, over one
/// retained state value the device updates in place.
fn telemetry_accumulator_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in_u = graph
        .add_external_value(
            "in_u",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
        )
        .expect("external sample value");
    let state = graph
        .add_external_value(
            "state",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained, 1),
        )
        .expect("external retained value");

    let fold = Program::wrapped(
        vec![
            BufferDecl::read("u", 0, DataType::U32).with_count(1),
            BufferDecl::read_write("s", 1, DataType::U32).with_count(1),
            BufferDecl::written("s_out", 2, BufferAccess::WriteOnly, DataType::U32).with_count(1),
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
            Node::store("s_out", Expr::u32(0), Expr::load("s", Expr::u32(0))),
        ],
    );
    let (_, snapshot) = graph
        .add_node(
            "fold_node",
            fold,
            vec![
                GraphInput {
                    buffer: "u".into(),
                    value: in_u,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
                GraphInput {
                    buffer: "s".into(),
                    value: state,
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "s_out".into(),
                name: "s_snapshot".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Invocation, 1),
                retained_successor_of: None,
            }],
        )
        .expect("fold node");

    let emit = Program::wrapped(
        vec![
            BufferDecl::read("snap", 0, DataType::U32).with_count(1),
            BufferDecl::output("y", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "y",
            Expr::u32(0),
            Expr::add(Expr::load("snap", Expr::u32(0)), Expr::u32(10)),
        )],
    );
    graph
        .add_node(
            "emit_node",
            emit,
            vec![GraphInput {
                buffer: "snap".into(),
                value: snapshot[0],
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
            }],
            vec![GraphOutput {
                buffer: "y".into(),
                name: "y".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, 1),
                retained_successor_of: None,
            }],
        )
        .expect("emit node");
    graph
}

#[test]
fn cuda_executes_retained_iterative_state_across_steps() {
    let run = DeviceGraphRun::admit(telemetry_accumulator_graph());
    let in_u = run.resource("in_u");
    let state = run.resource("state");
    let y = run.resource("y");

    let mut host_state = 0_u32;
    for sample in [5_u32, 3, 7] {
        let mut bindings = run.session.bindings().expect("binding set");
        bindings.insert(in_u, BoundResource::Host(u32_bytes(&[sample])));
        bindings.insert(state, BoundResource::Host(u32_bytes(&[host_state])));
        bindings.insert(y, BoundResource::Host(vec![0u8; 4]));

        let (completion, evidence) = run.submit(bindings);
        run.assert_device_executed(&evidence, "streaming telemetry", &[y], &[state]);

        host_state = host_state * 2 + sample;
        assert_eq!(
            read_u32(
                completion.retained.get(&state).expect("retained state"),
                "state"
            ),
            vec![host_state],
            "streaming telemetry: retained state must equal the host recurrence exactly"
        );
        assert_eq!(
            read_u32(completion.outputs.get(&y).expect("y output"), "y"),
            vec![host_state + 10],
            "streaming telemetry: emitted value must equal the host oracle exactly"
        );
    }
    assert_eq!(
        host_state, 33,
        "streaming telemetry: three steps of the recurrence over [5, 3, 7] reach 33"
    );
}

// -------------------------------------------------------------------------
// Domain 3: sparse matrix row reduction. Irregular ragged work.
//
// Declared tolerance: exact, zero representable steps. The trip count is data
// dependent, so a wrong bound reads foreign elements or drops real ones; both
// change the integer answer, and neither is a rounding difference.
// -------------------------------------------------------------------------

/// Per-row sums over a CSR offsets array, then the largest row sum.
///
/// The offsets include an empty row, which is the boundary case a fixed trip
/// count silently gets wrong.
fn csr_row_reduction_graph(rows: u64, nnz: u64) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let offsets = graph
        .add_external_value(
            "offsets",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, rows + 1),
        )
        .expect("external offsets value");
    let values = graph
        .add_external_value(
            "values",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, nnz),
        )
        .expect("external values value");

    let row_sums = Program::wrapped(
        vec![
            BufferDecl::read("off", 0, DataType::U32).with_count(rows as u32 + 1),
            BufferDecl::read("val", 1, DataType::U32).with_count(nnz as u32),
            BufferDecl::read_write("seg", 2, DataType::U32).with_count(rows as u32),
        ],
        [rows as u32, 1, 1],
        vec![
            Node::store("seg", Expr::gid_x(), Expr::u32(0)),
            Node::loop_for(
                "t",
                Expr::load("off", Expr::gid_x()),
                Expr::load("off", Expr::add(Expr::gid_x(), Expr::u32(1))),
                vec![Node::store(
                    "seg",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::load("seg", Expr::gid_x()),
                        Expr::load("val", Expr::var("t")),
                    ),
                )],
            ),
        ],
    );
    let (_, sums) = graph
        .add_node(
            "row_sum_node",
            row_sums,
            vec![
                GraphInput {
                    buffer: "off".into(),
                    value: offsets,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, rows + 1),
                },
                GraphInput {
                    buffer: "val".into(),
                    value: values,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, nnz),
                },
            ],
            vec![GraphOutput {
                buffer: "seg".into(),
                name: "row_sums".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output, rows),
                retained_successor_of: None,
            }],
        )
        .expect("row sum node");

    let widest = Program::wrapped(
        vec![
            BufferDecl::read("sums", 0, DataType::U32).with_count(rows as u32),
            BufferDecl::output("widest_row", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "widest_row",
            Expr::u32(0),
            Expr::max(
                Expr::max(
                    Expr::load("sums", Expr::u32(0)),
                    Expr::load("sums", Expr::u32(1)),
                ),
                Expr::max(
                    Expr::load("sums", Expr::u32(2)),
                    Expr::load("sums", Expr::u32(3)),
                ),
            ),
        )],
    );
    graph
        .add_node(
            "widest_row_node",
            widest,
            vec![GraphInput {
                buffer: "sums".into(),
                value: sums[0],
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Output, rows),
            }],
            vec![GraphOutput {
                buffer: "widest_row".into(),
                name: "widest_row".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, 1),
                retained_successor_of: None,
            }],
        )
        .expect("widest row node");
    graph
}

/// Independent host implementation of the CSR row reduction.
fn csr_row_reduction_oracle(offsets: &[u32], values: &[u32]) -> (Vec<u32>, u32) {
    let sums: Vec<u32> = offsets
        .windows(2)
        .map(|bounds| values[bounds[0] as usize..bounds[1] as usize].iter().sum())
        .collect();
    let widest = sums.iter().copied().max().expect("at least one row");
    (sums, widest)
}

#[test]
fn cuda_executes_irregular_ragged_connected_graph() {
    let offsets = [0_u32, 3, 3, 5, 8];
    let values = [1_u32, 2, 3, 4, 5, 6, 7, 8];
    let rows = (offsets.len() - 1) as u64;
    let run = DeviceGraphRun::admit(csr_row_reduction_graph(rows, values.len() as u64));

    let offsets_res = run.resource("offsets");
    let values_res = run.resource("values");
    let row_sums = run.resource("row_sums");
    let widest = run.resource("widest_row");

    let mut bindings = run.session.bindings().expect("binding set");
    bindings.insert(offsets_res, BoundResource::Host(u32_bytes(&offsets)));
    bindings.insert(values_res, BoundResource::Host(u32_bytes(&values)));
    bindings.insert(row_sums, BoundResource::Host(vec![0u8; 16]));
    bindings.insert(widest, BoundResource::Host(vec![0u8; 4]));

    let (completion, evidence) = run.submit(bindings);
    run.assert_device_executed(
        &evidence,
        "sparse matrix row reduction",
        &[row_sums, widest],
        &[],
    );

    let (expected_sums, expected_widest) = csr_row_reduction_oracle(&offsets, &values);
    assert_eq!(
        expected_sums,
        vec![6, 0, 9, 21],
        "sparse matrix row reduction: the fixture must contain an empty row"
    );
    assert_eq!(
        read_u32(
            completion.outputs.get(&row_sums).expect("row_sums output"),
            "row_sums"
        ),
        expected_sums,
        "sparse matrix row reduction: per-row sums must equal the host oracle exactly"
    );
    assert_eq!(
        read_u32(
            completion.outputs.get(&widest).expect("widest_row output"),
            "widest_row"
        ),
        vec![expected_widest],
        "sparse matrix row reduction: widest row must equal the host oracle exactly"
    );
}

// -------------------------------------------------------------------------
// Domain 4: dual-band audio gain. Independent concurrent arms.
//
// Declared tolerance: one representable step. The CUDA backend honors
// FloatLoweringMode::Contracted and refuses StrictIeee, so each arm's
// multiply-add is allowed to contract into a single FMA and drop the
// intermediate rounding of the product. The host oracle rounds twice. Every
// addend in this fixture is positive, so there is no cancellation and the
// difference between one rounding and two is bounded by one step of the
// result. Zero would reject a legal device answer and two would accept an
// answer with a real arithmetic error.
// -------------------------------------------------------------------------

const DUAL_BAND_TOLERANCE_ULP: u32 = 1;
const LOW_GAIN: f32 = 0.375;
const LOW_BIAS: f32 = 2.25;
const HIGH_GAIN: f32 = 0.6875;
const HIGH_BIAS: f32 = 1.125;

/// Two arms that read the same signal and never read each other, then a join.
fn dual_band_gain_graph(count: u64) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let signal = graph
        .add_external_value(
            "signal",
            f32_contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
        )
        .expect("external signal value");

    let arm = |input: &str, output: &str, gain: f32, bias: f32| {
        Program::wrapped(
            vec![
                BufferDecl::read(input, 0, DataType::F32).with_count(count as u32),
                BufferDecl::written(output, 1, BufferAccess::WriteOnly, DataType::F32)
                    .with_count(count as u32),
            ],
            [count as u32, 1, 1],
            vec![Node::store(
                output,
                Expr::gid_x(),
                Expr::add(
                    Expr::mul(Expr::load(input, Expr::gid_x()), Expr::f32(gain)),
                    Expr::f32(bias),
                ),
            )],
        )
    };

    let mut arm_values = Vec::new();
    for (name, input, output, gain, bias) in [
        ("low_band_node", "sig_lo", "lo", LOW_GAIN, LOW_BIAS),
        ("high_band_node", "sig_hi", "hi", HIGH_GAIN, HIGH_BIAS),
    ] {
        let (_, produced) = graph
            .add_node(
                name,
                arm(input, output, gain, bias),
                vec![GraphInput {
                    buffer: input.into(),
                    value: signal,
                    contract: f32_contract(
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                        count,
                    ),
                }],
                vec![GraphOutput {
                    buffer: output.into(),
                    name: output.into(),
                    contract: f32_contract(
                        BufferAccess::WriteOnly,
                        ValueLifetime::Invocation,
                        count,
                    ),
                    retained_successor_of: None,
                }],
            )
            .expect("band arm node");
        arm_values.push(produced[0]);
    }

    let mix = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::F32).with_count(count as u32),
            BufferDecl::read("b", 1, DataType::F32).with_count(count as u32),
            BufferDecl::output("mixed", 2, DataType::F32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "mixed",
            Expr::gid_x(),
            Expr::add(
                Expr::load("a", Expr::gid_x()),
                Expr::load("b", Expr::gid_x()),
            ),
        )],
    );
    graph
        .add_node(
            "mix_node",
            mix,
            vec![
                GraphInput {
                    buffer: "a".into(),
                    value: arm_values[0],
                    contract: f32_contract(
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                        count,
                    ),
                },
                GraphInput {
                    buffer: "b".into(),
                    value: arm_values[1],
                    contract: f32_contract(
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                        count,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: "mixed".into(),
                name: "mixed".into(),
                contract: f32_contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                retained_successor_of: None,
            }],
        )
        .expect("mix node");
    graph
}

/// Independent host implementation of the dual-band gain graph.
fn dual_band_gain_oracle(signal: &[f32]) -> Vec<f32> {
    signal
        .iter()
        .map(|sample| (sample * LOW_GAIN + LOW_BIAS) + (sample * HIGH_GAIN + HIGH_BIAS))
        .collect()
}

#[test]
fn cuda_executes_independent_concurrent_arms_connected_graph() {
    let signal = [0.1_f32, 0.3, 7.7, 1234.5];
    let count = signal.len() as u64;
    let run = DeviceGraphRun::admit(dual_band_gain_graph(count));

    let signal_res = run.resource("signal");
    let mixed = run.resource("mixed");

    let mut bindings = run.session.bindings().expect("binding set");
    bindings.insert(signal_res, BoundResource::Host(f32_bytes(&signal)));
    bindings.insert(mixed, BoundResource::Host(vec![0u8; 16]));

    let (completion, evidence) = run.submit(bindings);
    run.assert_device_executed(&evidence, "dual-band audio gain", &[mixed], &[]);

    let actual = read_f32(
        completion.outputs.get(&mixed).expect("mixed output"),
        "mixed",
    );
    let expected = dual_band_gain_oracle(&signal);
    assert_eq!(actual.len(), expected.len());
    for (lane, (measured, oracle)) in actual.iter().zip(expected.iter()).enumerate() {
        let distance = ulp_distance(*measured, *oracle);
        assert!(
            distance <= DUAL_BAND_TOLERANCE_ULP,
            "dual-band audio gain: lane {lane} is {distance} representable steps from the host \
             oracle, past the declared tolerance of {DUAL_BAND_TOLERANCE_ULP}. \
             device={measured:e} oracle={oracle:e}"
        );
    }
}

// -------------------------------------------------------------------------
// Device availability. No skips.
// -------------------------------------------------------------------------

/// Every backend this link unit registers must reach a device, and reach it
/// off the host.
///
/// The set is read from the registry rather than written down, so a driver
/// crate linked into this binary tomorrow is acquired here without anyone
/// editing this test. The floor assertion below the read is what keeps that
/// from becoming vacuous: an empty registry fails instead of passing.
#[test]
fn every_registered_production_backend_must_acquire_its_device() {
    let registrations =
        vyre_driver::registered_backends_by_precedence().expect("backend registry initialization");
    assert!(
        !registrations.is_empty(),
        "no backend is linked into this test binary. Fix: link a concrete driver crate."
    );

    let mut production = Vec::new();
    for registration in registrations {
        if registration.reference_oracle {
            continue;
        }
        production.push(registration.id);
        let backend = registration.acquire().unwrap_or_else(|error| {
            panic!(
                "registered backend `{}` did not acquire a device. A backend that is registered \
                 and cannot acquire is a configuration failure, not an absent device, and this \
                 run does not continue without it. Fix: repair the driver installation for `{}` \
                 or unlink the crate that registers it: {error}",
                registration.id, registration.id
            )
        });
        assert_eq!(
            backend.id(),
            registration.id,
            "an acquired backend must report the id it registered under"
        );
        assert!(
            registration.materializer.is_some(),
            "production backend `{}` registers no materializer, so no artifact can reach its \
             device. Fix: register the materializer facet for `{}`.",
            registration.id,
            registration.id
        );
        assert_dispatch_leaves_the_host(registration.id);
    }

    assert!(
        production.contains(&CUDA_BACKEND_ID),
        "the CUDA device this suite requires is not among the registered production backends \
         {production:?}"
    );
    assert_eq!(
        registered_backend_id(),
        Some(CUDA_BACKEND_ID),
        "the CUDA crate must report itself registered under its own stable id"
    );
}

#[test]
fn unavailable_requested_device_is_refused_by_name() {
    let registrations =
        vyre_driver::registered_backends_by_precedence().expect("backend registry initialization");
    let absent = "cuda-not-installed";
    assert!(
        registrations.iter().all(|reg| reg.id != absent),
        "the unavailable-device fixture `{absent}` must not be a registered backend"
    );

    let registration_error = vyre_driver::backend_registration(absent)
        .err()
        .unwrap_or_else(|| {
            panic!("resolving unavailable device `{absent}` must fail rather than substitute one")
        })
        .to_string();
    assert!(
        registration_error.contains(absent),
        "the refusal must name the requested device: {registration_error}"
    );
    assert!(
        registration_error.contains("Fix:"),
        "the refusal must state the corrective action: {registration_error}"
    );

    let acquire_error = vyre_driver::acquire(absent)
        .err()
        .unwrap_or_else(|| {
            panic!("acquiring unavailable device `{absent}` must fail rather than substitute one")
        })
        .to_string();
    assert!(
        acquire_error.contains(absent),
        "the acquisition refusal must name the requested device: {acquire_error}"
    );
    assert!(
        acquire_error.contains("Fix:"),
        "the acquisition refusal must state the corrective action: {acquire_error}"
    );
}
