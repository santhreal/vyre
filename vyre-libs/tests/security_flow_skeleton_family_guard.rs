//! Step 0 guard for the security flow-skeleton clone family.
//!
//! Three pairs of security entry points are built from one shared
//! reachability-plus-sanitizer-projection skeleton and differ only in the
//! source / sink / sanitizer predicates they supply:
//!
//! * `flows_to` / `flows_to_alias_only` / `taint_flow`  -  forward reach.
//! * `flows_to_to_sink` / `taint_pollution`  -  forward reach then sink hit.
//! * `bounded_by_comparison` / `dominance_predecessors`  -  backward reach.
//! * `flows_to_with_sanitizer`  -  the sanitizer projection of the hit form.
//!
//! Two independent halves:
//!
//! 1. `Program::fingerprint()` is pinned for every public entry point over a
//!    spread of shapes, including the boundary cases (one node / no edges,
//!    exactly one bitset word, one node past a word boundary, deep graph) and
//!    distinct buffer names. Rehoming a builder onto the shared skeleton must
//!    not move a single byte of emitted IR.
//! 2. Cross-entry-point behavior is checked through the reference interpreter,
//!    which pins the ONE thing each rehomed file is still allowed to own: its
//!    predicate. Paired entry points that declare the same predicate must agree
//!    bit for bit, and the pair that declares different edge masks must diverge
//!    exactly where those masks differ.

#![cfg(feature = "security")]
#![forbid(unsafe_code)]

use vyre::ir::Program;
use vyre_libs::security::bounded_by_comparison;
use vyre_libs::security::dominance_predecessors;
use vyre_libs::security::flows_to_to_sink;
use vyre_libs::security::flows_to_with_sanitizer;
use vyre_libs::security::taint_flow;
use vyre_libs::security::taint_pollution;
use vyre_libs::security::{flows_to, flows_to_alias_only};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::predicate::edge_kind;
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

fn hex(fingerprint: [u8; 32]) -> String {
    fingerprint.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every pinned entry point, keyed by a case label that names the builder and
/// the input it was built from. A pin only ever changes together with a
/// deliberate, recorded IR change.
fn pinned_cases() -> Vec<(&'static str, Program)> {
    vec![
        // ---- forward reach: flows_to ----
        (
            "flows_to/1n0e/fin-fout",
            flows_to(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "flows_to/4n3e/fin-fout",
            flows_to(ProgramGraphShape::new(4, 3), "fin", "fout"),
        ),
        (
            "flows_to/32n31e/fin-fout",
            flows_to(ProgramGraphShape::new(32, 31), "fin", "fout"),
        ),
        (
            "flows_to/33n32e/fin-fout",
            flows_to(ProgramGraphShape::new(33, 32), "fin", "fout"),
        ),
        (
            "flows_to/1024n4096e/renamed",
            flows_to(
                ProgramGraphShape::new(1024, 4096),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- forward reach: flows_to_alias_only ----
        (
            "flows_to_alias_only/1n0e/fin-fout",
            flows_to_alias_only(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "flows_to_alias_only/4n3e/fin-fout",
            flows_to_alias_only(ProgramGraphShape::new(4, 3), "fin", "fout"),
        ),
        (
            "flows_to_alias_only/33n32e/renamed",
            flows_to_alias_only(
                ProgramGraphShape::new(33, 32),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- forward reach: taint_flow ----
        (
            "taint_flow/1n0e/fin-fout",
            taint_flow(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "taint_flow/4n3e/fin-fout",
            taint_flow(ProgramGraphShape::new(4, 3), "fin", "fout"),
        ),
        (
            "taint_flow/32n31e/fin-fout",
            taint_flow(ProgramGraphShape::new(32, 31), "fin", "fout"),
        ),
        (
            "taint_flow/33n32e/fin-fout",
            taint_flow(ProgramGraphShape::new(33, 32), "fin", "fout"),
        ),
        (
            "taint_flow/1024n4096e/renamed",
            taint_flow(
                ProgramGraphShape::new(1024, 4096),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- backward reach: bounded_by_comparison ----
        (
            "bounded_by_comparison/1n0e/fin-fout",
            bounded_by_comparison(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "bounded_by_comparison/4n4e/fin-fout",
            bounded_by_comparison(ProgramGraphShape::new(4, 4), "fin", "fout"),
        ),
        (
            "bounded_by_comparison/32n31e/fin-fout",
            bounded_by_comparison(ProgramGraphShape::new(32, 31), "fin", "fout"),
        ),
        (
            "bounded_by_comparison/33n32e/fin-fout",
            bounded_by_comparison(ProgramGraphShape::new(33, 32), "fin", "fout"),
        ),
        (
            "bounded_by_comparison/1024n4096e/renamed",
            bounded_by_comparison(
                ProgramGraphShape::new(1024, 4096),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- backward reach: dominance_predecessors ----
        (
            "dominance_predecessors/1n0e/fin-fout",
            dominance_predecessors(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "dominance_predecessors/4n4e/fin-fout",
            dominance_predecessors(ProgramGraphShape::new(4, 4), "fin", "fout"),
        ),
        (
            "dominance_predecessors/32n31e/fin-fout",
            dominance_predecessors(ProgramGraphShape::new(32, 31), "fin", "fout"),
        ),
        (
            "dominance_predecessors/33n32e/fin-fout",
            dominance_predecessors(ProgramGraphShape::new(33, 32), "fin", "fout"),
        ),
        (
            "dominance_predecessors/1024n4096e/renamed",
            dominance_predecessors(
                ProgramGraphShape::new(1024, 4096),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- reach then sink hit: flows_to_to_sink ----
        (
            "flows_to_to_sink/1n0e",
            flows_to_to_sink(
                ProgramGraphShape::new(1, 0),
                "source",
                "sink",
                "reach",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "flows_to_to_sink/4n3e",
            flows_to_to_sink(
                ProgramGraphShape::new(4, 3),
                "source",
                "sink",
                "reach",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "flows_to_to_sink/33n32e/renamed",
            flows_to_to_sink(
                ProgramGraphShape::new(33, 32),
                "src_set",
                "sink_set",
                "reach_scratch",
                "hit_scratch",
                "any_hit",
            ),
        ),
        // ---- reach then sink hit: taint_pollution ----
        (
            "taint_pollution/1n0e",
            taint_pollution(
                ProgramGraphShape::new(1, 0),
                "source",
                "sink",
                "reach",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "taint_pollution/4n3e",
            taint_pollution(
                ProgramGraphShape::new(4, 3),
                "source",
                "sink",
                "reach",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "taint_pollution/33n32e/renamed",
            taint_pollution(
                ProgramGraphShape::new(33, 32),
                "src_set",
                "label_set",
                "reach_scratch",
                "hit_scratch",
                "any_hit",
            ),
        ),
        // ---- sanitizer projection: flows_to_with_sanitizer ----
        (
            "flows_to_with_sanitizer/1n0e",
            flows_to_with_sanitizer(
                ProgramGraphShape::new(1, 0),
                "source",
                "sink",
                "sanitizer",
                "clean",
                "reach",
                "alive",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "flows_to_with_sanitizer/4n3e",
            flows_to_with_sanitizer(
                ProgramGraphShape::new(4, 3),
                "source",
                "sink",
                "sanitizer",
                "clean",
                "reach",
                "alive",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "flows_to_with_sanitizer/33n32e/renamed",
            flows_to_with_sanitizer(
                ProgramGraphShape::new(33, 32),
                "src_set",
                "sink_set",
                "san_set",
                "clean_scratch",
                "reach_scratch",
                "alive_scratch",
                "hit_scratch",
                "any_hit",
            ),
        ),
    ]
}

/// Pre-merge fingerprints, recorded against `b72b96dbc8`.
const PINS: &[(&str, &str)] = &[
    (
        "flows_to/1n0e/fin-fout",
        "d42c4ab9661960f3c95b278844ebc134bc4bee18770e9d52f38eb7e35c035778",
    ),
    (
        "flows_to/4n3e/fin-fout",
        "9aee3aea6309ef03a2b93ec3b7ab00c7026ac6c1088f5953e285b22ccf408353",
    ),
    (
        "flows_to/32n31e/fin-fout",
        "e554000a5afd08c2bbacdad58676d5a63b67981756d1a3d0d02be4f9002e3532",
    ),
    (
        "flows_to/33n32e/fin-fout",
        "c934beb96caa1e56a4d8fed89e6f59b27af5b6888f0aabae04b91fd5cf3de0a4",
    ),
    (
        "flows_to/1024n4096e/renamed",
        "068e6b83d08b01b2f72a8bbd16a3f29897c31250276d45cf551ea9a5c1da47e2",
    ),
    (
        "flows_to_alias_only/1n0e/fin-fout",
        "0538da3a9575a96fd136b50ede6352bf2b79dd0743092c2b2567e6a21337dd79",
    ),
    (
        "flows_to_alias_only/4n3e/fin-fout",
        "c8e9fd7630765026e1f56dbcd4422e2fce0f6891716e8ad6f5705a08f8d9af40",
    ),
    (
        "flows_to_alias_only/33n32e/renamed",
        "fa6601c723716d81b6541e2243affb4790fc4aa76a82ac66f34e91f2c0f7ad61",
    ),
    (
        "taint_flow/1n0e/fin-fout",
        "ee7319154a76bf6153d1fb669bcff2f3d87c5331b9061a763d9ec78d0f6c59ae",
    ),
    (
        "taint_flow/4n3e/fin-fout",
        "107e134fbe0466c7ca72c0fe7766eae3f0c5cff4c18a64a9e0a08032593bd0d0",
    ),
    (
        "taint_flow/32n31e/fin-fout",
        "75a8a1e7c723e4326b28d4f65486e63a22b6bd351c3aabb0851026c9872060be",
    ),
    (
        "taint_flow/33n32e/fin-fout",
        "a2888e35fa76d79828ec4301270681d03a2e6ce3aea08547be2617460fb45414",
    ),
    (
        "taint_flow/1024n4096e/renamed",
        "741a70c43a97040c5fe1f44db59cfc566dbe2ee7ce558fb4ec54d2d267b60883",
    ),
    (
        "bounded_by_comparison/1n0e/fin-fout",
        "f31a1a51136ab035c71155e04bd26028b309a9346913d951a770a45e8b513372",
    ),
    (
        "bounded_by_comparison/4n4e/fin-fout",
        "bef8f8f9ab15db03555140fb488db0f40760d31cc83fe90c8fd4632a5cc0256f",
    ),
    (
        "bounded_by_comparison/32n31e/fin-fout",
        "be861261378775d3339a04a9ba8df8536760159ef6fbec2efb9e8ecf8e2686f8",
    ),
    (
        "bounded_by_comparison/33n32e/fin-fout",
        "0ca65843af6aad494af2851f0bead6831034695d00b69e1132da7e9b163b394f",
    ),
    (
        "bounded_by_comparison/1024n4096e/renamed",
        "3f6cace2639cc091f43b932a4b2de29dd011d555561b53c7d59abe11d50fe8fa",
    ),
    (
        "dominance_predecessors/1n0e/fin-fout",
        "a23ff3f9e6162cf4ed598b1f5493aabeb3a6a85a8478fedf77b764686c3e4ade",
    ),
    (
        "dominance_predecessors/4n4e/fin-fout",
        "14595f955e249bb54042d12b424e08ba21e86f45bcec5cb8229cc3df5aa05052",
    ),
    (
        "dominance_predecessors/32n31e/fin-fout",
        "bcc043ffbcb388328743a26a0634546bd8532d8798ce0a273b6d635f00b0268b",
    ),
    (
        "dominance_predecessors/33n32e/fin-fout",
        "879ec2c6a1fe6684dbb1f7c4c6e29995676b34ec2f33777e9050bf8c2b62c0ef",
    ),
    (
        "dominance_predecessors/1024n4096e/renamed",
        "e1f13c229784b01f940d9c9c06f5178242cd08cf66144d2814445b59cd207c40",
    ),
    (
        "flows_to_to_sink/1n0e",
        "14fb76d30ec2eb89381d407383c2e66bb1a7d37434ce4c0b4e2d172d2d468051",
    ),
    (
        "flows_to_to_sink/4n3e",
        "f8fcaa19ecf1daa9a7eb03707734271aeae9790828eb6c3cc884370fa2c3f5a7",
    ),
    (
        "flows_to_to_sink/33n32e/renamed",
        "e45a81d6d5ba4df636612b41c5ecabef327634005417139ee426d8057ab41767",
    ),
    (
        "taint_pollution/1n0e",
        "6ab7b0ebecf2c2988c01263bb2a937f499c6a0e2ecf5f751fe146df0335fa359",
    ),
    (
        "taint_pollution/4n3e",
        "42ff277b94718aa1c69aac7b9e7f7b4c3678486030cddad8696772b57bd94c6b",
    ),
    (
        "taint_pollution/33n32e/renamed",
        "f4fc5ccdb81c9fe202f3e079cee20e1b9f349a85991f18becb59e8a1c0e586f8",
    ),
    (
        "flows_to_with_sanitizer/1n0e",
        "8504e5f50086acda06d7ab4a72ed370c6cba350141c707bdccc6545688a62763",
    ),
    (
        "flows_to_with_sanitizer/4n3e",
        "fc88511044cf98e01e73a453cd747da4fb74c8c4228d8705cfbd03b5f8b12431",
    ),
    (
        "flows_to_with_sanitizer/33n32e/renamed",
        "647c66e1c35293addababae97c202b7cc217098f35510724ffd0238a54e51a54",
    ),
];

#[test]
fn security_flow_family_entry_point_fingerprints_are_pinned() {
    let cases = pinned_cases();
    assert_eq!(
        cases.len(),
        PINS.len(),
        "every built case needs exactly one pin"
    );
    let mut drift = Vec::new();
    for ((label, program), (pin_label, expected)) in cases.iter().zip(PINS) {
        assert_eq!(label, pin_label, "pin table order must match case order");
        let actual = hex(program.fingerprint());
        if actual != *expected {
            drift.push(format!("    (\"{label}\", \"{actual}\"),"));
        }
    }
    assert!(
        drift.is_empty(),
        "security flow-skeleton family IR moved. Observed fingerprints:\n{}",
        drift.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Cross-entry-point behavior. Each rehomed file may own its predicate and
// nothing else, so the predicate is what these assertions pin.
// ---------------------------------------------------------------------------

/// A four-node graph carrying one edge of each interesting kind out of node 0:
///
/// ```text
///   0 --kinds[0]--> 1
///   0 --kinds[1]--> 2
///   0 --kinds[2]--> 3
/// ```
fn graph_bytes(kinds: [u32; 3]) -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("pg_nodes", vec![0, 0, 0, 0]),
        ("pg_edge_offsets", vec![0, 3, 3, 3, 3]),
        ("pg_edge_targets", vec![1, 2, 3]),
        ("pg_edge_kind_mask", kinds.to_vec()),
        ("pg_node_tags", vec![0, 0, 0, 0]),
    ]
}

/// Feed a program in its own declared buffer order, taking each buffer's
/// contents from `named` and zero-filling anything it does not name.
fn eval(program: &Program, named: &[(&str, Vec<u32>)]) -> Vec<Vec<u32>> {
    let values: Vec<Value> = program
        .buffers()
        .iter()
        .map(|decl| {
            let words = named
                .iter()
                .find(|(name, _)| *name == decl.name())
                .map_or_else(
                    || vec![0u32; decl.count() as usize],
                    |(_, words)| words.clone(),
                );
            Value::from(pack_u32_slice(&words))
        })
        .collect();
    vyre_reference::reference_eval(program, &values)
        .expect("Fix: security family guard program must evaluate")
        .iter()
        .map(|value| decode_u32_le_bytes_all(&value.to_bytes()))
        .collect()
}

fn reach_inputs(kinds: [u32; 3], frontier: u32) -> Vec<(&'static str, Vec<u32>)> {
    let mut inputs = graph_bytes(kinds);
    inputs.push(("fin", vec![frontier]));
    inputs.push(("fout", vec![frontier]));
    inputs
}

#[test]
fn flows_to_and_taint_flow_declare_the_same_forward_predicate() {
    let kinds = [
        edge_kind::ASSIGNMENT,
        edge_kind::CALL_ARG,
        edge_kind::CONTROL,
    ];
    let shape = ProgramGraphShape::new(4, 3);
    let inputs = reach_inputs(kinds, 0b0001);
    let left = eval(&flows_to(shape, "fin", "fout"), &inputs);
    let right = eval(&taint_flow(shape, "fin", "fout"), &inputs);
    assert_eq!(
        left, right,
        "flows_to and taint_flow are one skeleton under two op ids; \
         their reached sets must be identical"
    );
    // Dataflow reaches the ASSIGNMENT and CALL_ARG neighbours, never CONTROL.
    assert_eq!(left[0][0], 0b0111, "FLOWS_TO_MASK must exclude CONTROL");
}

#[test]
fn flows_to_alias_only_narrows_the_forward_predicate_to_aliases() {
    let kinds = [
        edge_kind::ASSIGNMENT,
        edge_kind::CALL_ARG,
        edge_kind::CONTROL,
    ];
    let shape = ProgramGraphShape::new(4, 3);
    let inputs = reach_inputs(kinds, 0b0001);
    let dataflow = eval(&flows_to(shape, "fin", "fout"), &inputs);
    let alias = eval(&flows_to_alias_only(shape, "fin", "fout"), &inputs);
    assert_eq!(
        alias[0][0], 0b0011,
        "ALIAS_PROPAGATION_MASK must drop the CALL_ARG neighbour \
         (the `char *copy = strdup(msg)` false positive)"
    );
    assert_ne!(
        dataflow, alias,
        "the alias predicate is a strict subset; a shared skeleton must not \
         collapse the two masks"
    );
}

fn hit_inputs(source: u32, sink: u32) -> Vec<(&'static str, Vec<u32>)> {
    let mut inputs = graph_bytes([
        edge_kind::ASSIGNMENT,
        edge_kind::CALL_ARG,
        edge_kind::CONTROL,
    ]);
    inputs.push(("source", vec![source]));
    inputs.push(("reach", vec![source]));
    inputs.push(("sink", vec![sink]));
    inputs.push(("hits", vec![0]));
    inputs.push(("out_scalar", vec![0]));
    inputs
}

#[test]
fn flows_to_to_sink_and_taint_pollution_declare_the_same_hit_predicate() {
    let shape = ProgramGraphShape::new(4, 3);
    for (source, sink) in [
        (0b0001u32, 0b0010u32), // one dataflow hop lands on the sink
        (0b0001, 0b1000),       // only a CONTROL neighbour is tagged: no hit
        (0b0000, 0b1111),       // empty source
        (0b0001, 0b0000),       // empty sink
        (0b1111, 0b1111),       // saturated
    ] {
        let inputs = hit_inputs(source, sink);
        let left = eval(
            &flows_to_to_sink(shape, "source", "sink", "reach", "hits", "out_scalar"),
            &inputs,
        );
        let right = eval(
            &taint_pollution(shape, "source", "sink", "reach", "hits", "out_scalar"),
            &inputs,
        );
        assert_eq!(
            left, right,
            "flows_to_to_sink and taint_pollution supply the same predicate; \
             source={source:#06b} sink={sink:#06b} must agree"
        );
    }
}

#[test]
fn flows_to_to_sink_reports_a_reached_sink_and_stays_quiet_otherwise() {
    let shape = ProgramGraphShape::new(4, 3);
    let program = flows_to_to_sink(shape, "source", "sink", "reach", "hits", "out_scalar");
    let hit = eval(&program, &hit_inputs(0b0001, 0b0010));
    assert_eq!(
        hit.last().expect("out_scalar")[0],
        1,
        "a dataflow neighbour tagged as a sink is a hit"
    );
    let miss = eval(&program, &hit_inputs(0b0001, 0b1000));
    assert_eq!(
        miss.last().expect("out_scalar")[0],
        0,
        "a CONTROL-only neighbour tagged as a sink is not a hit"
    );
}

#[test]
fn sanitizer_projection_removes_sanitized_nodes_from_the_hit() {
    let shape = ProgramGraphShape::new(4, 3);
    let program = flows_to_with_sanitizer(
        shape,
        "source",
        "sink",
        "sanitizer",
        "clean",
        "reach",
        "alive",
        "hits",
        "out_scalar",
    );
    let base = |sanitizer: u32| {
        let mut inputs = graph_bytes([
            edge_kind::ASSIGNMENT,
            edge_kind::CALL_ARG,
            edge_kind::CONTROL,
        ]);
        inputs.push(("source", vec![0b0001]));
        inputs.push(("sanitizer", vec![sanitizer]));
        inputs.push(("clean", vec![0]));
        inputs.push(("reach", vec![0]));
        inputs.push(("alive", vec![0]));
        inputs.push(("sink", vec![0b0010]));
        inputs.push(("hits", vec![0]));
        inputs.push(("out_scalar", vec![0]));
        inputs
    };
    assert_eq!(
        eval(&program, &base(0)).last().expect("out_scalar")[0],
        1,
        "with no sanitizer the projection degenerates to the plain hit"
    );
    assert_eq!(
        eval(&program, &base(0b0010)).last().expect("out_scalar")[0],
        0,
        "sanitizing the sink node kills the hit"
    );
    assert_eq!(
        eval(&program, &base(0b0001)).last().expect("out_scalar")[0],
        0,
        "sanitizing the source node kills the hit before the walk starts"
    );
}

#[test]
fn backward_pair_agrees_on_a_dominance_only_graph() {
    let shape = ProgramGraphShape::new(4, 3);
    let inputs = reach_inputs([edge_kind::DOMINANCE; 3], 0b1000);
    let bounded = eval(&bounded_by_comparison(shape, "fin", "fout"), &inputs);
    let predecessors = eval(&dominance_predecessors(shape, "fin", "fout"), &inputs);
    assert_eq!(
        bounded, predecessors,
        "on a DOMINANCE-only graph the two backward predicates coincide"
    );
    assert_eq!(
        bounded[0][0], 0b1001,
        "a backward dominance step from {{3}} keeps the seed and reaches {{0}}"
    );
}

#[test]
fn backward_pair_diverges_exactly_on_block_member_edges() {
    let shape = ProgramGraphShape::new(4, 3);
    // Only the edge into node 3 is BLOCK_MEMBER, which is in the
    // dominance_predecessors mask and not in the bounded_by_comparison mask.
    let kinds = [
        edge_kind::DOMINANCE,
        edge_kind::DOMINANCE,
        edge_kind::BLOCK_MEMBER,
    ];
    let inputs = reach_inputs(kinds, 0b1000);
    let bounded = eval(&bounded_by_comparison(shape, "fin", "fout"), &inputs);
    let predecessors = eval(&dominance_predecessors(shape, "fin", "fout"), &inputs);
    assert_eq!(
        bounded[0][0], 0b1000,
        "bounded_by_comparison walks DOMINANCE only, so a BLOCK_MEMBER edge \
         into {{3}} reaches nothing"
    );
    assert_eq!(
        predecessors[0][0], 0b1001,
        "dominance_predecessors also walks BLOCK_MEMBER, so it reaches {{0}}"
    );
    assert_ne!(
        bounded, predecessors,
        "the backward pair must keep two distinct edge masks after the rehome"
    );
}
