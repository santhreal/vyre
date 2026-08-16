// Tests for `validate.rs`. Split out per audit item #85 to keep the
// parent file focused on production code.

use super::*;
use crate::ir::{BufferDecl, DataType, Expr, Node, Program, UnOp};
use crate::validate::fusion_safety::validate_fusion_alias_hazards;
use crate::validate::self_composition::validate_self_composition;
use proptest::prelude::*;
use std::collections::BTreeSet;

// ------------------------------------------------------------------
// The multi-walk validator the single-walk `PreorderValidator` replaced,
// kept as the second arm of a differential property test. It shares the
// program-header rules with production through `validate_program_level`,
// because those are not what the property compares: the two arms differ in
// how they walk the node tree, and a program-header diagnostic corrected in
// one copy and not the other would fail the property for the wrong reason.
// ------------------------------------------------------------------
fn validate_with_options_legacy(
    program: &Program,
    options: ValidationOptions<'_>,
) -> ValidationReport {
    let (mut report, buffer_map) = validate_program_level(program);

    let mut scope = FxHashMap::default();
    let mut limits = depth::LimitState::default();
    crate::validate::legacy_walk::validate_nodes(
        program.entry(),
        &buffer_map,
        &mut scope,
        false,
        0,
        &mut limits,
        options,
        &mut report,
    );
    validate_fusion_alias_hazards(program.entry(), &mut report.errors);
    validate_self_composition(program.entry(), &mut report.errors);

    report
}

// ------------------------------------------------------------------
// IR-shape corpus. One owner: `visit::fixtures`, the module
// that owns the public visitor these programs are walked with.
// ------------------------------------------------------------------
use crate::visit::{fixtures::arb_program, walk_nodes_and_exprs, ExprSink, NodeSink};

/// Every buffer name the PUBLIC visitor reaches, driving
/// [`walk_nodes_and_exprs`] directly. `referenced_buffers` is deliberately
/// not used here: it answers from the cached `ProgramFacts` SoA walk, so it
/// would compare the validator against that walk rather than against the
/// visitor this test exists to pin.
#[derive(Default)]
struct BufferNamesReached(BTreeSet<String>);

impl NodeSink for BufferNamesReached {
    fn accept_node(&mut self, node: &Node) {
        let mut record = |name: &crate::ir::Ident| {
            self.0.insert(name.as_str().to_string());
        };
        match node {
            Node::Store { buffer, .. }
            | Node::AllReduce { buffer, .. }
            | Node::Broadcast { buffer, .. }
            | Node::TileLoad { buffer, .. }
            | Node::TileStore { buffer, .. } => record(buffer),
            Node::IndirectDispatch { count_buffer, .. } => record(count_buffer),
            Node::AllGather {
                input: source,
                output: destination,
                ..
            }
            | Node::ReduceScatter {
                input: source,
                output: destination,
                ..
            } => {
                record(source);
                record(destination);
            }
            // An async transfer's `source` and `destination` name storage tiers,
            // not entries in the dispatch's buffer table, so the validator does
            // not resolve them and neither does this comparison.
            // `extension_adversarial::async_extension_tags_remain_structural`
            // pins that. Their `offset` and `size` operands are ordinary
            // expressions and are reached through `ExprVisitor` below.
            Node::AsyncLoad { .. } | Node::AsyncStore { .. } => {}
            _ => {}
        }
    }
}

impl ExprSink for BufferNamesReached {
    fn accept_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Load { buffer, .. }
            | Expr::BufLen { buffer }
            | Expr::BufferRef { buffer }
            | Expr::Atomic { buffer, .. } => {
                self.0.insert(buffer.as_str().to_string());
            }
            _ => {}
        }
    }
}

/// Pull `NAME` out of a diagnostic of the form ``... unknown buffer `NAME` ...``.
fn unknown_buffer_name(message: &str) -> Option<String> {
    let tail = message.split("unknown buffer `").nth(1)?;
    tail.split('`').next().map(str::to_string)
}

// ------------------------------------------------------------------
// Cross-walk guard for the merged corpus: the validator's traversal and
// the public visitor must reach the same buffer references. Declaring no
// buffers turns every reference the validator resolves into an
// `unknown buffer` diagnostic, which makes the two walks directly
// comparable. A node or expression kind that one walk descends into and
// the other skips shows up here as a set difference.
//
// The case count is raised because the corpus now reaches every `Node` variant:
// at 128 cases over 8 variants no async transfer, collective, trap or region was
// ever generated, so the walks were only ever compared on stores, loops and ifs.
// ------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        ..ProptestConfig::default()
    })]

    #[test]
    fn validator_walk_reaches_the_same_buffers_as_the_public_visitor(program in arb_program()) {
        let probe = program.with_rewritten_buffers(Vec::new());

        let mut by_visitor = BufferNamesReached::default();
        walk_nodes_and_exprs(&probe, &mut by_visitor);

        let report = validate_with_options(&probe, ValidationOptions::default());
        let by_validator: BTreeSet<String> = report
            .errors
            .iter()
            .filter_map(|issue| unknown_buffer_name(&issue.message()))
            .collect();

        prop_assert_eq!(by_visitor.0, by_validator);
    }
}

// ------------------------------------------------------------------
// Regression test: the single-pass validator must emit exactly the same
// errors (+ warnings) as the recursive walk it replaced.
//
// The corpus reaches every `Node` variant, so the shapes this compares now
// include async transfers, collectives, traps, opaque extensions and regions.
// The case count is raised to match: at 50 cases over 8 variants the async
// alias-operand divergence and the region-scope divergence were both
// unreachable.
// ------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        ..ProptestConfig::default()
    })]

    #[test]
    fn single_pass_validator_matches_legacy(program in arb_program()) {
        let legacy = validate_with_options_legacy(&program, ValidationOptions::default());
        let modern = validate_with_options(&program, ValidationOptions::default());

        // Deterministic ordering: sort both error sets by message.
        let mut legacy_errors = legacy.errors;
        let mut modern_errors = modern.errors;
        legacy_errors.sort_by(|a, b| a.message().cmp(&b.message()));
        modern_errors.sort_by(|a, b| a.message().cmp(&b.message()));

        for issue in &mut legacy_errors {
            issue.set_location(ValidationLocation::Program);
        }
        for issue in &mut modern_errors {
            issue.set_location(ValidationLocation::Program);
        }

        prop_assert_eq!(
            legacy_errors, modern_errors,
            "error mismatch between legacy and single-pass validator"
        );

        let mut legacy_warnings = legacy.warnings;
        let mut modern_warnings = modern.warnings;
        legacy_warnings.sort_by(|a, b| a.message.cmp(&b.message));
        modern_warnings.sort_by(|a, b| a.message.cmp(&b.message));

        prop_assert_eq!(
            legacy_warnings, modern_warnings,
            "warning mismatch between legacy and single-pass validator"
        );
    }
}

// ------------------------------------------------------------------
// F2 regression: let-binding a call result must not fabricate a U32
// type and fire false V045 on later assignments of a different type.
// ------------------------------------------------------------------

/// When `expr_type` returns `None` for `Expr::Call` (no dialect lookup provided),
/// `visit_let` previously recorded `DataType::U32` as the binding type. A later
/// assignment of `1.0f32` would then trigger a false V045 ("U32 expected, got F32").
///
/// After the fix the binding is recorded as `ty_known = false`, and V045 is
/// skipped (the program must validate without V045).
#[test]
fn call_result_binding_unknown_type_does_not_produce_false_v045() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
        [1, 1, 1],
        vec![
            // let x = some_call() (type is unknown because no lookup is registered).
            Node::Let {
                name: "x".into(),
                value: Expr::Call {
                    op_id: "unknown.dialect.op".into(),
                    args: vec![],
                },
            },
            // assign x = 1.0f32, valid if x is F32, but previously caused a false
            // V045 because x was recorded as U32 (the fabricated sentinel).
            Node::Assign {
                name: "x".into(),
                value: Expr::LitF32(1.0),
            },
            Node::Store {
                buffer: "out".into(),
                index: Expr::u32(0),
                value: Expr::var("x"),
            },
        ],
    );
    let report = validate_with_options(&program, ValidationOptions::default());
    // The only error must be V016 (no lookup for the call), NOT V045.
    let v045: Vec<_> = report
        .errors
        .iter()
        .filter(|e| e.code().as_str() == "V045")
        .collect();
    assert!(
        v045.is_empty(),
        "false V045 fired on call-result binding with unknown type: {:?}",
        v045
    );
    // Confirm that V016 IS emitted (the call itself is still rejected).
    assert!(
        report.errors.iter().any(|e| e.code().as_str() == "V016"),
        "expected V016 for call with no lookup, got: {:?}",
        report.errors
    );
}

// ------------------------------------------------------------------
// `fma_f32_violations`: the focused subset emit backends run before
// lowering. Pins the `V028` filter so a message change cannot silently
// disable the integer-Fma rejection (which would re-open the Law-10
// silent `a*b+c` miscompile), and proves the filter excludes unrelated
// validation errors so emit boundaries don't preempt downstream
// diagnostics.
// ------------------------------------------------------------------
#[test]
fn fma_f32_violations_flags_integer_fma_with_actionable_message() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::let_bind(
            "bad_fma",
            Expr::Fma {
                a: Box::new(Expr::u32(1)),
                b: Box::new(Expr::u32(2)),
                c: Box::new(Expr::u32(3)),
            },
        )],
    );
    let violations = fma_f32_violations(&program);
    assert_eq!(
        violations.len(),
        3,
        "every non-f32 Fma operand (a, b, c) must be reported, got: {violations:?}"
    );
    for violation in &violations {
        assert!(
            violation.code().as_str() == "V028",
            "fma_f32_violations must only return V028 errors, got: {}",
            violation.message()
        );
        assert!(
            violation
                .message()
                .contains("Fma requires three f32 operands")
                && violation.message().contains("must be `f32`")
                && violation.message().contains("Fix:"),
            "V028 message must name the f32 contract and a fix, got: {}",
            violation.message()
        );
    }
}

#[test]
fn fma_f32_violations_empty_for_all_f32_operands() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::Fma {
                a: Box::new(Expr::LitF32(2.0)),
                b: Box::new(Expr::LitF32(3.0)),
                c: Box::new(Expr::LitF32(4.0)),
            },
        )],
    );
    assert!(
        fma_f32_violations(&program).is_empty(),
        "f32 Fma is valid and must not be flagged"
    );
}

#[test]
fn fma_f32_violations_ignores_unrelated_validation_errors() {
    // A program with NO Fma but a genuine validation error (zero workgroup
    // dimension). `validate` reports it; `fma_f32_violations` must NOT, so
    // emit boundaries calling this never preempt the dedicated diagnostic.
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)],
        [0, 1, 1],
        Vec::new(),
    );
    assert!(
        !validate(&program).is_empty(),
        "zero workgroup dimension must be a validation error (guards the test premise)"
    );
    assert!(
        fma_f32_violations(&program).is_empty(),
        "non-Fma validation errors must be filtered out by fma_f32_violations"
    );
}

// ------------------------------------------------------------------
// Unpack UnOp recognition. `Unpack4/8 Low/High` previously fell through
// `validate_unop_operand`'s `_` catch-all and were rejected as "not
// recognized", even though that message lists them as valid and every
// backend lowers them. Validate must recognize them and check the
// integer-word operand contract instead.
// ------------------------------------------------------------------
#[test]
fn validate_recognizes_integer_unpack_ops() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::UnOp {
                op: UnOp::Unpack8High,
                operand: Box::new(Expr::u32(0xDEAD_BEEF)),
            },
        )],
    );
    let errors = validate(&program);
    assert!(
        !errors
            .iter()
            .any(|e| e.message().contains("is not recognized")),
        "integer unpack op must be recognized, got: {errors:?}"
    );
    assert!(
        !errors
            .iter()
            .any(|e| e.message().contains("unpack ops require")),
        "a u32 operand is valid for unpack ops, got: {errors:?}"
    );
}

#[test]
fn validate_rejects_non_integer_unpack_operand_on_type_not_existence() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::UnOp {
                op: UnOp::Unpack4Low,
                operand: Box::new(Expr::LitF32(1.5)),
            },
        )],
    );
    let errors = validate(&program);
    assert!(
        errors.iter().any(
            |e| e.message().contains("unpack ops require a 32-bit integer")
                && e.message().contains("Fix:")
        ),
        "f32 unpack operand must be rejected with the integer-word contract, got: {errors:?}"
    );
    assert!(
        !errors
            .iter()
            .any(|e| e.message().contains("is not recognized")),
        "unpack op must be rejected on operand type, not treated as unrecognized, got: {errors:?}"
    );
}

// ------------------------------------------------------------------
// Same-width integer store coercion (U32 <-> I32, U64 <-> I64).
//
// The typechecker types Mod / bitwise / shift results as U32 regardless of
// operand signedness (typecheck.rs `_ => DataType::U32`), while Add/Sub/Mul/Div
// preserve the operand type via Frame::Bin. A buffer element only distinguishes
// signedness when LOADED (sign- vs zero-extend on use); a STORE writes the raw
// 32/64-bit word, so storing a U32-typed value into an I32 buffer (or vice
// versa) is a bit-exact reinterpret. A shader emitter already coerces the store
// value to the element type, a machine-code emitter stores a typeless word, and
// the reference oracle stores the value's
// bytes, so every lower layer is byte-correct. The validator was the lone
// over-strict layer: it rejected `store(i32_buffer, rem(i32, i32))` (a valid,
// common signed-remainder store) with V045. These pin the coercion.
// ------------------------------------------------------------------
#[test]
fn store_signed_remainder_into_i32_buffer_validates() {
    // rem(i32, i32) is U32-typed but carries the SIGNED remainder bits; storing
    // it into an I32 buffer is a same-width reinterpret. Before the coercion this
    // was wrongly rejected V045 (the documented Mod-result-type gap).
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::I32).with_count(4),
            BufferDecl::read("a", 1, DataType::I32).with_count(4),
            BufferDecl::read("b", 2, DataType::I32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::rem(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
        )],
    );
    let errors = validate(&program);
    assert!(
        !errors
            .iter()
            .any(|e| e.code().as_str() == "V045" || e.message().contains("value has type")),
        "store of a same-width int (rem result, u32-typed) into an i32 buffer must \
         validate (bit-exact reinterpret), got: {errors:?}"
    );
}

#[test]
fn store_signed_div_into_u32_buffer_validates() {
    // The reverse direction: div(i32, i32) is I32-typed (Frame::Bin preserves the
    // operand type); storing it into a U32 buffer is the same same-width
    // reinterpret and must also validate.
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(4),
            BufferDecl::read("a", 1, DataType::I32).with_count(4),
            BufferDecl::read("b", 2, DataType::I32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::div(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
        )],
    );
    let errors = validate(&program);
    assert!(
        !errors
            .iter()
            .any(|e| e.code().as_str() == "V045" || e.message().contains("value has type")),
        "store of an i32-typed value into a u32 buffer must validate, got: {errors:?}"
    );
}

#[test]
fn store_float_into_int_buffer_still_rejected() {
    // The coercion is ONLY same-width INTEGER reinterpret. A float value into an
    // i32 buffer is a real type error (different bit semantics, needs an explicit
    // cast) and must STILL be rejected (proving the coercion did not over-broaden).
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::I32).with_count(4),
            BufferDecl::read("f", 1, DataType::F32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("f", Expr::u32(0)),
        )],
    );
    let errors = validate(&program);
    assert!(
        errors
            .iter()
            .any(|e| e.message().contains("Node::Store") && e.message().contains("element type")),
        "storing an f32 value into an i32 buffer must still be rejected (no int/float \
         coercion), got: {errors:?}"
    );
}

#[test]
fn assign_signed_remainder_to_i32_buffer_validates() {
    // The buffer-ASSIGN path (visit_assign) must apply the same same-width int
    // reinterpret coercion as Node::Store, otherwise `buf = rem(i32, i32)` is
    // rejected while the equivalent store is allowed (an inconsistency between
    // two writes of the same logical value).
    let program = Program::wrapped(
        vec![
            BufferDecl::read_write("buf", 0, DataType::I32).with_count(4),
            BufferDecl::read("a", 1, DataType::I32).with_count(4),
            BufferDecl::read("b", 2, DataType::I32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::assign(
            "buf",
            Expr::rem(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
        )],
    );
    let errors = validate(&program);
    assert!(
        !errors.iter().any(|e| e.code().as_str() == "V045"),
        "assigning a same-width int (rem result) to an i32 buffer must validate, got: {errors:?}"
    );
}

#[test]
fn store_bool_comparison_result_into_u32_buffer_validates() {
    // A comparison produces Bool; storing the 0/1 flag into a U32 output buffer
    // is a common, sound pattern (the emitter coerces Bool -> u32). The buffer
    // ASSIGN path always allowed it, but Node::Store rejected it until the compat
    // rule was unified into one `store_value_compatible` predicate.
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(4),
            BufferDecl::read("a", 1, DataType::I32).with_count(4),
            BufferDecl::read("b", 2, DataType::I32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::lt(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
        )],
    );
    let errors = validate(&program);
    assert!(
        !errors
            .iter()
            .any(|e| e.code().as_str() == "V045" || e.message().contains("value has type")),
        "storing a bool comparison result into a u32 buffer must validate, got: {errors:?}"
    );
}

// ------------------------------------------------------------------
// `V116` had two implementations that disagreed, and production was the
// under-reporting one.
//
// The flat whole-program pass in `validate::fusion_safety` recorded a transfer's
// `offset` and `size` operands as buffer accesses; the frame-scoped copy threaded
// through `PreorderValidator`'s stack machine recorded only `source` and
// `destination`. An atomic in a transfer size was therefore a hazard to one
// implementation and invisible to the other, and the one every caller of
// `validate` reached was the blind one.
//
// The measured disagreement is the async operand, not, as the frame machinery
// suggested, an atomic and a read in different frames: both implementations
// scoped alias state to one node sequence, so a nested body started with empty
// state in each and neither reported across a `Block`, an `If` branch or a
// `Loop` body.
// ------------------------------------------------------------------

fn alias_hazard_program(entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32)
                .with_count(8)
                .with_output_byte_range(0..16),
            BufferDecl::read("input", 1, DataType::U32).with_count(8),
            BufferDecl::read_write("rw", 2, DataType::U32).with_count(8),
            BufferDecl::read_write("other", 3, DataType::U32).with_count(8),
        ],
        [1, 1, 1],
        entry,
    )
}

fn atomic_add(buffer: &str) -> Expr {
    Expr::Atomic {
        op: crate::ir::AtomicOp::Add,
        buffer: buffer.into(),
        index: Box::new(Expr::u32(0)),
        expected: None,
        value: Box::new(Expr::u32(1)),
        ordering: crate::memory_model::MemoryOrdering::SeqCst,
    }
}

fn async_store_with_size(size: Expr) -> Node {
    Node::AsyncStore {
        source: "input".into(),
        destination: "out".into(),
        offset: Box::new(Expr::u32(0)),
        size: Box::new(size),
        tag: "stream0".into(),
    }
}

fn hazard_buffers(program: &Program) -> Vec<String> {
    validate(program)
        .iter()
        .filter(|error| error.code().as_str() == "V116")
        .filter_map(|error| {
            error
                .message()
                .split("fusion hazard on buffer `")
                .nth(1)?
                .split('`')
                .next()
                .map(str::to_string)
        })
        .collect()
}

#[test]
fn atomic_in_an_async_transfer_size_is_a_fusion_hazard() {
    let program = alias_hazard_program(vec![
        Node::let_bind("snapshot", Expr::load("rw", Expr::u32(0))),
        async_store_with_size(atomic_add("rw")),
    ]);

    assert_eq!(
        hazard_buffers(&program),
        vec!["rw".to_string()],
        "an atomic on `rw` in a transfer size must be a V116 hazard against the plain read \
         of `rw` before it: {:?}",
        validate(&program)
    );
}

#[test]
fn atomic_in_an_async_transfer_offset_is_a_fusion_hazard() {
    let program = alias_hazard_program(vec![
        Node::let_bind("snapshot", Expr::load("rw", Expr::u32(0))),
        Node::AsyncLoad {
            source: "input".into(),
            destination: "out".into(),
            offset: Box::new(atomic_add("rw")),
            size: Box::new(Expr::u32(1)),
            tag: "stream0".into(),
        },
    ]);

    assert_eq!(
        hazard_buffers(&program),
        vec!["rw".to_string()],
        "the offset operand carries the same hazard as the size operand: {:?}",
        validate(&program)
    );
}

#[test]
fn a_barrier_clears_an_async_transfer_operand_hazard() {
    // Negative control for the two tests above: the rule is "without an explicit
    // barrier", so the same program with a barrier between the read and the
    // transfer must be accepted. A V116 that fired on every transfer would pass
    // them both.
    let program = alias_hazard_program(vec![
        Node::let_bind("snapshot", Expr::load("rw", Expr::u32(0))),
        Node::barrier(),
        async_store_with_size(atomic_add("rw")),
    ]);

    assert!(
        hazard_buffers(&program).is_empty(),
        "a barrier between the read path and the atomic path clears the hazard: {:?}",
        validate(&program)
    );
}

#[test]
fn an_async_transfer_operand_atomic_on_another_buffer_is_not_a_hazard() {
    // Second negative control: the hazard is per buffer. An atomic on `other`
    // must not be reported against a read of `rw`.
    let program = alias_hazard_program(vec![
        Node::let_bind("snapshot", Expr::load("rw", Expr::u32(0))),
        async_store_with_size(atomic_add("other")),
    ]);

    assert!(
        hazard_buffers(&program).is_empty(),
        "an atomic on a different buffer is not a hazard against `rw`: {:?}",
        validate(&program)
    );
}

// ------------------------------------------------------------------
// `Node::Region` scopes its body, and the walk that replaced the recursive
// one must agree with it.
//
// The recursive walk passed no scope log for a region body, so a `let` inside a
// region leaked into the enclosing scope and stayed live for the rest of the
// program. `region_inline` flattens a small Region into its parent and re-wraps
// it in a `Node::Block` when a name would collide, which is only sound while a
// region body's bindings are scoped, so the leak was the defect.
// ------------------------------------------------------------------

fn region(generator: &str, body: Vec<Node>) -> Node {
    Node::Region {
        generator: generator.into(),
        source_region: None,
        body: std::sync::Arc::new(body),
    }
}

fn unknown_variable_errors(program: &Program) -> Vec<String> {
    validate(program)
        .iter()
        .filter(|error| error.code().as_str() == "V066")
        .map(|error| error.message().to_string())
        .collect()
}

#[test]
fn a_let_inside_a_region_does_not_outlive_the_region() {
    let program = alias_hazard_program(vec![
        region(
            "stage.accumulate",
            vec![Node::let_bind("acc", Expr::u32(1))],
        ),
        region(
            "stage.write",
            vec![Node::store("out", Expr::u32(0), Expr::var("acc"))],
        ),
    ]);

    assert!(
        unknown_variable_errors(&program)
            .iter()
            .any(|message| message.contains("acc")),
        "a region scopes its body, so `acc` is unbound in the next region: {:?}",
        validate(&program)
    );
}

#[test]
fn a_let_inside_a_block_does_not_outlive_the_block() {
    // `Block` and `Region` carry the same binding lifetime, so the same shape
    // must be rejected through both. A rule that held for only one of them would
    // let `region_inline` swap a Region for a Block and change scoping.
    let program = alias_hazard_program(vec![
        Node::Block(vec![Node::let_bind("acc", Expr::u32(1))]),
        Node::store("out", Expr::u32(0), Expr::var("acc")),
    ]);

    assert!(
        unknown_variable_errors(&program)
            .iter()
            .any(|message| message.contains("acc")),
        "a `Block` scopes its bindings: {:?}",
        validate(&program)
    );
}

#[test]
fn a_let_inside_a_region_does_not_outlive_the_scope_holding_the_region() {
    // The recursive walk's leak was unbounded, not one level: a binding declared
    // in a region nested inside an `If` survived the `If` too.
    let program = alias_hazard_program(vec![
        Node::If {
            cond: Expr::LitBool(true),
            then: vec![region(
                "stage.accumulate",
                vec![Node::let_bind("acc", Expr::u32(1))],
            )],
            otherwise: Vec::new(),
        },
        Node::store("out", Expr::u32(0), Expr::var("acc")),
    ]);

    assert!(
        unknown_variable_errors(&program)
            .iter()
            .any(|message| message.contains("acc")),
        "neither the region nor the `If` around it lets `acc` escape: {:?}",
        validate(&program)
    );
}

#[test]
fn sibling_regions_may_each_bind_the_same_name() {
    // Negative control on the scope rule: restoring the region's scope must not
    // turn two composed child regions that each declare `acc` into a sibling
    // duplicate.
    let program = alias_hazard_program(vec![
        region("stage.one", vec![Node::let_bind("acc", Expr::u32(1))]),
        region("stage.two", vec![Node::let_bind("acc", Expr::u32(2))]),
    ]);

    let errors = validate(&program);
    assert!(
        !errors.iter().any(|error| error.code().as_str() == "V032"),
        "two child regions binding the same name are not sibling duplicates: {errors:?}"
    );
}

// ------------------------------------------------------------------
// An async transfer's endpoints are storage-tier tags, not buffer references.
//
// `AsyncLoad`/`AsyncStore` name the tier a transfer moves bytes between, which
// is outside the dispatch's buffer table, so neither endpoint resolves against
// it. The operands that ARE program expressions, `offset` and `size`, do
// resolve, and validating them is what surfaced the `V116` under-report.
// ------------------------------------------------------------------

#[test]
fn an_async_transfer_endpoint_is_not_resolved_against_the_buffer_table() {
    let program = alias_hazard_program(vec![Node::AsyncLoad {
        source: "ssd".into(),
        destination: "vram".into(),
        offset: Box::new(Expr::u32(0)),
        size: Box::new(Expr::u32(1)),
        tag: "stream0".into(),
    }]);

    assert!(
        unknown_variable_errors(&program).is_empty(),
        "transfer endpoints are tier tags, not buffers: {:?}",
        validate(&program)
    );
    assert!(
        validate(&program).is_empty(),
        "a transfer between two tiers carries no diagnostic at all: {:?}",
        validate(&program)
    );
}

#[test]
fn an_async_transfer_offset_still_resolves_its_buffer_loads() {
    // The endpoints being opaque must not make the whole node opaque: a load
    // from an undeclared buffer inside `offset` is a program expression and is
    // rejected exactly as it would be in a store index.
    let program = alias_hazard_program(vec![Node::AsyncLoad {
        source: "ssd".into(),
        destination: "vram".into(),
        offset: Box::new(Expr::load("typo", Expr::u32(0))),
        size: Box::new(Expr::u32(1)),
        tag: "stream0".into(),
    }]);

    let errors = validate(&program);
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("unknown buffer `typo`")),
        "a load inside a transfer offset resolves like any other load: {errors:?}"
    );
}
