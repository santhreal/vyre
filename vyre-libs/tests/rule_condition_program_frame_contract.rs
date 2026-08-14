//! Frame contract for the shared rule-condition program builder.
//!
//! WHY: every scalar rule leaf in `vyre_libs::rule` (the six file-size
//! predicates, the two pattern-count predicates, the two literals, and the
//! pattern-existence check) is one call to `condition_program` with a different
//! `compute` closure. The builder is the choke point for all of them, it is not
//! registered in the operation inventory, and no test named it, so the binding
//! order, the argument slot each accessor reads, the region identity, and the
//! fact that `compute()` is what actually lands in `out` were all unpinned. A
//! swapped binding or a dropped region wrapper would have changed every rule
//! leaf at once with nothing red.
//!
//! Testing the choke point rather than each leaf is deliberate: a new leaf
//! added tomorrow inherits the frame, and the argument-slot cases below fail if
//! any accessor is repointed.
//!
//! What this does not catch: a leaf that stops calling `condition_program` and
//! hand-rolls an equivalent-looking program of its own.

#![forbid(unsafe_code)]

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_libs::rule::builder::RULE_SET_OP_ID;
use vyre_libs::rule::condition_op::{
    condition_program, file_size, pattern_count, pattern_state, threshold, WORKGROUP_SIZE,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

/// Argument buffers in binding order, as the six-slot condition ABI declares
/// them. `out` is binding six.
const ARGUMENTS: [&str; 6] = [
    "rule_id",
    "pattern_id",
    "pattern_state",
    "pattern_count",
    "file_size",
    "threshold",
];

const TEST_OP_ID: &str = "vyre-libs::rule::frame_contract_probe";

/// Evaluate a condition program over the six argument slots and return the
/// verdict word it publishes.
fn verdict(program: &Program, arguments: [u32; 6]) -> u32 {
    let inputs: Vec<Value> = arguments
        .iter()
        .map(|word| Value::from(pack_u32_slice(&[*word])))
        .collect();
    let outputs = vyre_reference::reference_eval(program, &inputs)
        .expect("Fix: a condition program must execute on the reference interpreter");
    assert_eq!(outputs.len(), 1, "a condition leaf publishes one verdict");
    decode_u32_le_bytes_all(&outputs[0].to_bytes())[0]
}

/// The six argument slots and the verdict slot are declared in binding order,
/// the verdict carries a static element count, and the program runs one lane.
/// Goes red if a binding is renumbered, a slot is dropped, or the leaf stops
/// being a single-lane scalar op.
///
/// The count is not decoration. A backend-allocated output with no static
/// element count and no output byte range fails IR validation (V130), which
/// makes every rule leaf unrunnable rather than merely unpinned, so the frame
/// has to declare it here where all eleven leaves inherit it.
#[test]
fn the_condition_frame_declares_six_arguments_and_one_verdict() {
    let program = condition_program(TEST_OP_ID, || Expr::u32(1));
    let declared: Vec<(&str, u32)> = program
        .buffers()
        .iter()
        .map(|decl| (decl.name(), decl.binding))
        .collect();
    let mut expected: Vec<(&str, u32)> = ARGUMENTS
        .iter()
        .enumerate()
        .map(|(binding, name)| (*name, binding as u32))
        .collect();
    expected.push(("out", 6));
    assert_eq!(declared, expected);
    assert_eq!(program.workgroup_size(), WORKGROUP_SIZE);
    assert_eq!(program.workgroup_size(), [1, 1, 1]);

    let verdict_slot = program.buffers().last().expect("the frame declares `out`");
    assert!(
        verdict_slot.is_backend_allocated_output(),
        "the verdict is the program's output, so the backend allocates it"
    );
    assert_eq!(
        verdict_slot.count, 1,
        "the verdict output must declare its one element, or the program \
         cannot pass IR validation"
    );
}

/// The body is one region carrying the caller's operation id, and the region
/// holds exactly one store of the computed expression into `out[0]`. Goes red
/// if the region wrapper is dropped (which is what makes a leaf an atomic
/// compile unit for the optimizer) or if the op id stops reaching the region.
#[test]
fn the_condition_frame_wraps_one_store_in_a_region_named_by_the_op_id() {
    let program = condition_program(TEST_OP_ID, || Expr::u32(1));
    let Some(Node::Region {
        generator, body, ..
    }) = program.entry().first()
    else {
        panic!("Fix: a condition program must be one wrapping region");
    };
    assert_eq!(program.entry().len(), 1);
    assert_eq!(generator.as_str(), TEST_OP_ID);
    assert_eq!(body.len(), 1);
    assert_eq!(
        body[0],
        Node::store("out", Expr::u32(0), Expr::u32(1)),
        "the region body is exactly the computed verdict store"
    );
}

/// `compute()` is the published verdict, not a placeholder. Both polarities are
/// checked so an inverted or constant-folded lowering cannot pass.
#[test]
fn the_computed_expression_is_what_the_leaf_publishes() {
    let program = condition_program(TEST_OP_ID, || Expr::lt(file_size(), threshold()));
    assert_eq!(verdict(&program, [0, 0, 0, 0, 1024, 2048]), 1);
    assert_eq!(verdict(&program, [0, 0, 0, 0, 4096, 2048]), 0);
    assert_eq!(
        verdict(&program, [0, 0, 0, 0, 2048, 2048]),
        0,
        "a strict predicate is false at the boundary"
    );

    let always = condition_program(TEST_OP_ID, || Expr::u32(1));
    assert_eq!(verdict(&always, [0, 0, 0, 0, 0, 0]), 1);
    let never = condition_program(TEST_OP_ID, || Expr::u32(0));
    assert_eq!(verdict(&never, [7, 7, 7, 7, 7, 7]), 0);
}

/// Each argument accessor reads its own slot. The probe holds every slot at
/// zero except the one under test, so an accessor pointed at a neighbouring
/// binding produces the wrong verdict. Goes red on any slot transposition.
#[test]
fn each_argument_accessor_reads_its_own_slot() {
    let by_state = condition_program(TEST_OP_ID, || Expr::ne(pattern_state(), Expr::u32(0)));
    assert_eq!(verdict(&by_state, [9, 9, 1, 0, 0, 0]), 1);
    assert_eq!(verdict(&by_state, [9, 9, 0, 9, 9, 9]), 0);

    let by_count = condition_program(TEST_OP_ID, || Expr::ge(pattern_count(), threshold()));
    assert_eq!(verdict(&by_count, [0, 0, 0, 5, 0, 3]), 1);
    assert_eq!(verdict(&by_count, [0, 0, 0, 2, 0, 3]), 0);

    let by_size = condition_program(TEST_OP_ID, || Expr::gt(file_size(), threshold()));
    assert_eq!(verdict(&by_size, [0, 0, 0, 0, 4, 3]), 1);
    assert_eq!(verdict(&by_size, [0, 0, 0, 4, 3, 3]), 0);
}

/// Every rule leaf declared in the tree, paired with the program it builds.
///
/// The pairing has to be written out, because the leaves are macro-expanded
/// modules with no run-time iterator. What is derived instead is the
/// completeness of the pairing: see
/// `the_leaf_table_covers_every_rule_operation_declared_in_the_tree`.
fn leaves() -> Vec<(&'static str, Program)> {
    use vyre_libs::rule::{
        file_size_eq, file_size_gt, file_size_gte, file_size_lt, file_size_lte, file_size_ne,
        literal_false, literal_true, pattern_count_gt, pattern_count_gte, pattern_exists,
    };

    vec![
        (file_size_eq::OP_ID, file_size_eq::FileSizeEq::program()),
        (file_size_gt::OP_ID, file_size_gt::FileSizeGt::program()),
        (file_size_gte::OP_ID, file_size_gte::FileSizeGte::program()),
        (file_size_lt::OP_ID, file_size_lt::FileSizeLt::program()),
        (file_size_lte::OP_ID, file_size_lte::FileSizeLte::program()),
        (file_size_ne::OP_ID, file_size_ne::FileSizeNe::program()),
        (
            pattern_count_gt::OP_ID,
            pattern_count_gt::PatternCountGt::program(),
        ),
        (
            pattern_count_gte::OP_ID,
            pattern_count_gte::PatternCountGte::program(),
        ),
        (literal_true::OP_ID, literal_true::LiteralTrue::program()),
        (literal_false::OP_ID, literal_false::LiteralFalse::program()),
        (
            pattern_exists::OP_ID,
            pattern_exists::PatternExists::program(),
        ),
    ]
}

/// The generated leaf modules are the frame plus their own predicate: same
/// buffer layout, same workgroup, their own op id on the region, and each one
/// actually executable. Covering the whole macro family, not one
/// representative, is the point.
#[test]
fn every_generated_leaf_carries_the_shared_frame_under_its_own_op_id() {
    let reference = condition_program(TEST_OP_ID, || Expr::u32(1));
    let leaves = leaves();

    for (op_id, program) in &leaves {
        assert_eq!(
            program.buffers(),
            reference.buffers(),
            "{op_id} must declare the shared condition frame"
        );
        assert_eq!(program.workgroup_size(), WORKGROUP_SIZE, "{op_id}");
        let Some(Node::Region {
            generator, body, ..
        }) = program.entry().first()
        else {
            panic!("Fix: {op_id} must be one wrapping region");
        };
        assert_eq!(generator.as_str(), *op_id);
        assert_eq!(body.len(), 1, "{op_id} publishes exactly one verdict");

        // Running it is what proves the frame is valid IR and not merely
        // well shaped: an output slot with no declared element count builds
        // fine and is refused at validation.
        let published = verdict(program, [0, 0, 0, 0, 0, 0]);
        assert!(
            published <= 1,
            "{op_id} published {published}, but a rule verdict is bounded to 0 or 1"
        );
    }

    // The predicates must disagree with each other; identical programs would
    // mean a macro dropped its predicate argument.
    let mut fingerprints: Vec<[u8; 32]> = leaves
        .iter()
        .map(|(_, program)| program.fingerprint())
        .collect();
    fingerprints.sort_unstable();
    fingerprints.dedup();
    assert_eq!(
        fingerprints.len(),
        leaves.len(),
        "each leaf must lower its own predicate, not a shared one"
    );
}

/// The leaf table above is complete against the tree.
///
/// WHY: a leaf is a macro-expanded module, so there is nothing to iterate at
/// run time and a hand-written table goes stale in silence. The declared set
/// is therefore read from `vyre-libs/src/rule` on each run: every
/// `vyre-libs::rule::` operation id found there is either covered by the table
/// or is the one composite that is not a leaf. Adding a twelfth predicate
/// turns this red until it is pinned, instead of shipping unjudged.
///
/// The floor is what stops a broken scan from passing vacuously: a reader that
/// finds nothing would otherwise report perfect coverage of an empty set.
#[test]
fn the_leaf_table_covers_every_rule_operation_declared_in_the_tree() {
    let root =
        vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")).join("src/rule");
    let mut sources = vec![root.with_extension("rs")];
    for entry in std::fs::read_dir(&root).expect("Fix: vyre-libs/src/rule must be readable") {
        let path = entry
            .expect("Fix: a rule source entry must be readable")
            .path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            sources.push(path);
        }
    }

    // Only a string literal counts, so a prose mention of an id in a doc
    // comment cannot invent a leaf.
    const OPENING: &str = "\"vyre-libs::rule::";
    let mut declared: Vec<String> = Vec::new();
    for path in &sources {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("Fix: {path:?} must be readable: {error}"));
        let mut rest = text.as_str();
        while let Some(start) = rest.find(OPENING) {
            rest = &rest[start + 1..];
            let end = rest
                .find('"')
                .expect("Fix: an operation id literal must be closed");
            declared.push(rest[..end].to_string());
            rest = &rest[end..];
        }
    }
    declared.sort();
    declared.dedup();

    assert!(
        declared.len() >= 12,
        "found only {} `vyre-libs::rule::` operation id literals under \
         src/rule; the scan is broken, so its coverage verdict means nothing",
        declared.len()
    );

    let mut covered: Vec<String> = leaves()
        .iter()
        .map(|(op_id, _)| (*op_id).to_string())
        .collect();
    // The one rule operation that is not a condition leaf: the composite that
    // lowers a whole formula, built by `rule::builder`, not by
    // `condition_program`.
    covered.push(RULE_SET_OP_ID.to_string());
    covered.sort();
    covered.dedup();

    assert_eq!(
        declared, covered,
        "the rule operations declared in the tree and the ones this file pins \
         have diverged"
    );
}
