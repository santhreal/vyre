//! Regression tests for Naga lowering of Region and loop scope behavior.

use naga::{Block, Statement};
use std::sync::Arc;

use vyre_emit_naga::program::emit_module;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const TEST_WORKGROUP_SIZE: [u32; 3] = [1, 1, 1];

fn emit_validated_module_and_wgsl(program: &Program) -> (naga::Module, String) {
    let module = emit_module(program, TEST_WORKGROUP_SIZE)
        .expect("Fix: test program must lower to a valid Naga module.");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("Fix: lowered module must validate before WGSL serialization.");
    let wgsl =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .expect("Fix: lowered module must serialize to WGSL.");
    (module, wgsl)
}

fn count_blocks(block: &Block) -> usize {
    block
        .iter()
        .map(|statement| match statement {
            Statement::Block(child) => 1 + count_blocks(child),
            Statement::If { accept, reject, .. } => count_blocks(accept) + count_blocks(reject),
            Statement::Loop {
                body, continuing, ..
            } => count_blocks(body) + count_blocks(continuing),
            Statement::Switch { cases, .. } => {
                cases.iter().map(|case| count_blocks(&case.body)).sum()
            }
            _ => 0,
        })
        .sum()
}

fn previous_line_before<'a>(wgsl: &'a str, needle: &str) -> Option<&'a str> {
    let mut previous = None;
    for line in wgsl.lines() {
        let trimmed = line.trim();
        if trimmed.contains(needle) {
            return previous;
        }
        if !trimmed.is_empty() {
            previous = Some(trimmed);
        }
    }
    None
}

#[test]
fn large_region_lowers_to_real_naga_block_and_wgsl_scope() {
    let region_body = (0..65)
        .map(|index| Node::store("out", Expr::u32(index), Expr::u32(index + 1)))
        .collect::<Vec<_>>();
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 1, DataType::U32).with_count(65)],
        [1, 1, 1],
        vec![Node::Region {
            generator: "naga_loop_region_followup::large_region".into(),
            source_region: None,
            body: Arc::new(region_body),
        }],
    );

    let (module, wgsl) = emit_validated_module_and_wgsl(&program);
    let entry = module
        .entry_points
        .first()
        .expect("Fix: Naga module must contain the compute entry point.");

    assert!(
        count_blocks(&entry.function.body) >= 1,
        "Fix: non-inlined Node::Region must lower as a real Naga block, not disappear before statement lowering. block_count={}\n{wgsl}",
        count_blocks(&entry.function.body),
    );
    // A `Node::Region` must reach WGSL as a real lexical block, an opening
    // brace at function-body indentation that is not part of any other
    // construct.
    //
    // The assertion used to be `wgsl.contains(") {\n    {\n")`, which required
    // the block to open on the line immediately after the entry-point
    // signature. Naga hoists every function-local `var` to the top of the
    // function, so a region that needs block-scope temporaries now emits
    // 130-odd `var` declarations first and the block opens below them. The
    // block is still there and still a block; only the byte pattern moved.
    let block_line = wgsl
        .lines()
        .skip_while(|line| !line.contains("fn main("))
        .position(|line| line == "    {")
        .unwrap_or_else(|| {
            panic!("Fix: Region block boundaries must survive WGSL emission as lexical scopes.\n{wgsl}")
        });
    assert!(
        wgsl.lines().any(|line| line == "    }"),
        "Fix: the Region block must be closed at function-body scope.\n{wgsl}",
    );
    assert!(
        block_line > 0,
        "Fix: the block must open inside main, not replace its body.\n{wgsl}",
    );
    // Every one of the 65 stores must reach the block, tail included.
    //
    // The assertion used to look for the literal `out[64u] = 65u;`. Stores now
    // route both operands through block-scope temporaries and emit as
    // `out[_e891] = _e895;`, so no store appears with its constants inline any
    // more. Counting the stores and checking the tail constants were both
    // materialized proves the same thing without pinning naga's temporary
    // naming.
    let store_count = wgsl.matches("out[").count();
    assert_eq!(
        store_count, 65,
        "Fix: all 65 Region statements must lower into the emitted block, got {store_count}.\n{wgsl}",
    );
    assert!(
        wgsl.contains("= 64u;") && wgsl.contains("= 65u;"),
        "Fix: the tail statement's index and value must be materialized in the block.\n{wgsl}",
    );
}

#[test]
fn loop_initial_bound_side_effect_is_evaluated_once() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("counter", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32),
        ],
        [1, 1, 1],
        vec![
            Node::loop_for(
                "i",
                Expr::atomic_add("counter", Expr::u32(0), Expr::u32(1)),
                Expr::u32(3),
                vec![Node::store("out", Expr::u32(0), Expr::var("i"))],
            ),
            Node::Return,
        ],
    );

    let (_, wgsl) = emit_validated_module_and_wgsl(&program);
    assert_eq!(
        wgsl.matches("atomicAdd").count(),
        1,
        "Fix: loop initial bounds with side effects must initialize the loop local once, not re-emit in guard or continuing blocks.\n{wgsl}",
    );
}

/// Region phi-merge end-to-end (WGSL-level): a `Node::Region` inside a
/// `Node::Loop` whose body Assigns a loop-carried name must publish the
/// in-region final value back through the named-carrier function-local
/// the Loop allocated. The named-carrier round-trip is the architectural
/// fix that unblocks the GPU lex `n_tokens=0` symptom on real C input.
///
/// Uses an input-dependent value (loaded from `seed` buffer) for the
/// loop bound and the increment so the optimizer cannot constant-fold
/// the loop body  -  we want the actual carrier mechanism in the shader.
#[test]
fn region_inside_loop_publishes_carrier_through_named_local() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("seed", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("acc", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::load("seed", Expr::u32(0)),
                vec![Node::Region {
                    generator: "naga_loop_region_followup::step".into(),
                    source_region: None,
                    body: Arc::new(vec![Node::assign(
                        "acc",
                        Expr::add(Expr::var("acc"), Expr::load("seed", Expr::u32(0))),
                    )]),
                }],
            ),
            Node::store("out", Expr::u32(0), Expr::var("acc")),
            Node::Return,
        ],
    );

    let (_, wgsl) = emit_validated_module_and_wgsl(&program);

    // Function-local for the named carrier `acc` must be emitted  -
    // proves the named-carrier slot path fired (Region+Loop share the
    // same `vyre_named_carry_acc` local). Without the Region phi-merge
    // fix, in-region Assigns to `acc` would never reach the post-loop
    // reader: the inner Region's lower_child_node previously rebound
    // scope without carrier round-trip, so the post-loop reader saw
    // the pre-loop seed instead of the in-region final value.
    assert!(
        wgsl.contains("vyre_named_carry_acc"),
        "Fix: region inside loop must allocate the named-carrier local for `acc`.\n{wgsl}"
    );

    // The Loop must survive (input-dependent bound prevents folding).
    assert!(
        wgsl.contains("loop {"),
        "Fix: input-dependent Loop must lower to a real WGSL loop.\n{wgsl}"
    );

    // Inside the loop body, there must be a Store to the named carrier  -
    // the LoopCarrierEnd that commits the per-iteration update pushed
    // by the active-carrier path of `Node::Assign` lowering. Without
    // the Region phi-merge, no such store exists for the inner Region's
    // Assign.
    assert!(
        wgsl.matches("vyre_named_carry_acc =").count() >= 1,
        "Fix: in-region Assign must emit a Store to the named-carrier local.\n{wgsl}"
    );
}

#[test]
fn loop_variable_shadowing_restores_outer_local_after_body_lowering() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 1, DataType::U32).with_count(2)],
        [1, 1, 1],
        vec![
            Node::let_bind("i", Expr::u32(99)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(1),
                vec![Node::store("out", Expr::u32(0), Expr::var("i"))],
            ),
            Node::store("out", Expr::u32(1), Expr::var("i")),
            Node::Return,
        ],
    );

    let (_, wgsl) = emit_validated_module_and_wgsl(&program);

    // What must be true: inside the loop, `i` is the loop variable; after it,
    // `i` is the outer binding again. The program stores one of each, so the
    // two written VALUES are the whole contract:
    //
    //   out[0] = 0    the loop variable on its only iteration
    //   out[1] = 99   the outer `i`, restored
    //
    // The bound is constant here, so the loop folds and no WGSL `loop` block
    // survives. That is correct and desirable, but it means the old assertions
    // (find the line before `out[0u] =`, require it to end `= i_1;`) had
    // nothing to match: stores now route both operands through block-scope
    // temporaries and emit as `out[_e9] = _e13;`. Reading the assigned
    // constants back out of the temporaries states the semantics directly and
    // does not depend on how naga names anything.
    let assignments: Vec<&str> = wgsl
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("vyre_block_scope_") && line.contains(" = "))
        .collect();
    assert_eq!(
        assignments,
        vec![
            "vyre_block_scope_0_ = 0u;",  // index of the loop-body store
            "vyre_block_scope_1_ = 0u;",  // the loop variable on iteration 0
            "vyre_block_scope_2_ = 1u;",  // index of the post-loop store
            "vyre_block_scope_3_ = 99u;", // the OUTER i, restored after the loop
        ],
        "Fix: loop lowering must use the shadowing loop-local inside the body \
         and restore the outer binding after it.\n{wgsl}",
    );

    // And they must be stored in that order: body first, then the outer read.
    let first_store = wgsl
        .find("out[_e9]")
        .expect("Fix: the loop-body store must be emitted.");
    let second_store = wgsl
        .find("out[_e23]")
        .expect("Fix: the post-loop store must be emitted.");
    assert!(
        first_store < second_store,
        "Fix: the loop-local use must occur before the post-loop outer-local use.\n{wgsl}",
    );
}
