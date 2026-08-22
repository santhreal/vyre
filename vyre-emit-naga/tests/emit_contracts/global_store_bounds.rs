//! Every store to a storage binding is issued under that binding's own
//! bounds test.
//!
//! WGSL has no unchecked store. Naga's bounds-check policy rewrites an
//! out-of-range storage write onto the nearest in-range element instead of
//! dropping it, so a lane past the end of a short buffer overwrites a live
//! element with whatever value that lane carried. The dispatch extent comes
//! from the longest buffer in the program, so any shorter buffer is reached by
//! lanes that do not belong to it. The reference interpreter discards such a
//! store, which is why an unguarded one is a parity defect and not a cosmetic
//! one.
//!
//! What this checks is the shape at the choke point: a `Statement::Store`
//! whose pointer roots in a storage global must sit directly inside the accept
//! arm of an `If` testing `index < length`. Checking the shape rather than a
//! value catches a route that was never wired to the guard, including one
//! whose own parity fixture happens to stay in range.
//!
//! The population is the shared adversarial success corpus read at run time
//! plus one descriptor per global-store route, so a new corpus case is covered
//! without being named here.

use super::*;
use naga::{AddressSpace, BinaryOperator, Expression, Handle};

/// Storage or workgroup global a pointer expression ultimately addresses.
///
/// Returns `None` for a pointer rooted in a local variable or a function
/// argument, which no bounds guard applies to.
fn pointer_root(
    function: &naga::Function,
    mut expr: Handle<Expression>,
) -> Option<Handle<naga::GlobalVariable>> {
    loop {
        match &function.expressions[expr] {
            Expression::GlobalVariable(global) => return Some(*global),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => expr = *base,
            _ => return None,
        }
    }
}

fn writes_storage(module: &naga::Module, function: &naga::Function, statement: &Statement) -> bool {
    let Statement::Store { pointer, .. } = statement else {
        return false;
    };
    pointer_root(function, *pointer).is_some_and(|global| {
        matches!(
            module.global_variables[global].space,
            AddressSpace::Storage { .. }
        )
    })
}

/// Whether `condition` is the `index < length` comparison a bounds guard
/// emits. A guard built from any other operator would admit the very lane it
/// is supposed to drop.
fn is_upper_bound_test(function: &naga::Function, condition: Handle<Expression>) -> bool {
    matches!(
        function.expressions[condition],
        Expression::Binary {
            op: BinaryOperator::Less,
            ..
        }
    )
}

/// Storage stores in `block` that no enclosing upper-bound test guards.
///
/// A store nested inside descriptor-level control flow still carries its own
/// guard, so the immediately enclosing arm is the one that has to be a bounds
/// test; inheriting the flag through `Statement::Block` only follows naga's
/// own regrouping.
///
/// A counted loop body counts as bounded. The only stores the emitter puts
/// inside a loop it generated itself are the async copy words, whose trip
/// count is already clamped to the destination length
/// (`async_op::emit_async_load`, `emit_async_store`), so a per-word test there
/// would be a branch that can never be taken. Every store the descriptor asks
/// for keeps its own guard whatever it is nested in, which is what the route
/// contract below checks directly.
fn unguarded_storage_stores(
    module: &naga::Module,
    function: &naga::Function,
    block: &naga::Block,
    guarded: bool,
) -> usize {
    block
        .iter()
        .map(|statement| {
            let here = usize::from(!guarded && writes_storage(module, function, statement));
            let nested = match statement {
                Statement::Block(child) => {
                    unguarded_storage_stores(module, function, child, guarded)
                }
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let bounded = is_upper_bound_test(function, *condition);
                    unguarded_storage_stores(module, function, accept, bounded)
                        + unguarded_storage_stores(module, function, reject, false)
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    unguarded_storage_stores(module, function, body, true)
                        + unguarded_storage_stores(module, function, continuing, true)
                }
                _ => 0,
            };
            here + nested
        })
        .sum()
}

fn assert_every_storage_store_is_guarded(module: &naga::Module, case: &str) {
    for entry in &module.entry_points {
        let unguarded =
            unguarded_storage_stores(module, &entry.function, &entry.function.body, false);
        assert_eq!(
            unguarded, 0,
            "{case}: {unguarded} storage store(s) are emitted without an `index < length` test. \
             Fix: route the store through `push_bounds_guarded_global_store`."
        );
    }
}

fn store_route_descriptors() -> Vec<KernelDescriptor> {
    let scalar = |id: &str, element_type: DataType| {
        descriptor(id)
            .slots([global_rw(0, element_type, "out").with_count(4)])
            .dispatch(64, 1, 1)
            .body(
                body()
                    .ops([
                        lit(0, 0),
                        lit(1, 1),
                        effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                    ])
                    .literals([LiteralValue::U32(0), LiteralValue::U32(7)]),
            )
            .build()
    };
    let vector = |id: &str, width: u8| {
        // One value id per lane, all reading the same literal: the route walks
        // `operands[2..]`, so a short operand list is a descriptor error rather
        // than a narrower store.
        let mut ops = vec![lit(0, 0)];
        let mut operands = vec![0, 0];
        for lane in 0..u32::from(width) {
            ops.push(lit(1, 1 + lane));
            operands.push(1 + lane);
        }
        ops.push(effect(KernelOpKind::VectorStoreGlobal { width }, operands));
        descriptor(id)
            .slots([global_rw(0, DataType::U32, "out").with_count(8)])
            .dispatch(64, 1, 1)
            .body(
                body()
                    .ops(ops)
                    .literals([LiteralValue::U32(0), LiteralValue::U32(7)]),
            )
            .build()
    };
    vec![
        scalar("store_u32", DataType::U32),
        scalar("store_u8", DataType::U8),
        scalar("store_i8", DataType::I8),
        scalar("store_f32", DataType::F32),
        vector("store_vec2", 2),
        vector("store_vec4", 4),
    ]
}

#[test]
fn every_global_store_route_emits_its_own_bounds_test() {
    for desc in store_route_descriptors() {
        let module = emit(&desc).unwrap_or_else(|error| {
            panic!(
                "Fix: store-route descriptor `{}` must emit: {error:?}",
                desc.id
            )
        });
        assert_every_storage_store_is_guarded(&module, &desc.id);
    }
}

#[test]
fn the_shared_success_corpus_emits_no_unguarded_storage_store() {
    let cases = vyre_lower::emit_adversarial_corpus::success_cases();
    assert!(
        !cases.is_empty(),
        "Fix: the shared success corpus is empty, so this contract checks nothing."
    );
    for case in cases {
        let verified = vyre_lower::verify_descriptor(&case.descriptor)
            .unwrap_or_else(|error| panic!("{}: corpus case must verify: {error:?}", case.id));
        let module = emit(&verified)
            .unwrap_or_else(|error| panic!("{}: corpus case must emit: {error:?}", case.id));
        assert_every_storage_store_is_guarded(&module, case.id);
    }
}

#[test]
fn a_workgroup_store_is_not_forced_through_a_bounds_test() {
    // A workgroup binding carries a compile-time length and is addressed
    // through the window the emitter resolved, so requiring a runtime test
    // there would only cost an instruction. The probe must therefore look at
    // the address space, not at the statement.
    let desc = descriptor("store_shared")
        .slots([
            global_rw(0, DataType::U32, "out").with_count(8),
            BindingSlot {
                slot: 1,
                element_type: DataType::U32,
                element_count: Some(8),
                memory_class: MemoryClass::Shared,
                visibility: BindingVisibility::ReadWrite,
                name: "tile".to_owned(),
            },
        ])
        .dispatch(8, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::StoreShared, [1, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)]),
        )
        .build();
    let module = emit(&desc).expect("shared-store descriptor must emit");
    let entry = &module.entry_points[0].function;
    let shared_stores = count_statements(&entry.body, &|statement| {
        matches!(statement, Statement::Store { .. })
    });
    assert_eq!(
        shared_stores, 1,
        "the shared store must still be emitted exactly once"
    );
    assert_every_storage_store_is_guarded(&module, "store_shared");
}
