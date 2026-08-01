//! A registered op whose body calls another registered op must lower.
//!
//! The composition-discipline gate tells an over-budget op to "split into
//! smaller compositions connected via `Expr::Call`". That instruction was
//! unfollowable: the canonical pre-emit pipeline inlines through
//! `vyre_foundation::ir::inline_calls`, whose default resolver returned
//! `None` for every op id, so any split op died with `InlineUnknownOp`
//! before a backend saw it. Only `vyre-aot` passed a real resolver. The
//! result was a gate mandating a structure the pipeline refused to
//! compile, and every op that needed splitting stayed a monolith.
//!
//! The default resolver now asks the installed dialect lookup. These tests
//! pin that a split op survives the whole path: the call resolves against
//! the registry, and real PTX comes out the other end.
//!
//! Scope: a callee takes SCALAR arguments. It cannot take a buffer, so a
//! phase that indexes a table is still not splittable. See BACKLOG.md R26
//! for what a first-class buffer-reference expression would need.

use vyre_driver::registry::{
    Category, DialectRegistry, OpDef, OpDefRegistration, Signature, TypedParam,
};
use vyre_driver::DispatchConfig;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const CALLEE_OP_ID: &str = "test::split_lowering::row_pair_sum";
const CALLER_OP_ID: &str = "test::split_lowering::caller";

/// Callee phase: a scalar computation over the value it is handed.
fn row_pair_sum() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("row", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [64, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::add(
                Expr::mul(Expr::load("row", Expr::u32(0)), Expr::u32(3)),
                Expr::u32(7),
            ),
        )],
    )
    .with_entry_op_id(CALLEE_OP_ID)
}

/// Caller: the shape an over-budget op takes after a split. It reads the
/// invocation id itself and hoists it to the call site as an argument,
/// because a callee body may not read a per-invocation built-in.
fn caller() -> Program {
    let row = Expr::InvocationId { axis: 0 };
    Program::wrapped(
        vec![
            BufferDecl::storage("nodes", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32).with_count(64),
        ],
        [64, 1, 1],
        vec![Node::if_then(
            Expr::lt(row.clone(), Expr::u32(32)),
            vec![Node::store(
                "out",
                row.clone(),
                Expr::call(CALLEE_OP_ID, vec![row]),
            )],
        )],
    )
    .with_entry_op_id(CALLER_OP_ID)
}

const CALLEE_SIG: Signature = Signature {
    inputs: &[TypedParam {
        name: "row",
        ty: "u32",
    }],
    outputs: &[TypedParam {
        name: "result",
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

inventory::submit! {
    OpDefRegistration::new(|| OpDef {
        id: CALLEE_OP_ID,
        dialect: "test",
        category: Category::Composite,
        signature: CALLEE_SIG,
        lowerings: vyre_foundation::LoweringTable::empty(),
        laws: &[],
        compose: Some(row_pair_sum),
    })
}

/// Force the process-wide registry to exist so the default inline
/// resolver has a provider to ask.
fn install_registry() {
    let _ = DialectRegistry::global();
}

/// The call resolves and disappears. Before the fix this failed with
/// `InlineUnknownOp` naming the callee.
#[test]
fn a_call_to_a_registered_op_resolves_in_the_pre_emit_pipeline() {
    install_registry();
    let prepared = vyre_lower::prepare_program_for_emit(&caller())
        .unwrap_or_else(|error| panic!("split op must survive pre-emit: {error}"));
    let dump = format!("{:?}", prepared.entry());
    assert!(
        !dump.contains("Call {"),
        "every call must be inlined out: {dump}"
    );
}

/// The callee's own input buffer must not survive into the caller: the
/// argument replaces every read of it.
#[test]
fn the_callees_input_buffer_does_not_leak_into_the_caller() {
    install_registry();
    let prepared = vyre_lower::prepare_program_for_emit(&caller()).expect("pre-emit");
    let dump = format!("{:?}", prepared.entry());
    assert!(
        !dump.contains("\"row\""),
        "no read may still name the callee's own buffer: {dump}"
    );
}

/// End to end: a split op emits real PTX, and the store address is
/// register-computed because the invocation id reached the store intact.
#[test]
fn a_split_op_emits_ptx() {
    install_registry();
    let ptx = vyre_driver_cuda::codegen::program_to_ptx(&caller(), &DispatchConfig::default())
        .unwrap_or_else(|error| panic!("split op must emit PTX: {error}"));
    let stores: Vec<&str> = ptx
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("st.global"))
        .collect();
    assert_eq!(stores.len(), 1, "one store, got {stores:?}");
    assert!(
        !stores[0].contains('+'),
        "the store address must be register-computed, not a constant offset: {}",
        stores[0]
    );
}
