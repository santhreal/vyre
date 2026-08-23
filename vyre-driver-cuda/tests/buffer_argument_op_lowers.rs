//! A composite op that indexes a caller-supplied table must reach PTX.
//!
//! This is the case the earlier scalar-only split could not express. A phase
//! that reads `table[row]` needs the table itself, not one value from it, so
//! before `Expr::BufferRef` existed such a phase could not be split out of an
//! over-budget op at all: the composition-discipline gate demanded a split the
//! pipeline could not compile.
//!
//! These tests pin buffer-argument signature validation, canonical semantic
//! resolution, and emitted PTX.

#![cfg(feature = "device-tests")]

use vyre_driver::DispatchConfig;
use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::operation::{OperationRegistration, OperationTier};

const LOOKUP_OP_ID: &str = "test::buffer_arg::table_lookup";
const CALLER_OP_ID: &str = "test::buffer_arg::caller";

/// Callee phase: read `table[i * 2]`. `table` is a buffer parameter, `i` a
/// scalar one, so one callee exercises both binding kinds.
fn table_lookup() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("table", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("i", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("result", 2, DataType::U32).with_count(1),
        ],
        [64, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::load(
                "table",
                Expr::mul(Expr::load("i", Expr::u32(0)), Expr::u32(2)),
            ),
        )],
    )
    .with_entry_op_id(LOOKUP_OP_ID)
}

/// Caller: hands the callee its own `nodes` table plus the invocation id,
/// which a callee body may not read for itself.
fn caller() -> Program {
    let row = Expr::InvocationId { axis: 0 };
    Program::wrapped(
        vec![
            BufferDecl::storage("nodes", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32).with_count(32),
        ],
        [64, 1, 1],
        vec![Node::if_then(
            Expr::lt(row.clone(), Expr::u32(32)),
            vec![Node::store(
                "out",
                row.clone(),
                Expr::call(LOOKUP_OP_ID, vec![Expr::buffer_ref("nodes"), row]),
            )],
        )],
    )
    .with_entry_op_id(CALLER_OP_ID)
}

const LOOKUP_SIG: Signature = Signature {
    inputs: &[
        TypedParam {
            name: "table",
            ty: "buffer<u32>",
        },
        TypedParam {
            name: "i",
            ty: "u32",
        },
    ],
    outputs: &[TypedParam {
        name: "result",
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

inventory::submit! {
    OperationRegistration::new_unconstrained(
        LOOKUP_OP_ID,
        OperationTier::External,
        Some(table_lookup),
        None,
        None,
    )
    .with_signature(LOOKUP_SIG)
    .with_category("test")
}

/// The call resolves, and the callee's read moves onto the caller's buffer.
#[test]
fn a_buffer_argument_retargets_the_read_onto_the_callers_table() {
    let prepared = vyre_lower::lower_physical(&caller())
        .unwrap_or_else(|error| {
            panic!("buffer-argument op must survive physical lowering: {error}")
        })
        .program;
    let dump = format!("{:?}", prepared.entry());
    assert!(
        !dump.contains("Call {"),
        "every call must be inlined out: {dump}"
    );
    assert!(
        !dump.contains("BufferRef"),
        "the buffer reference must be consumed by inlining, not survive into the emitted program: {dump}"
    );
    assert!(
        !dump.contains("\"table\""),
        "no read may still name the callee's own parameter buffer: {dump}"
    );
    assert!(
        dump.contains("\"nodes\""),
        "the retargeted read must name the caller's buffer: {dump}"
    );
}

/// End to end: the split emits PTX, with one load of the caller's table and
/// one store, and the store address is register-computed because the
/// invocation id reached it intact.
#[test]
fn a_buffer_argument_op_emits_ptx() {
    let ptx = vyre_driver_cuda::codegen::program_to_ptx(&caller(), &DispatchConfig::default())
        .unwrap_or_else(|error| panic!("buffer-argument op must emit PTX: {error}"));
    // The kernel-parameter block is read through the parameter base register
    // at fixed offsets (`[%rd0]`, `[%rd0 + 4]`, ...); those are binding
    // pointers and lengths, not data. The table read is the one that goes
    // through an address register the kernel computed for itself.
    let data_loads: Vec<&str> = ptx
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("ld.global"))
        .filter(|line| !line.contains("%rd0"))
        .collect();
    assert_eq!(
        data_loads.len(),
        1,
        "one load of the caller's table, got {data_loads:?}"
    );
    let stores: Vec<&str> = ptx
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("st.global"))
        .collect();
    assert_eq!(stores.len(), 1, "one store, got {stores:?}");
    assert!(
        !stores[0].contains('+'),
        "the store address must be register-computed: {}",
        stores[0]
    );
}

/// Passing a value where the signature declares `buffer<u32>` would bind the
/// callee's table parameter to a single number and make every index read slot
/// zero of whatever binding it landed on. V053 rejects it.
#[test]
fn passing_a_value_for_a_buffer_parameter_is_rejected() {
    let row = Expr::InvocationId { axis: 0 };
    let bad = Program::wrapped(
        vec![
            BufferDecl::storage("nodes", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32).with_count(32),
        ],
        [64, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::call(LOOKUP_OP_ID, vec![Expr::u32(0), row]),
        )],
    );
    let report = vyre_foundation::validate::validate(&bad);
    assert!(
        report.iter().any(|e| e.code().as_str() == "V053"),
        "a scalar in a buffer parameter must raise V053, got {:?}",
        report
    );
}

/// Naming a buffer the caller never declared would lower to a binding index
/// that does not exist. V052 rejects it.
#[test]
fn referencing_an_undeclared_buffer_is_rejected() {
    let row = Expr::InvocationId { axis: 0 };
    let bad = Program::wrapped(
        vec![
            BufferDecl::storage("nodes", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32).with_count(32),
        ],
        [64, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::call(LOOKUP_OP_ID, vec![Expr::buffer_ref("missing"), row]),
        )],
    );
    let report = vyre_foundation::validate::validate(&bad);
    assert!(
        report.iter().any(|e| e.code().as_str() == "V052"),
        "an undeclared buffer reference must raise V052, got {:?}",
        report
    );
}
