//! Contract tests for the typed region-based SSA representation.

use vyre_foundation::ir::{BufferAccess, BufferDecl, Expr, Node, Program};
use vyre_foundation::region_ssa::{
    lower_program_to_region_ssa, lower_region_ssa_to_program, verify_dominance, DominanceError,
    RegionBuilder, RegionModule, RegionOpKind, RegionSsaConstProp, RegionSsaError, ScalarLiteral,
    ValueId,
};
use vyre_spec::{BinOp, DataType};

#[test]
fn dominance_holds_by_construction_for_inaccessible_inner_scope() {
    let mut builder = RegionBuilder::new_function(
        "test_dominance",
        vec![(DataType::U32, Some("input_len".into()))],
        vec![DataType::U32],
    );

    let c1 = builder.emit_constant(ScalarLiteral::U32(10)).unwrap();
    let mut inner_val_leak = None;

    // Build a structured map region
    let map_results = builder
        .build_map_region(
            vec![c1],
            vec![10],
            vec![DataType::U32],
            vec![DataType::U32],
            |inner_b, entry_vals| {
                let elem = entry_vals[0];
                let inner_const = inner_b.emit_constant(ScalarLiteral::U32(42))?;
                inner_val_leak = Some(inner_const); // capture inner value ID

                let sum = inner_b.emit_binary(BinOp::Add, elem, inner_const, DataType::U32)?;
                Ok(vec![sum])
            },
        )
        .expect("map region builds cleanly");

    let leaked = inner_val_leak.expect("inner val was created");

    // Attempting to use the inner value OUTSIDE the region MUST fail dominance
    let err = builder
        .emit_binary(BinOp::Add, map_results[0], leaked, DataType::U32)
        .expect_err("using inner region value outside its scope must fail dominance");

    assert_eq!(err, DominanceError::OutOfScopeValue(leaked));

    // Finish building valid function without leaked value
    builder.terminate_return(map_results).unwrap();
    let func = builder
        .build()
        .expect("valid function passes dominance verification");
    assert!(verify_dominance(&func).is_ok());
}

#[test]
fn value_id_stability_across_transformation() {
    let mut builder = RegionBuilder::new_function(
        "test_stability",
        vec![(DataType::U32, Some("x".into()))],
        vec![DataType::U32],
    );

    let param_x = ValueId(0);
    let c1 = builder.emit_constant(ScalarLiteral::U32(10)).unwrap();
    let c2 = builder.emit_constant(ScalarLiteral::U32(20)).unwrap();

    // c1 + c2 should fold to constant 30
    let folded_sum = builder
        .emit_binary(BinOp::Add, c1, c2, DataType::U32)
        .unwrap();

    // param_x + folded_sum depends on param, so remains an Add op
    let result = builder
        .emit_binary(BinOp::Add, param_x, folded_sum, DataType::U32)
        .unwrap();

    builder.terminate_return(vec![result]).unwrap();
    let func = builder.build().expect("function builds cleanly");

    // Run SSA Constant Propagation
    let (opt_func, remap) = RegionSsaConstProp::run_function(&func);

    // Assert value ID stability
    assert!(
        remap.is_stable(param_x),
        "parameter ValueId must remain stable"
    );
    assert!(
        remap.is_stable(c1),
        "constant c1 ValueId must remain stable"
    );
    assert!(
        remap.is_stable(c2),
        "constant c2 ValueId must remain stable"
    );
    assert!(
        remap.is_stable(result),
        "dynamic operation result ValueId must remain stable"
    );

    // Confirm that the folded operation became a constant 30
    let opt_block = &opt_func.blocks[0];
    let folded_op = opt_block
        .ops
        .iter()
        .find(|op| op.results[0].id == folded_sum)
        .unwrap();
    assert_eq!(
        folded_op.kind,
        RegionOpKind::Constant(ScalarLiteral::U32(30))
    );
}

#[test]
fn lowering_from_program_roundtrips_conformance_programs() {
    // 1. Vector addition program
    let prog1 = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("out", 2, BufferAccess::ReadWrite, DataType::U32),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::add(
                    Expr::load("a", Expr::var("idx")),
                    Expr::load("b", Expr::var("idx")),
                ),
            ),
        ],
    );

    let ssa_module = lower_program_to_region_ssa(&prog1).expect("program lowers to SSA cleanly");
    assert_eq!(ssa_module.globals.len(), 3);
    assert_eq!(ssa_module.functions.len(), 1);

    // Verify SSA function satisfies dominance by construction
    verify_dominance(&ssa_module.functions[0]).expect("lowered SSA satisfies dominance");

    let roundtrip_prog =
        lower_region_ssa_to_program(&ssa_module).expect("SSA lowers to Program cleanly");
    assert_eq!(roundtrip_prog.buffers.len(), 3);
    assert_eq!(roundtrip_prog.buffers[0].name.as_ref(), "a");
    assert_eq!(roundtrip_prog.buffers[1].name.as_ref(), "b");
    assert_eq!(roundtrip_prog.buffers[2].name.as_ref(), "out");
    assert_eq!(roundtrip_prog.entry.len(), 1); // wrapped top-level logical region

    // 2. Loop accumulator program
    let prog2 = Program::wrapped(
        vec![BufferDecl::storage(
            "acc",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [64, 1, 1],
        vec![Node::loop_(
            "i",
            Expr::LitU32(0),
            Expr::LitU32(10),
            vec![Node::store("acc", Expr::var("i"), Expr::LitU32(1))],
        )],
    );

    let ssa_loop = lower_program_to_region_ssa(&prog2).expect("loop program lowers to SSA cleanly");
    verify_dominance(&ssa_loop.functions[0]).expect("loop SSA satisfies dominance");

    let roundtrip_loop =
        lower_region_ssa_to_program(&ssa_loop).expect("loop SSA roundtrips to Program");
    assert_eq!(roundtrip_loop.buffers.len(), 1);
    assert_eq!(roundtrip_loop.buffers[0].name.as_ref(), "acc");
}

/// The statement-IR literal a [`ScalarLiteral`] narrows to.
///
/// Statement IR carries four literal widths. Comparing against this rather than
/// against `Expr` is what keeps the expectation independent of the conversion
/// under test: `Expr` implements no equality, and matching one of its fifty
/// variants by hand would restate `to_expr`'s own arms.
#[derive(Debug, PartialEq)]
enum Lowered {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

/// Which statement-IR literal `expr` is.
///
/// The catch-all arm is the assertion: `to_expr` yields a literal of one of the
/// four widths statement IR carries, never a computed expression.
fn lowered(expr: &Expr) -> Lowered {
    match expr {
        Expr::LitU32(value) => Lowered::U32(*value),
        Expr::LitI32(value) => Lowered::I32(*value),
        Expr::LitF32(value) => Lowered::F32(*value),
        Expr::LitBool(value) => Lowered::Bool(*value),
        other => panic!("to_expr produced a non-literal expression: {other:?}"),
    }
}

/// Position of `literal`'s width in [`NARROWING`].
///
/// Exhaustive with no catch-all arm, so a [`ScalarLiteral`] width added later
/// fails to compile until its narrowing decision joins the table.
fn width_index(literal: &ScalarLiteral) -> usize {
    match literal {
        ScalarLiteral::U32(_) => 0,
        ScalarLiteral::I32(_) => 1,
        ScalarLiteral::U64(_) => 2,
        ScalarLiteral::I64(_) => 3,
        ScalarLiteral::F32(_) => 4,
        ScalarLiteral::F64(_) => 5,
        ScalarLiteral::Bool(_) => 6,
    }
}

/// One representable sample per width, the statement-IR literal it narrows to,
/// and a value of that width statement IR cannot carry.
///
/// A width every value of which survives narrowing has no rejected sample.
const NARROWING: [(ScalarLiteral, Lowered, Option<ScalarLiteral>); 7] = [
    (ScalarLiteral::U32(7), Lowered::U32(7), None),
    (ScalarLiteral::I32(-7), Lowered::I32(-7), None),
    (
        ScalarLiteral::U64(7),
        Lowered::U32(7),
        Some(ScalarLiteral::U64(4_294_967_296)),
    ),
    (
        ScalarLiteral::I64(-7),
        Lowered::I32(-7),
        Some(ScalarLiteral::I64(-2_147_483_649)),
    ),
    (ScalarLiteral::F32(1.5), Lowered::F32(1.5), None),
    (
        ScalarLiteral::F64(1.5),
        Lowered::F32(1.5),
        Some(ScalarLiteral::F64(0.1)),
    ),
    (ScalarLiteral::Bool(true), Lowered::Bool(true), None),
];

/// Every literal width either narrows to the same value or is rejected.
///
/// Statement IR carries only 32-bit and boolean literals, so lowering used to
/// cast a 64-bit literal with `as`, and a value outside the 32-bit range
/// reached the backend as a different number with nothing reporting it.
#[test]
fn every_scalar_literal_width_narrows_exactly_or_is_rejected() {
    for (index, (literal, expected, unrepresentable)) in NARROWING.iter().enumerate() {
        assert_eq!(
            width_index(literal),
            index,
            "the narrowing table is out of order at {literal:?}"
        );

        let narrowed = literal
            .to_expr()
            .unwrap_or_else(|error| panic!("{literal:?} is representable: {error}"));
        assert_eq!(
            &lowered(&narrowed),
            expected,
            "{literal:?} narrowed wrongly"
        );

        let Some(rejected) = unrepresentable else {
            continue;
        };
        let error = rejected
            .to_expr()
            .expect_err("a value statement IR cannot carry is rejected, not truncated");
        assert!(
            matches!(
                error,
                RegionSsaError::UnrepresentableLiteral { literal } if literal == *rejected
            ),
            "{rejected:?} produced the wrong error"
        );
    }
}

/// Lowering a whole module rejects an unrepresentable constant.
///
/// The narrowing decision belongs to the lowering path, not only to the
/// conversion: a constant op carrying a value statement IR cannot hold fails
/// the lowering that reaches it.
#[test]
fn lowering_a_module_rejects_an_unrepresentable_constant() {
    let mut builder =
        RegionBuilder::new_function("unrepresentable", Vec::new(), vec![DataType::U64]);
    let constant = builder
        .emit_constant(ScalarLiteral::U64(4_294_967_296))
        .expect("Region SSA carries the full 64-bit width");
    builder
        .terminate_return(vec![constant])
        .expect("the function terminates");
    let function = builder.build().expect("the function builds cleanly");

    let mut module = RegionModule::new("unrepresentable");
    module.functions.push(function);

    let error = lower_region_ssa_to_program(&module)
        .expect_err("a constant statement IR cannot carry fails lowering");

    assert!(
        matches!(
            error,
            RegionSsaError::UnrepresentableLiteral {
                literal: ScalarLiteral::U64(4_294_967_296)
            }
        ),
        "expected an unrepresentable-literal error, got {error}"
    );
}
