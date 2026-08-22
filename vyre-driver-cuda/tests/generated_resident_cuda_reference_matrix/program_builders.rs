use super::*;

/// A resident-matrix program: `reads` bound at 0..n, one `out` buffer last, and
/// one lane-guarded store of `value` at `dst`.
///
/// Every resident matrix pins this exact shape, including the lane index being
/// inlined as `gid_x` rather than bound to a name, so it is built in one place
/// instead of restated per dtype.
fn resident_lane_program(
    reads: &[(&str, DataType)],
    output: DataType,
    dst: Expr,
    value: Expr,
) -> Program {
    let lanes = LANE_COUNT as u32;
    let mut buffers: Vec<BufferDecl> = reads
        .iter()
        .enumerate()
        .map(|(binding, (name, ty))| {
            BufferDecl::read(name, binding as u32, ty.clone()).with_count(lanes)
        })
        .collect();
    buffers.push(BufferDecl::output("out", reads.len() as u32, output).with_count(lanes));
    Program::wrapped(
        buffers,
        [WORKGROUP_SIZE_X, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(lanes)),
            vec![Node::store("out", dst, value)],
        )],
    )
}

/// `out[gid] = build(lhs[gid], rhs[gid])` over two `input_type` buffers.
fn binary_lane_program(
    build: fn(Expr, Expr) -> Expr,
    input_type: DataType,
    output: DataType,
) -> Program {
    let value = build(
        Expr::load("lhs", Expr::gid_x()),
        Expr::load("rhs", Expr::gid_x()),
    );
    resident_lane_program(
        &[("lhs", input_type.clone()), ("rhs", input_type)],
        output,
        Expr::gid_x(),
        value,
    )
}

/// `out[gid] = build(input[gid])` over one `input_type` buffer.
fn unary_lane_program(
    build: impl FnOnce(Expr) -> Expr,
    input_type: DataType,
    output: DataType,
) -> Program {
    let value = build(Expr::load("input", Expr::gid_x()));
    resident_lane_program(&[("input", input_type)], output, Expr::gid_x(), value)
}

pub(crate) fn resident_bool_binary_program(case: &BoolBinaryCase) -> Program {
    binary_lane_program(case.build, DataType::Bool, DataType::Bool)
}

pub(crate) fn resident_u32_binary_program(case: &ResidentBinaryCase) -> Program {
    binary_lane_program(case.build, DataType::U32, DataType::U32)
}

pub(crate) fn resident_u32_unary_program(case: &ResidentUnaryCase) -> Program {
    unary_lane_program(case.build, DataType::U32, DataType::U32)
}

pub(crate) fn resident_i32_binary_program(case: &ResidentBinaryCase) -> Program {
    binary_lane_program(case.build, DataType::I32, DataType::I32)
}

pub(crate) fn resident_i32_unary_program(case: &ResidentUnaryCase) -> Program {
    unary_lane_program(case.build, DataType::I32, DataType::I32)
}

/// `out[build_dst(gid)] = build_value(input[build_src(gid)])`. The store
/// destination is deliberately not the lane index: a permutation matrix exists
/// to catch a lowering that assumes it is.
pub(crate) fn resident_memory_program(case: &ResidentMemoryCase) -> Program {
    let value = (case.build_value)(Expr::load("input", (case.build_src)(Expr::gid_x())));
    resident_lane_program(
        &[("input", case.ty.clone())],
        case.ty.clone(),
        (case.build_dst)(Expr::gid_x()),
        value,
    )
}

pub(crate) fn resident_bool_unary_program(case: &BoolUnaryCase) -> Program {
    unary_lane_program(case.build, DataType::Bool, DataType::Bool)
}

pub(crate) fn resident_bool_select_program() -> Program {
    let value = Expr::select(
        Expr::load("flag", Expr::gid_x()),
        Expr::load("lhs", Expr::gid_x()),
        Expr::load("rhs", Expr::gid_x()),
    );
    resident_lane_program(
        &[
            ("flag", DataType::Bool),
            ("lhs", DataType::Bool),
            ("rhs", DataType::Bool),
        ],
        DataType::Bool,
        Expr::gid_x(),
        value,
    )
}

pub(crate) fn resident_f32_compare_program(case: &F32CompareCase) -> Program {
    binary_lane_program(case.build, DataType::F32, DataType::U32)
}

pub(crate) fn resident_f32_binary_program(case: &F32BinaryCase) -> Program {
    binary_lane_program(case.build, DataType::F32, DataType::F32)
}

pub(crate) fn resident_f32_unary_program(case: &F32UnaryCase) -> Program {
    unary_lane_program(case.build, DataType::F32, DataType::F32)
}

pub(crate) fn resident_f32_classify_program(case: &F32ClassifyCase) -> Program {
    unary_lane_program(case.build, DataType::F32, DataType::U32)
}

/// The one resident matrix that is not a guarded store: the accumulator is a
/// read-write binding the atomic updates in place, and the result the matrix
/// checks is the accumulator itself.
pub(crate) fn resident_atomic_reduction_program(case: &ResidentAtomicCase) -> Program {
    harness::build_atomic_reduction_program(
        LANE_COUNT as u32,
        WORKGROUP_SIZE_X,
        BUCKET_MASK,
        case.build,
    )
}

pub(crate) fn resident_cast_program(case: &CastCase) -> Program {
    let output_type = case.output_type.clone();
    unary_lane_program(
        |input| Expr::cast(output_type.clone(), input),
        case.input_type.clone(),
        case.output_type.clone(),
    )
}

pub(crate) fn resident_fma_program() -> Program {
    let value = Expr::fma(
        Expr::load("a", Expr::gid_x()),
        Expr::load("b", Expr::gid_x()),
        Expr::load("c", Expr::gid_x()),
    );
    resident_lane_program(
        &[
            ("a", DataType::F32),
            ("b", DataType::F32),
            ("c", DataType::F32),
        ],
        DataType::F32,
        Expr::gid_x(),
        value,
    )
}
