use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// A buffer the backend allocates and writes, so it takes no host input slot.
pub(crate) fn u32_out_buffer(name: &'static str, binding: u32) -> BufferDecl {
    BufferDecl::output(name, binding, DataType::U32).with_count(1)
}

/// Two read buffers, one allocated output, and one lane that stores
/// `expr(a[0], b[0])`. Both the parity suite and the generated boundary matrix
/// judge the same binary shape, so the shape has one definition.
pub(crate) fn binary_program(expr: fn(Expr, Expr) -> Expr) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            BufferDecl::read("b", 1, DataType::U32),
            u32_out_buffer("out", 2),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("idx", Expr::u32(0)),
            Node::store(
                "out",
                Expr::var("idx"),
                expr(
                    Expr::load("a", Expr::var("idx")),
                    Expr::load("b", Expr::var("idx")),
                ),
            ),
        ],
    )
}

pub(crate) fn dispatch_no_input(program: &Program) -> Vec<Vec<u8>> {
    dispatch_with_inputs(program, &[])
}

pub(crate) fn dispatch_with_inputs(program: &Program, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let backend = CpuRefBackend;
    backend
        .dispatch(program, inputs, &DispatchConfig::default())
        .expect("Fix: cpu-ref dispatch must succeed for a valid Program.")
}
