//! Async memory transfer evaluation for reference node execution.

use vyre_foundation::ir::{Expr, Program};

use crate::execution::async_transfer::{self, AsyncTransfer};
use crate::execution::expr as eval_expr;
use crate::oob;
use crate::workgroup::{Invocation, Memory};
use crate::ReferenceError;

pub(crate) struct AsyncLoadEval<'a> {
    pub(crate) source: &'a str,
    pub(crate) destination: &'a str,
    pub(crate) offset: &'a Expr,
    pub(crate) size: &'a Expr,
    pub(crate) tag: &'a str,
}

pub(crate) struct AsyncStoreEval<'a> {
    pub(crate) source: &'a str,
    pub(crate) destination: &'a str,
    pub(crate) offset: &'a Expr,
    pub(crate) size: &'a Expr,
    pub(crate) tag: &'a str,
}

pub(crate) fn eval_async_load(
    request: AsyncLoadEval<'_>,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let start = eval_byte_count(
        request.offset,
        "async load source offset",
        invocation,
        memory,
        program,
    )?;
    let byte_count = eval_byte_count(request.size, "async load size", invocation, memory, program)?;
    let payload = read_bytes(memory, program, request.source, start, byte_count)?;
    ensure_writable_buffer(memory, program, request.destination)?;
    invocation.begin_async(
        request.tag,
        AsyncTransfer::load(request.destination, payload),
    )
}

pub(crate) fn eval_async_store(
    request: AsyncStoreEval<'_>,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let start = eval_byte_count(
        request.offset,
        "async store destination offset",
        invocation,
        memory,
        program,
    )?;
    let byte_count = eval_byte_count(
        request.size,
        "async store size",
        invocation,
        memory,
        program,
    )?;
    let payload = read_bytes(memory, program, request.source, 0, byte_count)?;
    ensure_writable_buffer(memory, program, request.destination)?;
    invocation.begin_async(
        request.tag,
        AsyncTransfer::store(request.destination, start, payload),
    )
}

pub(crate) fn eval_async_wait(
    tag: &str,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    apply_async_transfer(invocation.finish_async(tag)?, memory, program)
}

fn eval_byte_count(
    expr: &Expr,
    label: &str,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<usize, ReferenceError> {
    let value = eval_expr::eval(expr, invocation, memory, program)?;
    async_transfer::byte_count(&value, label)
}

fn read_bytes(
    memory: &Memory,
    program: &Program,
    source: &str,
    start: usize,
    byte_count: usize,
) -> Result<Vec<u8>, ReferenceError> {
    Ok(resolve_buffer(memory, program, source)?.read_window(start, byte_count))
}

fn ensure_writable_buffer(
    memory: &mut Memory,
    program: &Program,
    name: &str,
) -> Result<(), ReferenceError> {
    eval_expr::buffer_mut(memory, program, name).map(|_| ())
}

fn apply_async_transfer(
    transfer: AsyncTransfer,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), ReferenceError> {
    let buffer = eval_expr::buffer_mut(memory, program, transfer.destination())?;
    transfer.apply_to(buffer);
    Ok(())
}

fn resolve_buffer<'a>(
    memory: &'a Memory,
    program: &Program,
    name: &str,
) -> Result<&'a oob::Buffer, ReferenceError> {
    let decl = program.buffer(name).ok_or_else(|| {
        ReferenceError::new(format!(
            "missing buffer declaration `{name}`. Fix: declare every async transfer buffer."
        ))
    })?;
    if decl.access() == vyre_foundation::ir::BufferAccess::Workgroup {
        memory.workgroup.get(name)
    } else {
        memory.storage.get(name)
    }
    .ok_or_else(|| {
        ReferenceError::new(format!(
            "missing buffer `{name}`. Fix: initialize every declared async transfer buffer."
        ))
    })
}
