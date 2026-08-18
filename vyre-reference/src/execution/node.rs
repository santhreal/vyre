//! Statement executor that gives the parity engine a pure-Rust ground truth
//! for every `Node` variant.
//!
//! This module simulates the exact control-flow, memory, and barrier behavior
//! that a correct GPU backend must produce. Any divergence in `If`, `Loop`,
//! `Barrier`, or `Store` semantics is caught by the conform gate as a concrete
//! counterexample.

use vyre_foundation::ir::{Expr, Node, Program};

use crate::value::Value;
use crate::ReferenceError;
use crate::{
    execution::async_transfer::{self, AsyncTransfer},
    execution::expr as eval_expr,
    execution::node_tree::{contains_barrier, node_id},
    oob,
    workgroup::{Frame, Invocation, Memory},
};

/// Execute one scheduling step for an invocation.
///
/// When the step is the one that ends the invocation, every async transfer it
/// started must already have been waited on. That check lives here because
/// `step` is the only way this executor advances, so no driver can reach the end
/// of an invocation without passing through it.
///
/// # Errors
///
/// Returns [`ReferenceError`] for uniform-control-flow violations, out-of-bounds
/// stores, malformed loops, expression evaluation failures, or an async transfer
/// still pending when the invocation ends.
pub fn step<'a>(
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &'a Program,
) -> Result<(), crate::ReferenceError> {
    if invocation.done() || invocation.waiting_at_barrier {
        return Ok(());
    }

    step_frames(invocation, memory, program)?;

    if invocation.done() {
        invocation.assert_async_drained()?;
    }
    Ok(())
}

fn step_frames<'a>(
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &'a Program,
) -> Result<(), crate::ReferenceError> {
    loop {
        let Some(frame) = invocation.frames_mut().pop() else {
            return Ok(());
        };
        match frame {
            Frame::Nodes {
                nodes,
                index,
                scoped,
            } => {
                if step_nodes_frame(invocation, memory, program, nodes, index, scoped)? {
                    return Ok(());
                }
            }
            Frame::Loop {
                var,
                next,
                to,
                body,
            } => step_loop_frame(invocation, var, next, to, body)?,
        }
    }
}

fn step_nodes_frame<'a>(
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &'a Program,
    nodes: &'a [Node],
    index: usize,
    scoped: bool,
) -> Result<bool, crate::ReferenceError> {
    if index >= nodes.len() {
        if scoped {
            invocation.pop_scope();
        }
        return Ok(false);
    }

    invocation.frames_mut().push(Frame::Nodes {
        nodes,
        index: index + 1,
        scoped,
    });
    execute_node(&nodes[index], invocation, memory, program)?;
    Ok(true)
}

fn step_loop_frame<'a>(
    invocation: &mut Invocation<'a>,
    var: &'a str,
    next: u32,
    to: u32,
    body: &'a [Node],
) -> Result<(), crate::ReferenceError> {
    if next >= to {
        return Ok(());
    }
    invocation.frames_mut().push(Frame::Loop {
        var,
        next: next.wrapping_add(1),
        to,
        body,
    });
    invocation.push_scope();
    invocation.bind_loop_var(var, crate::value::Value::U32(next))?;
    invocation.frames_mut().push(Frame::Nodes {
        nodes: body,
        index: 0,
        scoped: true,
    });
    Ok(())
}

fn execute_node<'a>(
    node: &'a Node,
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &'a Program,
) -> Result<(), crate::ReferenceError> {
    match node {
        Node::Let { name, value } => eval_let(name, value, invocation, memory, program),
        Node::Assign { name, value } => eval_assign(name, value, invocation, memory, program),
        Node::Store {
            buffer,
            index,
            value,
        } => eval_store(buffer, index, value, invocation, memory, program),
        Node::If {
            cond,
            then,
            otherwise,
        } => eval_if(cond, then, otherwise, node, invocation, memory, program),
        Node::Loop {
            var,
            from,
            to,
            body,
        } => eval_loop(var, from, to, body, invocation, memory, program),
        Node::Return => eval_return(invocation),
        Node::Block(nodes) => eval_block(nodes, invocation),
        Node::Barrier { .. } => eval_barrier(invocation),
        Node::IndirectDispatch {
            count_buffer,
            count_offset,
        } => eval_indirect_dispatch(count_buffer, *count_offset, memory, program),
        Node::AsyncLoad {
            source,
            destination,
            offset,
            size,
            tag,
        } => eval_async_load(
            AsyncLoadEval {
                source,
                destination,
                offset,
                size,
                tag,
            },
            invocation,
            memory,
            program,
        ),
        Node::AsyncStore {
            source,
            destination,
            offset,
            size,
            tag,
        } => eval_async_store(
            AsyncStoreEval {
                source,
                destination,
                offset,
                size,
                tag,
            },
            invocation,
            memory,
            program,
        ),
        Node::AsyncWait { tag } => eval_async_wait(tag, invocation, memory, program),
        Node::Trap { address, tag } => {
            let address = eval_expr::eval(address, invocation, memory, program)?
                .try_as_u32()
                .ok_or_else(|| {
                    ReferenceError::new(format!(
                        "reference trap `{tag}` address is not a u32. Fix: pass a scalar u32 trap address."
                    ))
                })?;
            Err(crate::ReferenceError::new(format!(
                "reference dispatch trapped: address={address}, tag=`{tag}`. Fix: handle the trap condition or route this Program through a backend/runtime with replay support."
            )))
        }
        Node::Resume { tag } => Err(crate::ReferenceError::new(format!(
            "reference dispatch reached Resume `{tag}` without a replay runtime. Fix: lower Resume through a runtime-owned replay path before reference execution."
        ))),
        Node::AllReduce { buffer, group, .. } => Err(crate::ReferenceError::new(format!(
            "reference dispatch reached AllReduce on buffer `{buffer}` for group {}. Fix: run this Program on a distributed backend with collective support or lower the single-rank collective before reference execution.",
            group.as_u32()
        ))),
        Node::AllGather {
            input,
            output,
            group,
        } => Err(crate::ReferenceError::new(format!(
            "reference dispatch reached AllGather `{input}` -> `{output}` for group {}. Fix: run this Program on a distributed backend with collective support or lower the single-rank collective before reference execution.",
            group.as_u32()
        ))),
        Node::ReduceScatter {
            input,
            output,
            group,
            ..
        } => Err(crate::ReferenceError::new(format!(
            "reference dispatch reached ReduceScatter `{input}` -> `{output}` for group {}. Fix: run this Program on a distributed backend with collective support or lower the single-rank collective before reference execution.",
            group.as_u32()
        ))),
        Node::Broadcast {
            buffer,
            root,
            group,
        } => Err(crate::ReferenceError::new(format!(
            "reference dispatch reached Broadcast on buffer `{buffer}` from root {root} for group {}. Fix: run this Program on a distributed backend with collective support or lower the single-rank collective before reference execution.",
            group.as_u32()
        ))),
        Node::Region { body, .. } => eval_block(body, invocation),
        Node::Opaque(extension) => Err(crate::ReferenceError::new(format!(
            "reference interpreter does not support opaque node extension `{}`/`{}`. Fix: provide a reference evaluator for this NodeExtension or lower it to core Node variants before evaluation.",
            extension.extension_kind(),
            extension.debug_identity()
        ))),
        Node::TileLoad {
            tile,
            tile_type,
            buffer,
            origin,
            layout,
        } => eval_tile_load(
            tile.as_str(),
            tile_type,
            buffer.as_str(),
            origin,
            layout,
            invocation,
            memory,
            program,
        ),
        Node::TileStore {
            buffer,
            origin,
            tile,
        } => eval_tile_store(
            buffer.as_str(),
            origin,
            tile.as_str(),
            invocation,
            memory,
            program,
        ),
        Node::TileMatmul { acc, a, b } => {
            eval_tile_matmul(acc.as_str(), a.as_str(), b.as_str(), invocation)
        }
        Node::TileReduce {
            out,
            tile,
            op,
            axis,
        } => eval_tile_reduce(out.as_str(), tile.as_str(), *op, *axis, invocation),
        Node::TileElementwise { out, inputs, body } => {
            eval_tile_elementwise(out.as_str(), inputs, body, invocation, memory, program)
        }
        Node::TileDecl { name, tile } => {
            let elements = vec![Value::Float(0.0); tile.element_count()];
            invocation.bind(name.as_str(), Value::Array(elements))
        }
        _ => Err(crate::ReferenceError::new(
            "reference interpreter encountered an unknown Node variant. Fix: update vyre-reference before executing this IR.",
        )),
    }
}

fn eval_let(
    name: &str,
    value: &Expr,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let value = eval_expr::eval(value, invocation, memory, program)?;
    invocation.bind(name, value)
}

fn eval_assign(
    name: &str,
    value: &Expr,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let value = eval_expr::eval(value, invocation, memory, program)?;
    invocation.assign(name, value)
}

fn eval_store(
    buffer: &str,
    index: &Expr,
    value: &Expr,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let index = eval_expr::eval(index, invocation, memory, program)?;
    let index = index
        .try_as_u32()
        .ok_or_else(|| ReferenceError::new(format!(
                "store index {index:?} cannot be represented as u32. Fix: use a non-negative scalar index within u32."
        )))?;
    let value = eval_expr::eval(value, invocation, memory, program)?;
    let target = eval_expr::buffer_mut(memory, program, buffer)?;
    oob::store(target, index, &value);
    Ok(())
}

fn eval_indirect_dispatch(
    count_buffer: &str,
    count_offset: u64,
    memory: &Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    if count_offset % 4 != 0 {
        return Err(ReferenceError::new(format!(
            "indirect dispatch offset {count_offset} is not 4-byte aligned. Fix: use a u32-aligned dispatch tuple."
        )));
    }
    let decl = program.buffer(count_buffer).ok_or_else(|| {
        ReferenceError::new(format!(
            "indirect dispatch references unknown buffer `{count_buffer}`. Fix: declare the count buffer before execution."
        ))
    })?;
    let buffer = if decl.access() == vyre_foundation::ir::BufferAccess::Workgroup {
        memory.workgroup.get(count_buffer)
    } else {
        memory.storage.get(count_buffer)
    }
    .ok_or_else(|| {
        ReferenceError::new(format!(
            "indirect dispatch buffer `{count_buffer}` is missing. Fix: initialize the count buffer before execution."
        ))
    })?;
    let required_end = count_offset.checked_add(12).ok_or_else(|| {
        ReferenceError::new(
            "indirect dispatch byte range overflowed u64. Fix: shrink the count offset."
                .to_string(),
        )
    })?;
    let byte_len = buffer
        .bytes
        .read()
        .map_err(|_| {
            ReferenceError::new(format!(
                "indirect dispatch buffer `{count_buffer}` lock is poisoned. Fix: rebuild the interpreter memory state before execution."
            ))
        })?
        .len();
    if u64::try_from(byte_len).unwrap_or(u64::MAX) < required_end {
        return Err(ReferenceError::new(format!(
            "indirect dispatch buffer `{count_buffer}` is too short for a 3-word dispatch tuple at byte offset {count_offset}. Fix: provide 12 readable bytes starting at that offset."
        )));
    }
    Err(ReferenceError::new(format!(
        "Node::IndirectDispatch cannot execute in the sequential reference interpreter because dynamic indirect dispatch requires runtime queue scheduling. Fix: run this program on a backend/runtime that supports indirect dispatch or lower `{count_buffer}` at byte offset {count_offset} to a static workgroup grid before reference execution."
    )))
}

struct AsyncLoadEval<'a> {
    source: &'a str,
    destination: &'a str,
    offset: &'a Expr,
    size: &'a Expr,
    tag: &'a str,
}

struct AsyncStoreEval<'a> {
    source: &'a str,
    destination: &'a str,
    offset: &'a Expr,
    size: &'a Expr,
    tag: &'a str,
}

fn eval_async_load(
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

fn eval_async_store(
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

fn eval_async_wait(
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

fn eval_if<'a>(
    cond: &Expr,
    then: &'a [Node],
    otherwise: &'a [Node],
    node: &Node,
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let cond_value = eval_expr::eval(cond, invocation, memory, program)?.truthy();
    if contains_barrier(then) || contains_barrier(otherwise) {
        invocation.uniform_checks.push((node_id(node), cond_value));
    }
    let branch = if cond_value { then } else { otherwise };
    invocation.push_scope();
    invocation.frames_mut().push(Frame::Nodes {
        nodes: branch,
        index: 0,
        scoped: true,
    });
    Ok(())
}

fn eval_loop<'a>(
    var: &'a str,
    from: &Expr,
    to: &Expr,
    body: &'a [Node],
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let from_value = eval_expr::eval(from, invocation, memory, program)?;
    let to_value = eval_expr::eval(to, invocation, memory, program)?;
    let from = from_value.try_as_u32().ok_or_else(|| {
        ReferenceError::new(format!(
                "loop lower bound {from_value:?} cannot be represented as u32. Fix: use an in-range unsigned loop bound."
        ))
    })?;
    let to = to_value.try_as_u32().ok_or_else(|| ReferenceError::new(format!(
            "loop upper bound {to_value:?} cannot be represented as u32. Fix: use an in-range unsigned loop bound."
    )))?;
    invocation.frames_mut().push(Frame::Loop {
        var,
        next: from,
        to,
        body,
    });
    Ok(())
}

fn eval_return(invocation: &mut Invocation<'_>) -> Result<(), crate::ReferenceError> {
    invocation.frames_mut().clear();
    invocation.returned = true;
    Ok(())
}

fn eval_block<'a>(
    nodes: &'a [Node],
    invocation: &mut Invocation<'a>,
) -> Result<(), crate::ReferenceError> {
    invocation.push_scope();
    invocation.frames_mut().push(Frame::Nodes {
        nodes,
        index: 0,
        scoped: true,
    });
    Ok(())
}

fn eval_barrier(invocation: &mut Invocation<'_>) -> Result<(), crate::ReferenceError> {
    invocation.waiting_at_barrier = true;
    Ok(())
}

// Inline: covers `crate::oob::Buffer`, which no integration test can reach, so a
// fixture cannot bind named buffers into `Memory` from outside the crate.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oob::Buffer;
    use crate::workgroup::InvocationIds;
    use vyre_foundation::ir::{BufferDecl, DataType};

    fn run_program(program: &Program, memory: &mut Memory) -> Result<(), crate::ReferenceError> {
        let mut invocation = Invocation::new(InvocationIds::ZERO, program.entry());
        while !invocation.done() {
            step(&mut invocation, memory, program)?;
        }
        Ok(())
    }

    fn bytes(memory: &Memory, name: &str) -> Vec<u8> {
        memory
            .storage
            .get(name)
            .expect("Fix: test buffer exists")
            .bytes
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    #[test]
    fn async_load_wait_copies_payload_into_destination() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("src", 0, DataType::Bytes).with_count(8),
                BufferDecl::output("dst", 1, DataType::Bytes).with_count(8),
            ],
            [1, 1, 1],
            vec![
                Node::async_load_gpu_driven("src", "dst", Expr::u32(2), Expr::u32(4), "copy"),
                Node::AsyncWait { tag: "copy".into() },
            ],
        );
        let mut memory = Memory::empty()
            .with_storage(
                "src",
                Buffer::new(vec![10, 11, 12, 13, 14, 15, 16, 17], DataType::Bytes),
            )
            .with_storage("dst", Buffer::new(vec![0; 8], DataType::Bytes));

        run_program(&program, &mut memory).unwrap();

        assert_eq!(bytes(&memory, "dst"), vec![12, 13, 14, 15, 0, 0, 0, 0]);
    }

    #[test]
    fn async_store_wait_copies_payload_at_destination_offset() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("src", 0, DataType::Bytes).with_count(4),
                BufferDecl::output("dst", 1, DataType::Bytes).with_count(8),
            ],
            [1, 1, 1],
            vec![
                Node::async_store("src", "dst", Expr::u32(3), Expr::u32(4), "store"),
                Node::AsyncWait {
                    tag: "store".into(),
                },
            ],
        );
        let mut memory = Memory::empty()
            .with_storage("src", Buffer::new(vec![21, 22, 23, 24], DataType::Bytes))
            .with_storage("dst", Buffer::new(vec![0; 8], DataType::Bytes));

        run_program(&program, &mut memory).unwrap();

        assert_eq!(bytes(&memory, "dst"), vec![0, 0, 0, 21, 22, 23, 24, 0]);
    }

    #[test]
    fn sequential_eval_tile_matmul_and_reduce() {
        use vyre_foundation::ir::{BufferAccess, Layout, Residency, SubgroupReduceOp, Tile};

        let tile_reg = Tile::new(
            DataType::F32,
            vec![2, 2],
            Layout::RowMajor,
            Residency::Register,
        );
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
                BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::F32).with_count(4),
                BufferDecl::output("out", 2, DataType::F32).with_count(4),
                BufferDecl::output("red", 3, DataType::F32).with_count(2),
            ],
            [1, 1, 1],
            vec![
                Node::tile_decl("c", tile_reg.clone()),
                Node::tile_load(
                    "t_a",
                    tile_reg.clone(),
                    "a",
                    vec![Expr::u32(0), Expr::u32(0)],
                    Layout::RowMajor,
                ),
                Node::tile_load(
                    "t_b",
                    tile_reg,
                    "b",
                    vec![Expr::u32(0), Expr::u32(0)],
                    Layout::RowMajor,
                ),
                Node::tile_matmul("c", "t_a", "t_b"),
                Node::tile_store("out", vec![Expr::u32(0), Expr::u32(0)], "c"),
                Node::tile_reduce("r", "c", SubgroupReduceOp::Add, 1),
                Node::tile_store("red", vec![Expr::u32(0)], "r"),
            ],
        );

        let mut a_bytes = Vec::new();
        for &v in &[1.0f32, 2.0, 3.0, 4.0] {
            a_bytes.extend_from_slice(&v.to_ne_bytes());
        }
        let mut b_bytes = Vec::new();
        for &v in &[5.0f32, 6.0, 7.0, 8.0] {
            b_bytes.extend_from_slice(&v.to_ne_bytes());
        }

        let mut memory = Memory::empty()
            .with_storage("a", Buffer::new(a_bytes, DataType::F32))
            .with_storage("b", Buffer::new(b_bytes, DataType::F32))
            .with_storage("out", Buffer::new(vec![0; 16], DataType::F32))
            .with_storage("red", Buffer::new(vec![0; 8], DataType::F32));

        run_program(&program, &mut memory).unwrap();

        let out_bytes = bytes(&memory, "out");
        let out_f32: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(out_f32, vec![19.0, 22.0, 43.0, 50.0]);

        let red_bytes = bytes(&memory, "red");
        let red_f32: Vec<f32> = red_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(red_f32, vec![41.0, 93.0]);
    }

    /// WHY: the reference tree must not hold two verdicts for one program. Both
    /// executors reach the three async rules (a tag started twice, a wait with
    /// nothing pending, a transfer still queued at the end of an invocation)
    /// through `execution::async_transfer::PendingAsyncTransfers`. The third one
    /// used to differ: the hashmap interpreter refused it and this executor
    /// dropped it, so the same program was accepted by one path and rejected by
    /// the other. A transfer nobody waited on means the invocation's result
    /// depends on bytes nobody synchronized, so both refuse.
    ///
    /// Closes: an async start node whose wait never runs, on both transitions to
    /// done (frames exhausted, and `Node::Return`), for every `Async*` variant
    /// the IR declares, with the two executors held to one message.
    ///
    /// Does not catch: a wait that runs on some paths and not others. Only the
    /// lane that skipped the wait is refused here, so a program whose skipping
    /// branch is unreachable for the fixture's inputs still passes. The static
    /// form of the rule in `vyre-foundation` validation is what covers that.
    mod pending_async_transfers {
        use super::*;
        use crate::value::Value;

        const PENDING_TAG: &str = "pending_copy";

        /// Whether an async node kind STARTS a transfer, so the
        /// end-of-invocation rule must fire for it, or only OBSERVES one.
        #[derive(Debug, PartialEq, Eq)]
        enum AsyncRole {
            StartsTransfer,
            ObservesTransfer,
        }

        const ASYNC_NODE_ROLES: &[(&str, AsyncRole)] = &[
            ("AsyncLoad", AsyncRole::StartsTransfer),
            ("AsyncStore", AsyncRole::StartsTransfer),
            ("AsyncWait", AsyncRole::ObservesTransfer),
        ];

        /// Every `Async*` variant the IR declares.
        ///
        /// Read from `NODE_VARIANT_NAMES`, the registry the `Node` declaration
        /// itself emits, so adding an async variant turns this suite RED until
        /// someone classifies it and gives it a fixture.
        fn declared_async_variants() -> Vec<&'static str> {
            let mut names: Vec<&'static str> = vyre_foundation::ir::NODE_VARIANT_NAMES
                .iter()
                .copied()
                .filter(|name| name.starts_with("Async"))
                .collect();
            names.sort_unstable();
            names
        }

        fn async_start_node(kind: &str) -> Node {
            match kind {
                "AsyncLoad" => Node::async_load_gpu_driven(
                    "src",
                    "dst",
                    Expr::u32(0),
                    Expr::u32(4),
                    PENDING_TAG,
                ),
                "AsyncStore" => {
                    Node::async_store("src", "dst", Expr::u32(0), Expr::u32(4), PENDING_TAG)
                }
                other => panic!(
                    "Fix: classify `Node::{other}` in ASYNC_NODE_ROLES and give it a fixture in async_start_node."
                ),
            }
        }

        /// One invocation, so the ids in a refusal are `InvocationIds::ZERO` on
        /// either executor and the two messages are comparable verbatim.
        fn single_lane_program(body: Vec<Node>) -> Program {
            Program::wrapped(
                vec![
                    BufferDecl::read("src", 0, DataType::U32).with_count(1),
                    BufferDecl::output("dst", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                body,
            )
        }

        fn statement_executor(program: &Program) -> Result<(), crate::ReferenceError> {
            let validation_report = vyre_foundation::validate::validate_with_options(
                program,
                vyre_foundation::validate::ValidationOptions::default(),
            );
            if let Some(source) = validation_report.errors.into_iter().next() {
                return Err(crate::ReferenceError::validation(source));
            }
            let mut memory = Memory::empty()
                .with_storage("src", Buffer::new(vec![1, 2, 3, 4], DataType::U32))
                .with_storage("dst", Buffer::new(vec![0; 4], DataType::U32));
            run_program(program, &mut memory)
        }

        fn hashmap_executor(program: &Program) -> Result<(), crate::ReferenceError> {
            let inputs = vec![Value::from(vec![1_u8, 2, 3, 4]), Value::from(vec![0_u8; 4])];
            crate::reference_eval(program, &inputs).map(|_| ())
        }

        fn refusal(
            executor: fn(&Program) -> Result<(), crate::ReferenceError>,
            program: &Program,
        ) -> String {
            executor(program)
                .expect_err(
                    "Fix: an async transfer nobody waited on must refuse, not be dropped silently.",
                )
                .to_string()
        }

        #[test]
        fn every_async_variant_the_ir_declares_is_classified() {
            let classified = {
                let mut names: Vec<&str> = ASYNC_NODE_ROLES.iter().map(|(name, _)| *name).collect();
                names.sort_unstable();
                names
            };
            assert_eq!(
                declared_async_variants(),
                classified,
                "Fix: classify every declared `Async*` node variant in ASYNC_NODE_ROLES; an unclassified one has no end-of-invocation coverage."
            );
        }

        #[test]
        fn a_transfer_left_pending_refuses_when_the_frames_run_out() {
            for (kind, _) in ASYNC_NODE_ROLES
                .iter()
                .filter(|(_, role)| *role == AsyncRole::StartsTransfer)
            {
                let program = single_lane_program(vec![async_start_node(kind)]);
                let message = refusal(statement_executor, &program);
                assert!(
                    (message.contains("still pending") || message.contains("in flight"))
                        && message.contains(PENDING_TAG)
                        && message.contains("AsyncWait"),
                    "Fix: `Node::{kind}` left pending must name the tag and the missing AsyncWait, got: {message}"
                );
            }
        }

        #[test]
        fn a_transfer_left_pending_refuses_when_the_invocation_returns_early() {
            for (kind, _) in ASYNC_NODE_ROLES
                .iter()
                .filter(|(_, role)| *role == AsyncRole::StartsTransfer)
            {
                let program = single_lane_program(vec![async_start_node(kind), Node::Return]);
                let message = refusal(statement_executor, &program);
                assert!(
                    (message.contains("still pending") || message.contains("in flight"))
                        && message.contains(PENDING_TAG)
                        && message.contains("AsyncWait"),
                    "Fix: `Return` must not launder a pending `Node::{kind}` past the check, got: {message}"
                );
            }
        }

        #[test]
        fn both_executors_refuse_a_pending_transfer_with_one_message() {
            for (kind, _) in ASYNC_NODE_ROLES
                .iter()
                .filter(|(_, role)| *role == AsyncRole::StartsTransfer)
            {
                let program = single_lane_program(vec![async_start_node(kind)]);
                assert_eq!(
                    refusal(statement_executor, &program),
                    refusal(hashmap_executor, &program),
                    "Fix: the reference tree must not hold two verdicts for one program; `Node::{kind}` differs by executor."
                );
            }
        }

        #[test]
        fn a_waited_transfer_is_accepted_by_both_executors() {
            for (kind, _) in ASYNC_NODE_ROLES
                .iter()
                .filter(|(_, role)| *role == AsyncRole::StartsTransfer)
            {
                let program = single_lane_program(vec![
                    async_start_node(kind),
                    Node::async_wait(PENDING_TAG),
                    Node::Return,
                ]);
                statement_executor(&program).unwrap_or_else(|error| {
                    panic!("Fix: a waited `Node::{kind}` must still be accepted, got: {error}")
                });
                hashmap_executor(&program).unwrap_or_else(|error| {
                    panic!("Fix: a waited `Node::{kind}` must still be accepted, got: {error}")
                });
            }
        }
    }
}
fn eval_tile_load(
    tile_name: &str,
    tile_type: &vyre_foundation::ir::Tile,
    buffer: &str,
    origin: &[Expr],
    layout: &vyre_foundation::ir::Layout,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let mut origin_coords = Vec::with_capacity(origin.len());
    for expr in origin {
        let v = eval_expr::eval(expr, invocation, memory, program)?;
        let coord = v.try_as_u32().ok_or_else(|| {
            crate::ReferenceError::new("tile load origin coord must be u32".to_string())
        })?;
        origin_coords.push(coord);
    }
    let target = eval_expr::buffer(memory, program, buffer)?;
    let elements = crate::execution::tile::load_elements(target, &origin_coords, tile_type, layout);
    invocation.bind(tile_name, Value::Array(elements))
}

fn eval_tile_store(
    buffer: &str,
    origin: &[Expr],
    tile_name: &str,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), crate::ReferenceError> {
    let mut origin_coords = Vec::with_capacity(origin.len());
    for expr in origin {
        let v = eval_expr::eval(expr, invocation, memory, program)?;
        let coord = v.try_as_u32().ok_or_else(|| {
            crate::ReferenceError::new("tile store origin coord must be u32".to_string())
        })?;
        origin_coords.push(coord);
    }
    let tile_val = invocation.local(tile_name).ok_or_else(|| {
        crate::ReferenceError::new(format!(
            "tile `{tile_name}` not found in scope for tile store"
        ))
    })?;
    let elements = match tile_val {
        Value::Array(elems) => elems.clone(),
        single => vec![single.clone()],
    };
    let target = eval_expr::buffer_mut(memory, program, buffer)?;
    crate::execution::tile::store_elements(target, &origin_coords, &elements);
    Ok(())
}

fn eval_tile_matmul(
    acc_name: &str,
    a_name: &str,
    b_name: &str,
    invocation: &mut Invocation<'_>,
) -> Result<(), crate::ReferenceError> {
    let acc_val = invocation
        .local(acc_name)
        .cloned()
        .unwrap_or(Value::Array(Vec::new()));
    let a_val = invocation.local(a_name).cloned().ok_or_else(|| {
        crate::ReferenceError::new(format!("tile `{a_name}` not found for matmul"))
    })?;
    let b_val = invocation.local(b_name).cloned().ok_or_else(|| {
        crate::ReferenceError::new(format!("tile `{b_name}` not found for matmul"))
    })?;
    let a_elems = crate::execution::tile::to_elements(&a_val);
    let b_elems = crate::execution::tile::to_elements(&b_val);
    let mut acc_elems = crate::execution::tile::to_elements(&acc_val);
    crate::execution::tile::matmul(&mut acc_elems, &a_elems, &b_elems);
    invocation.assign(acc_name, Value::Array(acc_elems))
}

fn eval_tile_reduce(
    out_name: &str,
    tile_name: &str,
    op: vyre_foundation::ir::SubgroupReduceOp,
    axis: u32,
    invocation: &mut Invocation<'_>,
) -> Result<(), crate::ReferenceError> {
    let tile_val = invocation.local(tile_name).cloned().ok_or_else(|| {
        crate::ReferenceError::new(format!("tile `{tile_name}` not found for reduce"))
    })?;
    let elements = match tile_val {
        Value::Array(e) => e,
        s => vec![s],
    };
    let res = crate::execution::tile::reduce(&elements, op, axis);
    invocation.bind(out_name, Value::Array(res))
}

fn eval_tile_elementwise<'a>(
    out_name: &str,
    inputs: &[vyre_foundation::ir::Ident],
    body: &'a [Node],
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &'a Program,
) -> Result<(), crate::ReferenceError> {
    let mut input_arrays = Vec::with_capacity(inputs.len());
    let mut max_len = 0;
    let mut saved_inputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let val = invocation
            .local(input.as_str())
            .cloned()
            .ok_or_else(|| crate::ReferenceError::new(format!("tile input `{input}` not found")))?;
        let elems = match &val {
            Value::Array(e) => e.clone(),
            s => vec![s.clone()],
        };
        max_len = max_len.max(elems.len());
        input_arrays.push(elems);
        saved_inputs.push(val);
    }
    for input in inputs {
        invocation.unbind(input.as_str());
    }
    let mut out_elems = Vec::with_capacity(max_len);
    for idx in 0..max_len {
        invocation.push_scope();
        for (i, input) in inputs.iter().enumerate() {
            let elem = input_arrays[i]
                .get(idx)
                .cloned()
                .unwrap_or(Value::Float(0.0));
            invocation.bind(input.as_str(), elem)?;
        }
        for node in body {
            execute_node(node, invocation, memory, program)?;
        }
        let out_val = invocation
            .local(out_name)
            .cloned()
            .unwrap_or(Value::Float(0.0));
        out_elems.push(out_val);
        invocation.pop_scope();
    }
    for (input, val) in inputs.iter().zip(saved_inputs) {
        let _ = invocation.bind(input.as_str(), val);
    }
    invocation.bind(out_name, Value::Array(out_elems))
}
