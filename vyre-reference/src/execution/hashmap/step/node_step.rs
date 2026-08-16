#[cfg(feature = "subgroup-ops")]
use super::super::invocation::HashmapInvocationSnapshot;
use super::super::{
    eval_expr,
    invocation::HashmapInvocation,
    memory::{buffer_mut, HashmapMemory},
    sync::{contains_barrier, node_id},
};
use crate::execution::async_transfer::{self, AsyncTransfer};
use crate::execution::call::{callable_signature, invoke_signature, resolve_call};
use crate::ReferenceError;
use crate::{oob, value::Value, workgroup::Frame};
use vyre_foundation::ir::{Expr, Node};

pub(crate) fn step_nodes_frame<'a>(
    invocation: &mut HashmapInvocation<'a>,
    memory: &mut HashmapMemory,
    nodes: &'a [Node],
    index: usize,
    scoped: bool,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<bool, ReferenceError> {
    if index >= nodes.len() {
        if scoped {
            invocation.locals.pop_scope();
        }
        return Ok(scoped);
    }
    invocation.frames.push(Frame::Nodes {
        nodes,
        index: index + 1,
        scoped,
    });
    let node = &nodes[index];
    match node {
        Node::Let { name, value } => {
            let v = eval_expr(
                value,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            invocation.locals.bind(name, v)?;
        }
        Node::Assign { name, value } => {
            let v = eval_expr(
                value,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            invocation.locals.assign(name, v)?;
        }
        Node::Store {
            buffer,
            index,
            value,
        } => {
            let idx = eval_expr (index , invocation , memory , #[cfg (feature = "subgroup-ops")] snapshots ,) ? . try_as_u32 () . ok_or_else (| | { ReferenceError::new("store index cannot be represented as u32. Fix: use a non-negative scalar index within u32.") }) ? ;
            let v = eval_expr(
                value,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            let target = buffer_mut(memory, buffer)?;
            oob::store(target, idx, &v);
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let cond_value = eval_expr(
                cond,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?
            .truthy();
            if contains_barrier(then) || contains_barrier(otherwise) {
                invocation.uniform_checks.push((node_id(node), cond_value));
            }
            let branch = if cond_value { then } else { otherwise };
            invocation.locals.push_scope();
            invocation.frames.push(Frame::Nodes {
                nodes: branch,
                index: 0,
                scoped: true,
            });
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            let from_value = eval_expr (from , invocation , memory , #[cfg (feature = "subgroup-ops")] snapshots ,) ? . try_as_u32 () . ok_or_else (| | { ReferenceError::new("loop lower bound cannot be represented as u32. Fix: use an in-range unsigned loop bound.") }) ? ;
            let to_value = eval_expr (to , invocation , memory , #[cfg (feature = "subgroup-ops")] snapshots ,) ? . try_as_u32 () . ok_or_else (| | { ReferenceError::new("loop upper bound cannot be represented as u32. Fix: use an in-range unsigned loop bound.") }) ? ;
            invocation.frames.push(Frame::Loop {
                var,
                next: from_value,
                to: to_value,
                body,
            });
        }
        Node::Return => {
            invocation.frames.clear();
            invocation.returned = true;
        }
        Node::Block(nodes) => {
            invocation.locals.push_scope();
            invocation.frames.push(Frame::Nodes {
                nodes,
                index: 0,
                scoped: true,
            });
        }
        Node::Barrier { .. } => {
            invocation.waiting_at_barrier = true;
        }
        Node::IndirectDispatch {
            count_buffer,
            count_offset,
        } => {
            let count_offset = u32::try_from(*count_offset).map_err(|_| {
                ReferenceError::new(format!(
                    "indirect dispatch count offset {count_offset} exceeds u32. Fix: keep indirect dispatch offsets within the reference interpreter index domain."
                ))
            })?;
            eval_indirect_dispatch(count_buffer, count_offset, memory)?;
        }
        Node::AsyncLoad {
            source,
            destination,
            offset,
            size,
            tag,
        } => {
            let transfer = eval_async_load(
                source,
                destination,
                offset,
                size,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            invocation.begin_async(tag, transfer)?;
        }
        Node::AsyncStore {
            source,
            destination,
            offset,
            size,
            tag,
        } => {
            let transfer = eval_async_store(
                source,
                destination,
                offset,
                size,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            invocation.begin_async(tag, transfer)?;
        }
        Node::AsyncWait { tag } => {
            apply_async_transfer(invocation.finish_async(tag)?, memory)?;
        }
        Node::Trap { address, tag } => {
            let address = eval_expr(
                address,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?
            .try_as_u32()
            .ok_or_else(|| {
                ReferenceError::new(format!(
                    "reference trap `{tag}` address is not a u32. Fix: pass a scalar u32 trap address."
                ))
            })?;
            return Err(ReferenceError::new(format!(
                "reference dispatch trapped: address={address}, tag=`{tag}`. Fix: handle the trap condition or route this Program through a backend/runtime with replay support."
            )));
        }
        Node::Resume { tag } => {
            return Err(ReferenceError::new(format!(
                "reference dispatch reached Resume `{tag}` without a replay runtime. Fix: lower Resume through a runtime-owned replay path before reference execution."
            )));
        }
        Node::AllReduce { buffer, group, .. } => {
            return Err(ReferenceError::new(format!(
                "hashmap reference interpreter reached AllReduce on buffer `{buffer}` for group {}. Fix: run this Program on a distributed backend with collective support or lower the single-rank collective before reference execution.",
                group.as_u32()
            )));
        }
        Node::AllGather {
            input,
            output,
            group,
        } => {
            return Err(ReferenceError::new(format!(
                "hashmap reference interpreter reached AllGather `{input}` -> `{output}` for group {}. Fix: run this Program on a distributed backend with collective support or lower the single-rank collective before reference execution.",
                group.as_u32()
            )));
        }
        Node::ReduceScatter {
            input,
            output,
            group,
            ..
        } => {
            return Err(ReferenceError::new(format!(
                "hashmap reference interpreter reached ReduceScatter `{input}` -> `{output}` for group {}. Fix: run this Program on a distributed backend with collective support or lower the single-rank collective before reference execution.",
                group.as_u32()
            )));
        }
        Node::Broadcast {
            buffer,
            root,
            group,
        } => {
            return Err(ReferenceError::new(format!(
                "hashmap reference interpreter reached Broadcast on buffer `{buffer}` from root {root} for group {}. Fix: run this Program on a distributed backend with collective support or lower the single-rank collective before reference execution.",
                group.as_u32()
            )));
        }
        Node::Region { body, .. } => {
            invocation.locals.push_scope();
            invocation.frames.push(Frame::Nodes {
                nodes: body,
                index: 0,
                scoped: true,
            });
        }
        Node::Opaque(extension) => {
            return Err(ReferenceError::new(format!(
                "hashmap reference interpreter does not support opaque node extension `{}`/`{}`. Fix: provide a reference evaluator for this NodeExtension or lower it to core Node variants before evaluation.",
                extension.extension_kind(),
                extension.debug_identity()
            )));
        }
        _ => {
            return Err(ReferenceError::new("hashmap reference interpreter encountered an unknown node variant. Fix: add explicit reference semantics for the new Node before dispatch."));
        }
    }
    Ok(true)
}

pub(crate) fn step_loop_frame<'a>(
    invocation: &mut HashmapInvocation<'a>,
    var: &'a str,
    next: u32,
    to: u32,
    body: &'a [Node],
) -> Result<(), ReferenceError> {
    if next >= to {
        return Ok(());
    }
    invocation.frames.push(Frame::Loop {
        var,
        next: next.wrapping_add(1),
        to,
        body,
    });
    invocation.locals.push_scope();
    invocation.locals.bind_loop_var(var, Value::U32(next))?;
    invocation.frames.push(Frame::Nodes {
        nodes: body,
        index: 0,
        scoped: true,
    });
    Ok(())
}

pub(crate) fn eval_call(
    expr: *const Expr,
    op_id: &str,
    inputs: &[Expr],
    invocation: &mut HashmapInvocation<'_>,
    memory: &mut HashmapMemory,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<Value, ReferenceError> {
    let resolved = resolve_call(expr, op_id, &mut invocation.op_cache)?;
    let signature = callable_signature(op_id, &resolved.operation)?;
    invoke_signature(op_id, signature, inputs, |arg| {
        eval_expr(
            arg,
            invocation,
            memory,
            #[cfg(feature = "subgroup-ops")]
            snapshots,
        )
    })
}

fn eval_indirect_dispatch(
    count_buffer: &str,
    count_offset: u32,
    _memory: &HashmapMemory,
) -> Result<(), ReferenceError> {
    Err(ReferenceError::new(format!(
        "Node::IndirectDispatch cannot execute in the hashmap reference interpreter because dynamic indirect dispatch requires runtime queue scheduling. Fix: run this program on a backend/runtime that supports indirect dispatch or lower `{count_buffer}` at byte offset {count_offset} to a static workgroup grid before reference execution."
    )))
}

fn eval_async_load(
    source: &str,
    destination: &str,
    offset: &Expr,
    size: &Expr,
    invocation: &mut HashmapInvocation<'_>,
    memory: &mut HashmapMemory,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<AsyncTransfer, ReferenceError> {
    let start = eval_byte_count(
        offset,
        "async load source offset",
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    let byte_count = eval_byte_count(
        size,
        "async load size",
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    let payload = read_bytes(memory, source, start, byte_count)?;
    ensure_buffer_exists(memory, destination)?;
    Ok(AsyncTransfer::load(destination, payload))
}

fn eval_async_store(
    source: &str,
    destination: &str,
    offset: &Expr,
    size: &Expr,
    invocation: &mut HashmapInvocation<'_>,
    memory: &mut HashmapMemory,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<AsyncTransfer, ReferenceError> {
    let start = eval_byte_count(
        offset,
        "async store destination offset",
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    let byte_count = eval_byte_count(
        size,
        "async store size",
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    let payload = read_bytes(memory, source, 0, byte_count)?;
    ensure_buffer_exists(memory, destination)?;
    Ok(AsyncTransfer::store(destination, start, payload))
}

fn eval_byte_count(
    expr: &Expr,
    label: &str,
    invocation: &mut HashmapInvocation<'_>,
    memory: &mut HashmapMemory,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<usize, ReferenceError> {
    let value = eval_expr(
        expr,
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    async_transfer::byte_count(&value, label)
}

fn read_bytes(
    memory: &HashmapMemory,
    source: &str,
    start: usize,
    byte_count: usize,
) -> Result<Vec<u8>, ReferenceError> {
    Ok(super::super::memory::resolve_buffer(memory, source)?.read_window(start, byte_count))
}

fn ensure_buffer_exists(memory: &HashmapMemory, name: &str) -> Result<(), ReferenceError> {
    super::super::memory::resolve_buffer(memory, name).map(|_| ())
}

fn apply_async_transfer(
    transfer: AsyncTransfer,
    memory: &mut HashmapMemory,
) -> Result<(), ReferenceError> {
    let buffer = buffer_mut(memory, transfer.destination())?;
    transfer.apply_to(buffer);
    Ok(())
}

// Inline: covers the crate-private `apply_async_transfer` and `read_bytes`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::super::super::memory::HashmapMemory;
    use crate::oob::Buffer;
    use rustc_hash::FxHashMap;
    use vyre_foundation::ir::DataType;

    /// Poisons the `Arc<RwLock<Vec<u8>>>` inside a `Buffer` by taking the write
    /// lock in a thread and panicking before releasing it, then confirms that the
    /// fixed `read_bytes` helper fails closed (panics) instead of silently
    /// recovering the half-mutated guard via `into_inner()`.
    ///
    /// Before the VRH-001 fix this test would NOT panic (the recovery path
    /// returned a corrupt guard and the function returned `Ok`).  After the fix
    /// the call to `read_bytes` propagates the poison panic.
    #[test]
    fn read_bytes_fails_closed_on_poisoned_buffer_lock() {
        let buffer = Buffer::new(vec![0xab_u8; 8], DataType::U32);
        let poisoner = buffer.bytes.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.write().unwrap();
            panic!("VRH-001: poison read lock mid-write");
        })
        .join();

        let mut storage = FxHashMap::default();
        storage.insert("src".to_string(), buffer);
        let memory = HashmapMemory::new(storage);

        let result = std::panic::catch_unwind(|| {
            // `read_bytes` acquires buffer.bytes.read(); it must panic, not recover.
            super::read_bytes(&memory, "src", 0, 4)
        });
        assert!(
            result.is_err(),
            "Fix: read_bytes must panic on a poisoned buffer lock, not silently recover with into_inner()"
        );
        let payload = result.unwrap_err();
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("<non-string panic>");
        assert!(
            message.contains("reference Buffer byte lock was poisoned"),
            "Fix: panic message must name the poisoned lock contract, got: {message}"
        );
    }

    /// Mirrors `read_bytes_fails_closed_on_poisoned_buffer_lock` for the write
    /// path inside `apply_async_transfer`.
    #[test]
    fn apply_async_transfer_fails_closed_on_poisoned_buffer_lock() {
        use super::AsyncTransfer;
        let buffer = Buffer::new(vec![0xcd_u8; 8], DataType::U32);
        let poisoner = buffer.bytes.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.write().unwrap();
            panic!("VRH-001: poison write lock mid-async-copy");
        })
        .join();

        let mut storage = FxHashMap::default();
        storage.insert("dst".to_string(), buffer);
        let mut memory = HashmapMemory::new(storage);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::apply_async_transfer(
                AsyncTransfer::store("dst", 0, vec![0x11, 0x22, 0x33, 0x44]),
                &mut memory,
            )
        }));
        assert!(
            result.is_err(),
            "Fix: apply_async_transfer must panic on a poisoned buffer lock, not silently recover with into_inner()"
        );
        let payload = result.unwrap_err();
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("<non-string panic>");
        assert!(
            message.contains("reference Buffer byte lock was poisoned"),
            "Fix: panic message must name the poisoned lock contract, got: {message}"
        );
    }
}
