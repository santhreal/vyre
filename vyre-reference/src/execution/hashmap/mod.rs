//! HashMap-backed reference interpreter split into execution, state, memory,
//! synchronization, and optional subgroup semantics.
//!
//! This root module owns expression evaluation and the split modules own their
//! state, memory, execution, synchronization, and subgroup contracts.

pub(crate) mod memory;
pub(crate) mod state;
pub(crate) mod step;
pub(crate) mod subgroup;
pub(crate) mod sync;

use memory::{atomic_buffer_mut, output_value, resolve_buffer, HashmapMemory};
#[cfg(feature = "subgroup-ops")]
use state::HashmapInvocationSnapshot;
use state::{create_invocations, run_invocations, HashmapInvocation};
use step::{axis_value, eval_call, eval_to_index};
#[cfg(feature = "subgroup-ops")]
use subgroup::{eval_subgroup_ballot, eval_subgroup_reduce, eval_subgroup_shuffle};
use sync::element_count;

use crate::{
    atomics,
    oob::{self, Buffer},
    value::Value,
};
use rustc_hash::FxHashMap;
use vyre::ir::{AtomicOp, BufferAccess, Expr, MemoryOrdering, Node, Program};
use vyre::Error;

/// Order in which the interpreter steps workgroups and the invocations within
/// each workgroup.
///
/// The GPU makes NO ordering guarantee across invocations for NON-atomic stores:
/// two lanes that plain-`store` the same slot leave a driver-defined winner. The
/// single-threaded reference resolves that race DETERMINISTICALLY (last stepped
/// lane wins), which HIDES the hazard, the output looks stable here but is
/// nondeterministic on real hardware. Running the identical dispatch once
/// [`Forward`](LaneOrder::Forward) and once [`Reversed`](LaneOrder::Reversed) and
/// comparing outputs surfaces it: a race-free program (disjoint output slots, or
/// commutative atomics for any shared slot) is order-invariant; a program with a
/// non-atomic cross-lane write-write conflict produces a DIFFERENT result, exactly
/// the way it would nondeterministically diverge across GPU runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneOrder {
    /// The canonical order: workgroups `0..N`, invocations in `create_invocations`
    /// (z,y,x-nested linear) order. Byte-for-byte the interpreter's original path.
    Forward,
    /// Workgroups and intra-workgroup invocations both stepped in reverse. Only the
    /// STEPPING order changes; every invocation keeps its true global/local ids.
    Reversed,
}

/// A `MemoryOrdering::GridSync` barrier, the grid-wide fence `fuse_programs` inserts
/// between arms whose later arm reads an earlier arm's cross-workgroup
/// (launch-geometry) output. On real hardware the driver lowers it into separate
/// globally-ordered dispatch segments; the reference interpreter mirrors that by
/// advancing the whole grid through one segment before the next.
fn is_grid_sync_barrier(node: &Node) -> bool {
    matches!(
        node,
        Node::Barrier {
            ordering: MemoryOrdering::GridSync
        }
    )
}

/// Whether a GridSync barrier appears anywhere in the SEQUENTIAL scope tree, the
/// top level or nested inside transparent `Block` / `Region` scopes. It does NOT
/// descend into data-dependent control flow (`If` / `Loop`), where a grid-wide fence
/// is ill-defined and fusion never emits one.
fn contains_grid_sync(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match node {
        Node::Block(inner) => contains_grid_sync(inner),
        Node::Region { body, .. } => contains_grid_sync(body),
        other => is_grid_sync_barrier(other),
    })
}

/// Flatten every transparent scope (`Block` / `Region`) that CONTAINS a GridSync so
/// each GridSync becomes a top-level node ready to split on. A re-fused program (e.g.
/// the exclusive scan = fuse(inclusive-chain, subtract)) nests the inner chain's
/// A→B / B→C GridSyncs one arm-scope deeper, so a single-level unwrap misses them.
/// Scopes WITHOUT a GridSync are kept intact (their locals keep their own scope);
/// only GridSync-carrying wrappers are dissolved, and post-fusion arm names are
/// already globally unique, so dropping such a wrapper's scope cannot collide.
fn flatten_grid_sync_scopes(nodes: &[Node], out: &mut Vec<Node>) {
    for node in nodes {
        match node {
            Node::Block(inner) if contains_grid_sync(inner) => {
                flatten_grid_sync_scopes(inner, out);
            }
            Node::Region { body, .. } if contains_grid_sync(body) => {
                flatten_grid_sync_scopes(body, out);
            }
            other => out.push(other.clone()),
        }
    }
}

/// Split a flattened body (all GridSyncs top-level) into execution segments at each
/// GridSync barrier. Running the ENTIRE grid through segment `k` before any
/// workgroup enters segment `k+1` reproduces the driver's dispatch split and makes
/// GridSync globally ordered (fixes multi-block prefix-scan Pass-B reading Pass-A's
/// per-block totals, and the same shape in every fused multi-pass kernel).
fn split_top_level_grid_sync(nodes: &[Node]) -> Vec<&[Node]> {
    let mut segments = Vec::new();
    let mut start = 0;
    for (index, node) in nodes.iter().enumerate() {
        if is_grid_sync_barrier(node) {
            segments.push(&nodes[start..index]);
            start = index + 1;
        }
    }
    segments.push(&nodes[start..]);
    segments
}

/// True when `reference_eval` RETURNS this buffer among its outputs. This is the SINGLE
/// source of truth for the interpreter's output ABI: `reference_eval` collects exactly
/// these decls, in `Program::buffers` order, into its result `Vec`. Test harnesses that
/// need the position of a named output MUST use [`output_index`] (which filters by this
/// predicate) rather than re-deriving the selection, a hand-rolled copy silently drifts
/// (e.g. keying on `is_pipeline_live_out` alone admits `ReadOnly` live-outs the
/// interpreter never returns, shifting every later index).
///
/// The "backend-allocated output" half is `BufferDecl::is_backend_allocated_output`, the
/// single cross-backend contract in vyre-foundation shared with the CpuRef/device
/// backends; this adds the interpreter's extra `ReadWrite` inputs-are-also-returned rule.
pub fn is_reference_output(decl: &vyre::ir::BufferDecl) -> bool {
    decl.is_backend_allocated_output() || decl.access() == BufferAccess::ReadWrite
}

/// Position of the buffer `name` within `reference_eval`'s returned outputs, the
/// buffers matching [`is_reference_output`], in `Program::buffers` order, or `None`
/// when the program declares no such returned output under that name.
pub fn output_index(program: &Program, name: &str) -> Option<usize> {
    program
        .buffers()
        .iter()
        .filter(|decl| is_reference_output(decl))
        .position(|decl| decl.name() == name)
}

#[doc = " Execute a vyre IR program using hashmap-backed locals."]
pub(crate) fn run_hashmap_reference(
    program: &Program,
    inputs: &[Value],
    min_dispatch_elements: u32,
    lane_order: LaneOrder,
) -> Result<Vec<Value>, Error> {
    #[cfg(feature = "subgroup-ops")]
    let validation_report = vyre::validate::validate::validate_with_options(
        program,
        vyre::validate::ValidationOptions::default().with_backend_capabilities(
            vyre::validate::BackendCapabilities {
                supports_subgroup_ops: true,
                ..Default::default()
            },
        ),
    );
    #[cfg(not(feature = "subgroup-ops"))]
    let validation_report = vyre::validate::validate::validate_with_options(
        program,
        vyre::validate::ValidationOptions::default(),
    );
    let validation_errors = validation_report.errors;
    if !validation_errors.is_empty() {
        let message_len = validation_errors
            .iter()
            .map(|error| error.message().len())
            .sum::<usize>()
            + validation_errors.len().saturating_sub(1) * 2;
        let mut messages = String::with_capacity(message_len);
        for (index, error) in validation_errors.iter().enumerate() {
            if index != 0 {
                messages.push_str("; ");
            }
            messages.push_str(error.message());
        }
        return Err(Error::interp(format!(
            "program failed IR validation: {messages}. Fix: repair the Program before invoking the reference interpreter."
        )));
    }
    let mut storage = FxHashMap::default();
    let logical_input_count = program
        .buffers()
        .iter()
        .filter(|decl| {
            decl.access() != BufferAccess::Workgroup && !decl.is_backend_allocated_output()
        })
        .count();
    let legacy_input_count = program
        .buffers()
        .iter()
        .filter(|decl| decl.access() != BufferAccess::Workgroup)
        .count();
    let legacy_input_mode =
        inputs.len() == legacy_input_count && inputs.len() != logical_input_count;
    let mut input_index = 0usize;
    let mut output_decls = Vec::new();
    let mut max_output_elements = 0u32;
    let mut max_input_elements = 1u32;
    let mut program_graph_node_count = None;
    let mut has_workgroup_buffer = false;
    for decl in program.buffers() {
        if decl.access() == BufferAccess::Workgroup {
            has_workgroup_buffer = true;
            continue;
        }
        if decl.binding() == 0 && decl.name() == "pg_nodes" {
            program_graph_node_count = Some(decl.count());
        }
        let required_bytes = declared_min_byte_len(decl)?;
        let backend_allocated = decl.is_backend_allocated_output();
        let bytes = if backend_allocated {
            if legacy_input_mode {
                let _legacy_output_initializer = inputs.get(input_index).ok_or_else(|| {
                    Error::interp(format!(
                        "missing legacy output initializer for buffer `{}`. Fix: pass one Value for each non-workgroup buffer or migrate to logical inputs only.",
                        decl.name()
                    ))
                })?;
                input_index += 1;
            }
            vec![0u8; required_bytes]
        } else {
            let value = inputs.get(input_index).ok_or_else(|| {
                Error::interp(format!(
                    "missing input for buffer `{}`. Fix: pass one Value for each non-output, non-workgroup buffer in Program::buffers order.",
                    decl.name()
                ))
            })?;
            input_index += 1;
            value.to_bytes()
        };
        if bytes.len() < required_bytes {
            return Err(Error::interp(format!(
                "buffer `{}` has {} bytes but requires at least {} bytes ({} elements of {}). Fix: provide a larger input buffer.",
                decl.name(),
                bytes.len(),
                required_bytes,
                decl.count(),
                decl.element()
            )));
        }
        let elements = element_count(decl, bytes.len())?;
        if is_reference_output(decl) {
            max_output_elements = max_output_elements.max(elements);
            output_decls.push(decl.clone());
        } else {
            max_input_elements = max_input_elements.max(elements);
        }
        storage.insert(
            decl.name().to_string(),
            Buffer::new(bytes, decl.element().clone()),
        );
    }
    if input_index != inputs.len() {
        return Err(Error::interp(
            "unused input values supplied. Fix: pass exactly one Value per non-workgroup buffer declaration.",
        ));
    }
    if program.workgroup_size().contains(&0) {
        return Err(Error::interp(
            "workgroup size contains zero. Fix: all dimensions must be >= 1.",
        ));
    }
    let [sx, sy, sz] = program.workgroup_size();
    let invocations_per_workgroup = [sx, sy, sz]
        .iter()
        .copied()
        .fold(1u32, u32::saturating_mul)
        .max(1);
    let force_full_span = has_workgroup_buffer || program.stats().atomic_op_count > 0;
    let dispatch_elements = max_output_elements
        .max(program_graph_node_count.unwrap_or(0))
        .max(1)
        .max(if output_decls.is_empty() || force_full_span {
            max_input_elements
        } else {
            1
        })
        // Caller-supplied grid floor. Buffer-shape inference cannot see the true
        // per-INVOCATION count of a byte-scan program: the haystack is packed 4
        // bytes/u32 and the scan length is a runtime VALUE (an input buffer of one
        // element), so a program that runs one invocation per haystack BYTE would
        // otherwise be under-dispatched to `haystack_len / 4` (or the largest
        // table) invocations and SILENTLY skip high positions. A caller that knows
        // the real grid (e.g. `haystack_len`) passes it here so the reference
        // interpreter covers exactly what the real dispatch would, no silent
        // under-coverage (Law 10).
        .max(min_dispatch_elements);
    let total_wg = dispatch_elements.div_ceil(invocations_per_workgroup).max(1);
    let active: Vec<usize> = [sx, sy, sz]
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
        .map(|(i, _)| i)
        .collect();
    let n = active.len().max(1);
    let mut counts = [1u32, 1, 1];
    if active.is_empty() {
        counts[0] = total_wg;
    } else {
        let base = (total_wg as f64).powf(1.0 / n as f64).ceil() as u32;
        for &axis in &active {
            counts[axis] = base.max(1);
        }
    }
    let [workgroup_count_x, workgroup_count_y, workgroup_count_z] = counts;
    let entry = program.entry();
    #[cfg(feature = "subgroup-ops")]
    let uses_subgroup_ops = vyre::program_caps::scan(program).subgroup_ops;
    // Grid-sync-aware execution: if the body carries `GridSync` barriers (a fused
    // multi-pass kernel the driver would split into ordered dispatches), run the
    // WHOLE grid through each inter-barrier segment before the next, so a later pass
    // never reads a prior pass's not-yet-written cross-workgroup output. GridSyncs
    // can nest inside transparent Block/Region scopes (a re-fused program buries an
    // inner chain's barriers an arm-scope deep), so flatten those scopes first. A
    // body with no GridSync keeps the exact single-segment path (`entry`), preserving
    // the original single-pass behavior byte-for-byte.
    let has_grid_sync = contains_grid_sync(entry);
    let flattened: Vec<Node> = if has_grid_sync {
        let mut nodes = Vec::new();
        flatten_grid_sync_scopes(entry, &mut nodes);
        nodes
    } else {
        Vec::new()
    };
    let segments: Vec<&[Node]> = if has_grid_sync {
        split_top_level_grid_sync(&flattened)
    } else {
        vec![entry]
    };
    // Canonical workgroup dispatch order (z,y,x-nested). `LaneOrder::Reversed`
    // steps this list, and the invocations within each workgroup, back to front
    // to flip the deterministic last-writer of any non-atomic same-slot store, so a
    // forward-vs-reversed output comparison surfaces a hidden cross-lane race (see
    // [`LaneOrder`]). Forward keeps the exact original nested-loop order.
    let mut wg_coords: Vec<[u32; 3]> = Vec::new();
    for wg_z in 0..workgroup_count_z {
        for wg_y in 0..workgroup_count_y {
            for wg_x in 0..workgroup_count_x {
                wg_coords.push([wg_x, wg_y, wg_z]);
            }
        }
    }
    if lane_order == LaneOrder::Reversed {
        wg_coords.reverse();
    }
    let mut memory = HashmapMemory::new(storage);
    for &segment in &segments {
        for &wg in &wg_coords {
            memory.reset_workgroup(program)?;
            let mut invocations = create_invocations(program, wg, segment)?;
            if lane_order == LaneOrder::Reversed {
                // Reverse the STEP order only; each invocation retains its true
                // global/local ids and linear_local_index (fields move with the
                // element), so semantics are unchanged for a race-free program.
                invocations.reverse();
            }
            run_invocations(
                &mut memory,
                &mut invocations,
                #[cfg(feature = "subgroup-ops")]
                uses_subgroup_ops,
            )?;
        }
    }
    let mut storage = memory.storage;
    output_decls . into_iter () . map (| decl | { storage . remove (decl . name ()) . map (| buffer | output_value (buffer , & decl)) . ok_or_else (| | { let name = decl . name () ; Error :: interp (format ! ("missing output buffer `{name}` after dispatch. Fix: keep buffer declarations unique.")) }) }) . collect ()
}

fn declared_min_byte_len(decl: &vyre::ir::BufferDecl) -> Result<usize, Error> {
    match decl.static_byte_len() {
        Ok(Some(byte_len)) => Ok(byte_len),
        Ok(None) if decl.count() == 0 => Ok(0),
        Ok(None) => Err(Error::interp(format!(
            "buffer `{}` has unsized element type {}. Fix: provide a fixed-width buffer element type before invoking the reference interpreter.",
            decl.name(),
            decl.element()
        ))),
        Err(error) => Err(Error::interp(error)),
    }
}

fn eval_expr(
    expr: &Expr,
    invocation: &mut HashmapInvocation<'_>,
    memory: &mut HashmapMemory,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<Value, Error> {
    match expr {
        Expr::LitU32(value) => Ok(Value::U32(*value)),
        Expr::LitI32(value) => Ok(Value::I32(*value)),
        Expr::LitF32(value) => Ok(Value::Float(f64::from(crate::execution::typed_ops::canonical_f32(
            *value,
        )))),
        Expr::LitBool(value) => Ok(Value::Bool(*value)),
        Expr::Var(name) => invocation.locals.local(name).ok_or_else(|| {
            Error::interp(format!(
                "reference to undeclared variable `{name}`. Fix: add a Let before this use."
            ))
        }),
        Expr::Load { buffer, index } => {
            let idx = eval_to_index(
                index,
                "load index",
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            Ok(oob::load(resolve_buffer(memory, buffer)?, idx))
        }
        Expr::BufLen { buffer } => Ok(Value::U32(resolve_buffer(memory, buffer)?.len())),
        Expr::InvocationId { axis } => axis_value(invocation.ids.global, *axis),
        Expr::WorkgroupId { axis } => axis_value(invocation.ids.workgroup, *axis),
        Expr::LocalId { axis } => axis_value(invocation.ids.local, *axis),
        Expr::BinOp { op, left, right } => {
            let left = eval_expr(
                left,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            let right = eval_expr(
                right,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            crate::execution::op_count::record_op();
            crate::execution::typed_ops::eval_binop(*op, left, right)
        }
        Expr::UnOp { op, operand } => {
            let operand = eval_expr(
                operand,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            crate::execution::op_count::record_op();
            crate::execution::typed_ops::eval_unop(op, operand)
        }
        Expr::Call { op_id, args } => eval_call(
            expr as *const Expr,
            op_id,
            args,
            invocation,
            memory,
            #[cfg(feature = "subgroup-ops")]
            snapshots,
        ),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            let cond = eval_expr(
                cond,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?
            .truthy();
            let true_val = eval_expr(
                true_val,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            let false_val = eval_expr(
                false_val,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            Ok(if cond { true_val } else { false_val })
        }
        Expr::Cast { target, value } => {
            let value = eval_expr(
                value,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?;
            crate::execution::expr_cast::cast_value(target, &value)
        }
        Expr::Fma { a, b, c } => {
            let a = eval_expr(
                a,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?
            .try_as_f32()
            .ok_or_else(|| {
                Error::interp("fma operand `a` is not a float. Fix: cast to f32 before fma.")
            })?;
            let b = eval_expr(
                b,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?
            .try_as_f32()
            .ok_or_else(|| {
                Error::interp("fma operand `b` is not a float. Fix: cast to f32 before fma.")
            })?;
            let c = eval_expr(
                c,
                invocation,
                memory,
                #[cfg(feature = "subgroup-ops")]
                snapshots,
            )?
            .try_as_f32()
            .ok_or_else(|| {
                Error::interp("fma operand `c` is not a float. Fix: cast to f32 before fma.")
            })?;
            let a = crate::execution::typed_ops::canonical_f32(a);
            let b = crate::execution::typed_ops::canonical_f32(b);
            let c = crate::execution::typed_ops::canonical_f32(c);
            crate::execution::op_count::record_op();
            Ok(Value::Float(f64::from(crate::execution::typed_ops::canonical_f32(
                a.mul_add(b, c),
            ))))
        }
        Expr::Atomic {
            op,
            buffer,
            index,
            expected,
            value,
            ordering: _,
        } => eval_atomic(
            *op,
            buffer,
            index,
            expected.as_deref(),
            value,
            invocation,
            memory,
            #[cfg(feature = "subgroup-ops")]
            snapshots,
        ),
        Expr::Opaque(extension) => Err(Error::interp(format!(
            "hashmap reference interpreter does not support opaque expression extension `{}`/`{}`. Fix: provide a reference evaluator for this ExprNode or lower it to core Expr variants before evaluation.",
            extension.extension_kind(),
            extension.debug_identity()
        ))),
        Expr::SubgroupBallot { cond } => {
            #[cfg(feature = "subgroup-ops")]
            {
                eval_subgroup_ballot(cond, invocation, snapshots, memory)
            }
            #[cfg(not(feature = "subgroup-ops"))]
            {
                let cond = eval_expr(cond, invocation, memory)?.truthy();
                Ok(Value::U32(u32::from(cond)))
            }
        }
        Expr::SubgroupShuffle { value, lane } => {
            #[cfg(feature = "subgroup-ops")]
            {
                eval_subgroup_shuffle(value, lane, invocation, snapshots, memory)
            }
            #[cfg(not(feature = "subgroup-ops"))]
            {
                let value_val = eval_expr(value, invocation, memory)?;
                let lane_val = eval_expr(lane, invocation, memory)?;
                let lane_u32 = lane_val . try_as_u32 () . ok_or_else (| | { Error :: interp ("subgroup_shuffle lane index is not a u32. Fix: use a scalar u32 lane argument." ,) }) ? ;
                Ok(if lane_u32 == 0 {
                    value_val
                } else {
                    Value::U32(0)
                })
            }
        }
        Expr::SubgroupReduce { op, value } => {
            #[cfg(feature = "subgroup-ops")]
            {
                eval_subgroup_reduce(*op, value, invocation, snapshots, memory)
            }
            #[cfg(not(feature = "subgroup-ops"))]
            {
                // Single-lane interpreter: a reduction over one lane is that
                // lane's value for every operator (Add/Mul/Min/Max/And/Or/Xor).
                let _ = op;
                eval_expr(value, invocation, memory)
            }
        }
        _ => Err(Error::interp(
            "hashmap reference interpreter encountered an unknown expression variant. Fix: add explicit reference semantics for the new ExprNode before dispatch.",
        )),
    }
}
#[allow(clippy::too_many_arguments)]
fn eval_atomic(
    op: AtomicOp,
    buffer: &str,
    index: &Expr,
    expected: Option<&Expr>,
    value: &Expr,
    invocation: &mut HashmapInvocation<'_>,
    memory: &mut HashmapMemory,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<Value, Error> {
    match (op, expected) {
        (AtomicOp::CompareExchange, None) => {
            return Err(Error::interp(
                "compare-exchange atomic is missing expected value. Fix: set Expr::Atomic.expected for AtomicOp::CompareExchange.",
            ));
        }
        (AtomicOp::CompareExchange, Some(_)) => {}
        (_, Some(_)) => {
            return Err(Error::interp(
                "non-compare-exchange atomic includes an expected value. Fix: use Expr::Atomic.expected only with AtomicOp::CompareExchange.",
            ));
        }
        (_, None) => {}
    }
    let idx = eval_to_index(
        index,
        "atomic index",
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    let expected = expected . map (| expr | { eval_expr (expr , invocation , memory , #[cfg (feature = "subgroup-ops")] snapshots ,) ? . try_as_u32 () . ok_or_else (| | { Error :: interp (format ! ("atomic expected value {expr:?} cannot be represented as u32. Fix: use a scalar u32-compatible argument.")) }) }) . transpose () ? ;
    let value = eval_expr(
        value,
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    let value = value.try_as_u32().ok_or_else(|| {
        Error::interp(
            "atomic value cannot be represented as u32. Fix: use a scalar u32-compatible argument.",
        )
    })?;
    let target = atomic_buffer_mut(memory, buffer)?;
    let Some(old) = oob::atomic_load(target, idx) else {
        return Ok(Value::U32(0));
    };
    let (old, new) = atomics::apply(op, old, expected, value)?;
    oob::atomic_store(target, idx, new);
    Ok(Value::U32(old))
}

/// Structural locks for the GridSync segmentation that makes fused multi-pass kernels
/// globally ordered under `reference_eval` (the fix for multi-block prefix-scan Pass-B
/// reading Pass-A's not-yet-written per-block totals). These pin the private splitting
/// helpers IN the crate that owns them, the end-to-end value parity lives downstream in
/// `vyre-primitives`'s multi_block/line_index tests, but the split MECHANICS belong here.
#[cfg(test)]
mod grid_sync_segmentation {
    use super::*;
    use std::sync::Arc;
    use vyre::ir::{Expr, Ident, MemoryOrdering};

    fn gs() -> Node {
        Node::barrier_with_ordering(MemoryOrdering::GridSync)
    }
    fn seqcst() -> Node {
        Node::barrier_with_ordering(MemoryOrdering::SeqCst)
    }
    fn other() -> Node {
        Node::return_()
    }
    fn region(body: Vec<Node>) -> Node {
        Node::Region {
            generator: Ident::from("g"),
            source_region: None,
            body: Arc::new(body),
        }
    }
    fn gs_count(nodes: &[Node]) -> usize {
        nodes
            .iter()
            .filter(|node| is_grid_sync_barrier(node))
            .count()
    }
    fn has_scope(nodes: &[Node]) -> bool {
        nodes
            .iter()
            .any(|node| matches!(node, Node::Block(_) | Node::Region { .. }))
    }

    #[test]
    fn is_grid_sync_barrier_matches_only_gridsync() {
        assert!(is_grid_sync_barrier(&gs()));
        // A workgroup-scoped SeqCst barrier is NOT a grid fence and must not split.
        assert!(!is_grid_sync_barrier(&seqcst()));
        assert!(!is_grid_sync_barrier(&other()));
    }

    #[test]
    fn contains_grid_sync_finds_top_level_and_nested_scopes() {
        assert!(contains_grid_sync(&[other(), gs(), other()]));
        assert!(!contains_grid_sync(&[other(), other()]));
        assert!(!contains_grid_sync(&[seqcst()]));
        assert!(contains_grid_sync(&[Node::block(vec![gs()])]));
        assert!(contains_grid_sync(&[region(vec![other(), gs()])]));
        // A Region wrapping a Block wrapping the barrier (the re-fused exclusive scan).
        assert!(contains_grid_sync(&[region(vec![Node::block(vec![gs()])])]));
    }

    #[test]
    fn contains_grid_sync_does_not_descend_into_data_dependent_control_flow() {
        // Fusion never emits a grid fence inside an `If`/`Loop`; the splitter must not
        // treat one there as a top-level segment boundary (it would be ill-defined).
        let inside_if = Node::if_then(Expr::bool(true), vec![gs()]);
        assert!(!contains_grid_sync(&[inside_if]));
    }

    #[test]
    fn split_partitions_at_each_top_level_barrier() {
        let body = vec![other(), gs(), other(), gs(), other()];
        let segments = split_top_level_grid_sync(&body);
        assert_eq!(segments.len(), 3, "two barriers => three segments");
        assert!(segments.iter().all(|segment| segment.len() == 1));
        // The barriers are the split points and appear in NO segment.
        assert!(segments.iter().all(|segment| gs_count(segment) == 0));
    }

    #[test]
    fn split_yields_one_segment_without_a_barrier() {
        let body = vec![other(), other()];
        let segments = split_top_level_grid_sync(&body);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].len(), 2);
    }

    #[test]
    fn split_emits_empty_trailing_segment_for_trailing_barrier() {
        let body = vec![other(), gs()];
        let segments = split_top_level_grid_sync(&body);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 1);
        assert_eq!(segments[1].len(), 0);
    }

    #[test]
    fn flatten_dissolves_gridsync_scopes_and_keeps_the_rest() {
        // A Block carrying a GridSync is dissolved so the barrier surfaces to top level.
        let mut dissolved = Vec::new();
        flatten_grid_sync_scopes(&[Node::block(vec![other(), gs(), other()])], &mut dissolved);
        assert_eq!(dissolved.len(), 3);
        assert!(
            !has_scope(&dissolved),
            "GridSync-carrying Block must be dissolved"
        );
        assert_eq!(gs_count(&dissolved), 1);

        // A scope WITHOUT a GridSync is preserved intact (its locals keep their scope).
        let mut preserved = Vec::new();
        flatten_grid_sync_scopes(&[Node::block(vec![other(), other()])], &mut preserved);
        assert_eq!(preserved.len(), 1);
        assert!(
            has_scope(&preserved),
            "a scope with no GridSync must be preserved"
        );
    }

    #[test]
    fn flatten_recurses_through_nested_gridsync_scopes_then_splits() {
        // The re-fused exclusive-scan shape nests the barrier one scope deeper; the
        // recursion must reach it so the subsequent split sees it at top level.
        let nested = region(vec![Node::block(vec![other(), gs(), other()])]);
        let mut flattened = Vec::new();
        flatten_grid_sync_scopes(&[nested], &mut flattened);
        assert!(
            !has_scope(&flattened),
            "all GridSync-carrying scopes must dissolve"
        );
        assert_eq!(gs_count(&flattened), 1);
        assert_eq!(
            split_top_level_grid_sync(&flattened).len(),
            2,
            "the surfaced barrier must partition into two segments"
        );
    }
}
