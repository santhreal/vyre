//! HashMap-backed reference interpreter split into execution, state, memory,
//! synchronization, and optional subgroup semantics.
//!
//! This root module owns expression evaluation and the split modules own their
//! state, memory, execution, synchronization, and subgroup contracts.

pub(crate) mod invocation;
pub(crate) mod memory;
pub(crate) mod step;
pub(crate) mod subgroup;
pub(crate) mod sync;

#[cfg(feature = "subgroup-ops")]
use invocation::HashmapInvocationSnapshot;
use invocation::{create_invocations, run_invocations, HashmapInvocation};
use memory::{atomic_buffer_mut, output_value, resolve_buffer, HashmapMemory};
use step::{axis_value, eval_call, eval_to_index};
#[cfg(feature = "subgroup-ops")]
use subgroup::{eval_subgroup_ballot, eval_subgroup_reduce, eval_subgroup_shuffle};
use sync::element_count;

use crate::ReferenceError;
use crate::{
    atomics,
    oob::{self, Buffer},
    value::Value,
};
use rustc_hash::FxHashMap;
use vyre_foundation::ir::{AtomicOp, BufferAccess, Expr, Node, Program};

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
    /// Workgroups and intra-workgroup invocations both stepped starting `by` positions
    /// in, wrapping around. Only the STEPPING order changes; every invocation keeps its
    /// true global/local ids.
    ///
    /// Reversal alone is a symmetric permutation, so a defect that maps lane identity
    /// onto step position survives any "reverse it back" repair and stays invisible to a
    /// forward-vs-reversed comparison of a reversal-symmetric program. A rotation is
    /// asymmetric, so it separates lane identity from step position for real.
    Rotated(u32),
}

/// Permute a dispatch-order list in place according to `lane_order`.
///
/// One home for the permutation so the workgroup list and the per-workgroup
/// invocation list cannot drift into stepping different orders.
fn apply_step_order<T>(items: &mut [T], lane_order: LaneOrder) {
    match lane_order {
        LaneOrder::Forward => {}
        LaneOrder::Reversed => items.reverse(),
        LaneOrder::Rotated(by) => {
            if !items.is_empty() {
                items.rotate_left(by as usize % items.len());
            }
        }
    }
}

/// Split `program` into the segments a whole-grid fence divides it into.
///
/// The split itself is `vyre_foundation::transform::grid_sync_split`, the same
/// transform a backend without a native cooperative launch runs. This crate had
/// its own copy, which flattened only unconditional wrappers and dropped the
/// `Let` bindings a segment inherited from an earlier one, so a program that
/// bound a value before the fence and read it after was rejected as referencing
/// an undeclared variable. Running the whole grid through segment `k` before
/// any workgroup enters `k+1` is what a launch boundary does.
fn grid_sync_segments(program: &Program) -> Result<Option<Vec<Program>>, ReferenceError> {
    if !vyre_foundation::transform::grid_sync_split::contains_grid_sync(program) {
        return Ok(None);
    }
    let segments = vyre_foundation::transform::grid_sync_split::split_on_grid_sync(program)
        .map_err(|error| {
            ReferenceError::new(format!(
                "cannot order a whole-grid fence: {error}. Fix: bound the fenced program's node count, or remove the fence."
            ))
        })?;
    Ok(Some(segments))
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
pub fn is_reference_output(decl: &vyre_foundation::ir::BufferDecl) -> bool {
    decl.is_backend_allocated_output() || decl.access() == BufferAccess::ReadWrite
}

/// Does the caller have to supply a `Value` for this buffer?
///
/// The other half of the interpreter's ABI: `reference_eval` consumes exactly
/// one `Value` per matching decl, in `Program::buffers` order.
///
/// The rule itself is `BufferDecl::consumes_host_input`, which `vyre_driver`'s
/// binding-role mapping and every backend also read, so the oracle asks for the
/// same list a device dispatch asks for. This function used to spell the rule
/// out as `access() != Workgroup && !is_backend_allocated_output()`, which
/// admitted three declarations no backend stages from the host: a `Shared`-kind
/// buffer, a `Persistent`-kind buffer, and a `pipeline_live_out` buffer whose
/// access is not `ReadWrite`. A program declaring one of those could not pass
/// parity, because the oracle wanted one more value than the device, and the
/// failure named a missing input rather than the disagreement.
///
/// Callers that build an input vector read this rather than re-deriving the
/// selection. A copy that drifts shifts every later input by one, which surfaces
/// as a missing value for whichever buffer the offset ran past.
#[must_use]
pub fn is_reference_input(decl: &vyre_foundation::ir::BufferDecl) -> bool {
    decl.consumes_host_input()
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
    explicit_grid: Option<[u32; 3]>,
) -> Result<Vec<Value>, ReferenceError> {
    #[cfg(feature = "subgroup-ops")]
    let validation_report = vyre_foundation::validate::validate_with_options(
        program,
        vyre_foundation::validate::ValidationOptions::default().with_backend_capabilities(
            vyre_foundation::validate::BackendCapabilities {
                supports_subgroup_ops: true,
                supports_tensor_cores: true,
                ..Default::default()
            },
        ),
    );
    #[cfg(not(feature = "subgroup-ops"))]
    let validation_report = vyre_foundation::validate::validate_with_options(
        program,
        vyre_foundation::validate::ValidationOptions::default(),
    );
    if let Some(source) = validation_report.errors.into_iter().next() {
        return Err(ReferenceError::validation(source));
    }
    // Every public entry point reaches this function, so the termination
    // contract is armed once here. A caller that already armed a budget, or an
    // enclosing evaluation, keeps its own ceiling and this guard is inert.
    let _budget = crate::execution::step_budget::arm(program);
    let mut storage = FxHashMap::default();
    // The interpreter's ABI is exactly the artifact ABI: one Value per
    // `is_reference_input` buffer. It used to also accept a vector sized to
    // every non-workgroup buffer, treating the extra entries as initializers
    // for backend-allocated outputs. Nothing writes those bytes on any path, so
    // the compatibility branch bought a fixture the right to be malformed: an
    // op whose fixture carried a placeholder for a backend-allocated output
    // passed every CPU lens and was rejected only by the strict artifact ABI on
    // a device, which is the wrong place and the wrong run to find out.
    let logical_input_count = program
        .buffers()
        .iter()
        .filter(|decl| is_reference_input(decl))
        .count();
    if inputs.len() > logical_input_count {
        return Err(ReferenceError::new(format!(
            "reference_eval received {} input Value(s) for a program with {logical_input_count} \
             reference input buffer(s), so {} of them is an unused input Value. Fix: pass one \
             Value per buffer accepted by `vyre_reference::is_reference_input`, in \
             `Program::buffers` order, and none for a backend-allocated output.",
            inputs.len(),
            inputs.len() - logical_input_count
        )));
    }
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
        // The oracle must refuse exactly what the device backends refuse. A
        // backend-allocated output with no static count has no size source on any
        // path: answering it with an empty buffer here certified programs that
        // both device backends reject, which is a certification hole rather than a
        // cosmetic inconsistency.
        decl.require_static_readback_size()
            .map_err(|message| ReferenceError::new(message))?;
        // One predicate, read here and exported for callers building the input
        // vector, so the two cannot drift apart.
        let bytes = if is_reference_input(decl) {
            let value = inputs.get(input_index).ok_or_else(|| {
                ReferenceError::new(format!(
                    "missing input for buffer `{}`. Fix: pass one Value per buffer accepted by \
                     `vyre_reference::is_reference_input`, in `Program::buffers` order, and none \
                     for a backend-allocated output.",
                    decl.name()
                ))
            })?;
            input_index += 1;
            value.to_bytes()
        } else {
            vec![0u8; required_bytes]
        };
        check_min_byte_len(decl, bytes.len(), required_bytes)?;
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
    // No count check closes the loop. The arm above refuses a longer vector and
    // the per-buffer lookup refuses a shorter one, so by here `input_index` is
    // the reference input count and equals `inputs.len()`. A third check on
    // that pair could not fail, and a check that cannot fail certifies nothing.
    if program.workgroup_size().contains(&0) {
        return Err(ReferenceError::new(
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
    // An explicit workgroup grid overrides shape inference entirely. Buffer-shape
    // inference distributes `total_wg` only across axes whose workgroup size is >1
    // (the powf split below), so a program dispatched over an inactive axis (a
    // `[256, 1, 1]` workgroup fanned across `grid.y` = query, as batched
    // persistent-BFS does) would collapse to `grid.y == 1` and SILENTLY compute
    // only the first slice (a Law-10 under-coverage). A caller that knows the real
    // dispatch grid passes it here so the interpreter covers exactly what the GPU
    // would, per-workgroup-axis. `None` keeps the inference path unchanged.
    let counts = if let Some([gx, gy, gz]) = explicit_grid {
        [gx.max(1), gy.max(1), gz.max(1)]
    } else {
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
        counts
    };
    let [workgroup_count_x, workgroup_count_y, workgroup_count_z] = counts;
    let entry = program.entry();
    // The budget was armed before the grid was known, so it carries the fixed
    // floor. Both terms of the program's own declared work are fixed now, so a
    // program whose extents are constant is admitted at the work it declares
    // rather than refused against a ceiling sized for a smaller corpus.
    crate::execution::step_budget::admit_declared_work(
        entry,
        u64::from(workgroup_count_x)
            .saturating_mul(u64::from(workgroup_count_y))
            .saturating_mul(u64::from(workgroup_count_z))
            .saturating_mul(u64::from(invocations_per_workgroup)),
    );
    #[cfg(feature = "subgroup-ops")]
    let uses_subgroup_ops = vyre_foundation::program_caps::scan(program).subgroup_ops;
    // A whole-grid fence orders every invocation in the dispatch, so the whole
    // grid advances through one segment before any workgroup enters the next.
    // A body with no fence is one segment and takes the original single-pass
    // path byte for byte.
    let split = grid_sync_segments(program)?;
    let segments: Vec<&[Node]> = match &split {
        Some(segments) => segments.iter().map(Program::entry).collect(),
        None => vec![entry],
    };
    // Canonical workgroup dispatch order (z,y,x-nested). A non-`Forward`
    // [`LaneOrder`] permutes this list, and the invocations within each workgroup, to
    // flip the deterministic last-writer of any non-atomic same-slot store, so an
    // output comparison against `Forward` surfaces a hidden cross-lane race. Forward
    // keeps the exact original nested-loop order.
    let mut wg_coords: Vec<[u32; 3]> = Vec::new();
    for wg_z in 0..workgroup_count_z {
        for wg_y in 0..workgroup_count_y {
            for wg_x in 0..workgroup_count_x {
                wg_coords.push([wg_x, wg_y, wg_z]);
            }
        }
    }
    apply_step_order(&mut wg_coords, lane_order);
    let mut memory = HashmapMemory::new(storage);
    for &segment in &segments {
        for &wg in &wg_coords {
            memory.reset_workgroup(program)?;
            let mut invocations = create_invocations(program, wg, segment)?;
            // Permute the STEP order only; each invocation retains its true
            // global/local ids and linear_local_index (fields move with the
            // element), so semantics are unchanged for a race-free program.
            apply_step_order(&mut invocations, lane_order);
            run_invocations(
                &mut memory,
                &mut invocations,
                #[cfg(feature = "subgroup-ops")]
                uses_subgroup_ops,
            )?;
        }
    }
    let mut storage = memory.storage;
    output_decls . into_iter () . map (| decl | { storage . remove (decl . name ()) . map (| buffer | output_value (buffer , & decl)) . ok_or_else (| | { let name = decl . name () ; ReferenceError::new(format ! ("missing output buffer `{name}` after dispatch. Fix: keep buffer declarations unique.")) }) }) . collect ()
}

/// Reject a caller-supplied buffer that is smaller than its declaration.
///
/// Single owner of the undersize diagnostic so the legacy output-initializer
/// path and the ordinary input path cannot drift into reporting the same
/// contract violation two different ways, or one of them not reporting it.
fn check_min_byte_len(
    decl: &vyre_foundation::ir::BufferDecl,
    supplied_bytes: usize,
    required_bytes: usize,
) -> Result<(), ReferenceError> {
    if supplied_bytes < required_bytes {
        return Err(ReferenceError::new(format!(
            "buffer `{}` has {} bytes but requires at least {} bytes ({} elements of {}). Fix: provide a larger input buffer.",
            decl.name(),
            supplied_bytes,
            required_bytes,
            decl.count(),
            decl.element()
        )));
    }
    Ok(())
}

fn declared_min_byte_len(decl: &vyre_foundation::ir::BufferDecl) -> Result<usize, ReferenceError> {
    match decl.static_byte_len() {
        Ok(Some(byte_len)) => Ok(byte_len),
        Ok(None) if decl.count() == 0 => Ok(0),
        Ok(None) => Err(ReferenceError::new(format!(
            "reference input buffer `{}` has unsized element type {}. Fix: use a fixed-width buffer element type.",
            decl.name(),
            decl.element()
        ))),
        Err(error) => Err(ReferenceError::new(error)),
    }
}

fn eval_expr(
    expr: &Expr,
    invocation: &mut HashmapInvocation<'_>,
    memory: &mut HashmapMemory,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<Value, ReferenceError> {
    match expr {
        Expr::LitU32(value) => Ok(Value::U32(*value)),
        Expr::LitI32(value) => Ok(Value::I32(*value)),
        Expr::LitF32(value) => Ok(Value::Float(f64::from(crate::execution::typed_ops::canonical_f32(
            *value,
        )))),
        Expr::LitBool(value) => Ok(Value::Bool(*value)),
        Expr::Var(name) => invocation.locals.local(name).ok_or_else(|| {
            ReferenceError::new(format!(
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
        Expr::LogicalIndex { axis } => axis_value(invocation.ids.global, *axis),
        Expr::LogicalTileId { axis } => axis_value(invocation.ids.workgroup, *axis),
        Expr::LogicalWithinTileId { axis } => axis_value(invocation.ids.local, *axis),
        Expr::SubgroupLocalId => {
            #[cfg(feature = "subgroup-ops")]
            {
                Ok(Value::U32(
                    invocation.linear_local_index % subgroup::subgroup_simulator().width() as u32,
                ))
            }
            #[cfg(not(feature = "subgroup-ops"))]
            {
                Ok(Value::U32(0))
            }
        }
        Expr::SubgroupSize => {
            #[cfg(feature = "subgroup-ops")]
            {
                Ok(Value::U32(subgroup::subgroup_simulator().width() as u32))
            }
            #[cfg(not(feature = "subgroup-ops"))]
            {
                Ok(Value::U32(1))
            }
        }
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
                ReferenceError::new("fma operand `a` is not a float. Fix: cast to f32 before fma.")
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
                ReferenceError::new("fma operand `b` is not a float. Fix: cast to f32 before fma.")
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
                ReferenceError::new("fma operand `c` is not a float. Fix: cast to f32 before fma.")
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
        Expr::Opaque(extension) => Err(ReferenceError::new(format!(
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
                let lane_u32 = lane_val . try_as_u32 () . ok_or_else (| | { ReferenceError::new("subgroup_shuffle lane index is not a u32. Fix: use a scalar u32 lane argument.") }) ? ;
                Ok(if lane_u32 == 0 {
                    value_val
                } else {
                    Value::U32(0)
                })
            }
        }
        #[cfg(feature = "subgroup-ops")]
        Expr::SubgroupReduce { op, value } => {
            eval_subgroup_reduce(*op, value, invocation, snapshots, memory)
        }
        // Single-lane interpreter: a reduction over one lane is that lane's
        // value for every operator (Add/Mul/Min/Max/And/Or/Xor), so the
        // operator is not read.
        #[cfg(not(feature = "subgroup-ops"))]
        Expr::SubgroupReduce { op: _, value } => eval_expr(value, invocation, memory),
        _ => Err(ReferenceError::new("hashmap reference interpreter encountered an unknown expression variant. Fix: add explicit reference semantics for the new ExprNode before dispatch.")),
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
) -> Result<Value, ReferenceError> {
    if op == AtomicOp::CompareExchange && expected.is_none() {
        return Err(ReferenceError::new("compare-exchange atomic is missing expected value. Fix: set Expr::Atomic.expected for AtomicOp::CompareExchange."));
    }
    if op != AtomicOp::CompareExchange && expected.is_some() {
        return Err(ReferenceError::new("non-compare-exchange atomic includes an expected value. Fix: use Expr::Atomic.expected only with AtomicOp::CompareExchange."));
    }
    let idx = eval_to_index(
        index,
        "atomic index",
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    let expected = expected . map (| expr | { eval_expr (expr , invocation , memory , #[cfg (feature = "subgroup-ops")] snapshots ,) ? . try_as_u32 () . ok_or_else (| | { ReferenceError::new(format ! ("atomic expected value {expr:?} cannot be represented as u32. Fix: use a scalar u32-compatible argument.")) }) }) . transpose () ? ;
    let value = eval_expr(
        value,
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?;
    let value = value.try_as_u32().ok_or_else(|| {
        ReferenceError::new(
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
