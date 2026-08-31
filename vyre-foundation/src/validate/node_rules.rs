//! The per-node validation rules, one owner each.
//!
//! Two walks run these rules: the production single-pass `PreorderValidator` in
//! `rule_pipeline`, and the recursive walk in `nodes` that the differential
//! property test keeps as its second arm. The property compares the two
//! traversals, so the traversals must stay independent, but the rule each
//! traversal applies at a node must not: a rule stated twice is a rule that can
//! be corrected once. Both arms already shared the program-header rules through
//! `rule_pipeline::validate_program_level` for exactly that reason, and every
//! node-level rule now lives here on the same terms.
//!
//! A rule takes the operands it reads plus the error sink. Anything the walk
//! owns - divergence, depth, scope frames, alias state - stays in the walk and
//! is passed in as a value where a rule needs it.

use rustc_hash::FxHashMap;

use super::binding::Binding;
use super::typecheck::{expr_type, ScopeTypes};
use super::{bytes_rejection, err, ValidationError, ValidationOptions};
use crate::ir_inner::model::expr::{Expr, Ident};
use crate::ir_inner::model::node::{Node, NodeExtension};
use crate::ir_inner::model::op_signature::{BufferAccess, DataType};
use crate::ir_inner::model::program::BufferDecl;
use crate::validate::{ValidationLocation, ValidationPhase};

/// Bindings displaced by a nested scope, in insertion order, so leaving the
/// scope can put the outer values back.
pub(crate) type ScopeLog = Vec<(Ident, Option<Binding>)>;

/// The buffer table a node rule resolves names against.
pub(crate) type BufferTable<'p> = FxHashMap<&'p str, &'p BufferDecl>;

/// The local bindings visible at a node.
pub(crate) type Scope = FxHashMap<Ident, Binding>;

/// V112: statements after a `return` in the same sequence are unreachable.
pub(crate) fn check_unreachable_after_return(nodes: &[Node], errors: &mut Vec<ValidationError>) {
    if let Some(position) = nodes.iter().position(|node| matches!(node, Node::Return)) {
        if position != nodes.len().saturating_sub(1) {
            errors.push(err(
                "V112",
                ValidationPhase::Node,
                ValidationLocation::Program,
                "unreachable statements after `return`".to_string(),
                "remove statements after `return` or reorder them.".to_string(),
            ));
        }
    }
}

/// V011, V045, V119, V120: the assignment target must exist, accept writes, and
/// accept the value's type.
pub(crate) fn check_assign(
    name: &Ident,
    value: &Expr,
    buffers: &BufferTable<'_>,
    scope: &Scope,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(binding) = scope.get(name.as_str()) {
        if !binding.mutable {
            errors.push(err(
                "V011",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!("assignment to loop variable `{name}`"),
                "loop variables are immutable.".to_string(),
            ));
        }
        if binding.ty_known {
            if let Some(value_ty) = expr_type(value, &mut ScopeTypes::new(buffers, scope)) {
                if value_ty != binding.ty {
                    errors.push(err("V045", ValidationPhase::Node, ValidationLocation::Program, format!(
                        "assignment to `{name}` has type `{value_ty}` but the binding was declared as `{declared}`",
                        declared = binding.ty
                    ), format!(
                        "cast the value to `{declared}` or introduce a new binding with the intended type.",
                        declared = binding.ty
                    )));
                }
            }
        }
    } else if let Some(buffer) = buffers.get(name.as_str()) {
        if buffer.access != BufferAccess::ReadWrite {
            errors.push(err("V119", ValidationPhase::Node, ValidationLocation::Program, format!(
                "assignment to buffer `{name}` requires read-write storage but declared access is `{access:?}`",
                access = buffer.access
            ), "use a read-write/output buffer or store into a mutable local binding"));
        }
        if let Some(value_ty) = expr_type(value, &mut ScopeTypes::new(buffers, scope)) {
            let element = &buffer.element;
            if !store_value_compatible(&value_ty, element) {
                errors.push(err(
                    "V045",
                    ValidationPhase::Node,
                    ValidationLocation::Program,
                    format!(
                        "assignment to buffer `{name}` has type `{value_ty}` but the buffer element type is `{element}`"
                    ),
                    format!(
                        "cast the value to `{element}` or write to a buffer with the intended element type."
                    ),
                ));
            }
        }
    } else {
        errors.push(err(
            "V120",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!("assignment to undeclared variable `{name}`"),
            format!("add `let {name} = ...;` before this assignment."),
        ));
    }
}

/// V121, V122, V036 and the packed-byte rejection: a store must write a
/// bit-compatible value at a `u32` index inside the declared element count.
pub(crate) fn check_store(
    buffer_name: &Ident,
    index: &Expr,
    value: &Expr,
    buffers: &BufferTable<'_>,
    scope: &Scope,
    errors: &mut Vec<ValidationError>,
) {
    bytes_rejection::check_store(buffer_name, buffers, errors);
    let Some(buffer) = buffers.get(buffer_name.as_str()) else {
        return;
    };
    if let Some(value_ty) = expr_type(value, &mut ScopeTypes::new(buffers, scope)) {
        let element = &buffer.element;
        if !store_value_compatible(&value_ty, element) {
            let legal_targets = store_value_targets(element);
            errors.push(err(
                "V121",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                    "Node::Store buffer `{buffer_name}` value has type `{value_ty}` but element type is `{element}`"
                ),
                format!("cast/store using one of {legal_targets}."),
            ));
        }
    }
    if let Some(index_ty) = expr_type(index, &mut ScopeTypes::new(buffers, scope)) {
        if index_ty != DataType::U32 {
            errors.push(err(
                "V122",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                    "Node::Store buffer `{buffer_name}` index has type `{index_ty}` but must be `u32`"
                ),
                "cast the index to U32 before storing.".to_string(),
            ));
        }
    }
    check_constant_store_index(buffer_name, buffer, index, errors);
}

/// V123: an `If` condition must be `u32` or `bool`.
pub(crate) fn check_if_condition(
    cond: &Expr,
    buffers: &BufferTable<'_>,
    scope: &Scope,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(cond_ty) = expr_type(cond, &mut ScopeTypes::new(buffers, scope)) {
        if !matches!(cond_ty, DataType::U32 | DataType::Bool) {
            errors.push(err(
                "V123",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!("Node::If condition has type `{cond_ty}` but must be `u32` or `bool`"),
                "cast or rewrite the condition expression to produce `u32` or `bool`.".to_string(),
            ));
        }
    }
}

/// V124, V125: both loop bounds must be `u32`.
pub(crate) fn check_loop_bounds(
    from: &Expr,
    to: &Expr,
    buffers: &BufferTable<'_>,
    scope: &Scope,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(from_ty) = expr_type(from, &mut ScopeTypes::new(buffers, scope)) {
        if from_ty != DataType::U32 {
            errors.push(err(
                "V124",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                    "Node::Loop from-bound has type `{from_ty}`; legal loop bound type is `u32`"
                ),
                "cast the `from` bound to `u32`.".to_string(),
            ));
        }
    }
    if let Some(to_ty) = expr_type(to, &mut ScopeTypes::new(buffers, scope)) {
        if to_ty != DataType::U32 {
            errors.push(err(
                "V125",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!("Node::Loop to-bound has type `{to_ty}`; legal loop bound type is `u32`"),
                "cast the `to` bound to `u32`.".to_string(),
            ));
        }
    }
}

/// The binding a loop counter takes: an immutable `u32` whose uniformity the
/// caller has already derived from the bounds and the enclosing divergence.
///
/// Both the back-edge scope a barrier check reads and the body scope the walk
/// pushes bind the counter, and they must bind it identically or a barrier
/// inside the body is judged against a counter the body does not have.
#[must_use]
pub(crate) fn loop_var_binding(uniform: bool) -> Binding {
    Binding {
        ty: DataType::U32,
        ty_known: true,
        mutable: false,
        uniform,
    }
}

/// V126, V127: an indirect dispatch reads a `u32` count tuple from a declared
/// buffer at a 4-byte aligned offset.
pub(crate) fn check_indirect_dispatch(
    count_buffer: &Ident,
    count_offset: u64,
    buffers: &BufferTable<'_>,
    errors: &mut Vec<ValidationError>,
) {
    if count_offset % 4 != 0 {
        errors.push(err(
            "V126",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!("indirect dispatch offset {count_offset} is not 4-byte aligned"),
            "use an offset aligned to a u32 dispatch count tuple.".to_string(),
        ));
    }
    if !buffers.contains_key(count_buffer.as_str()) {
        errors.push(err(
            "V127",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!("indirect dispatch references unknown buffer `{count_buffer}`"),
            "declare the count buffer before validation.".to_string(),
        ));
    }
}

/// V128: an async transfer pairs with its wait through a non-empty tag.
pub(crate) fn check_async_tag(tag: &Ident, errors: &mut Vec<ValidationError>) {
    if tag.is_empty() {
        errors.push(err(
            "V128",
            ValidationPhase::Node,
            ValidationLocation::Program,
            "async stream tag is empty".to_string(),
            "use a stable non-empty tag to pair AsyncLoad and AsyncWait nodes.".to_string(),
        ));
    }
}

/// V128, V134, V139: an async transfer pairs with its wait, targets a writable
/// buffer, and computes offset and size from workgroup-uniform expressions.
pub(crate) fn check_async_transfer(
    destination: &Ident,
    offset: &Expr,
    size: &Expr,
    tag: &Ident,
    buffers: &BufferTable<'_>,
    scope: &FxHashMap<Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    check_async_tag(tag, errors);
    bytes_rejection::check_async_destination(destination.as_str(), buffers, errors);
    check_async_uniformity(offset, size, buffers, scope, errors);
}

/// V139: an async transfer offset and size must be workgroup-uniform expressions.
pub(crate) fn check_async_uniformity(
    offset: &Expr,
    size: &Expr,
    buffers: &BufferTable<'_>,
    scope: &FxHashMap<Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    let load_policy = |buf_ident: &Ident| {
        buffers.get(buf_ident.as_str()).is_some_and(|b| {
            b.access == BufferAccess::Uniform || b.access == BufferAccess::ReadOnly
        })
    };
    if !super::uniformity::is_uniform_with_load_policy(offset, scope, load_policy) {
        errors.push(err(
            "V139",
            ValidationPhase::Node,
            ValidationLocation::Program,
            "async transfer offset expression is not workgroup-uniform".to_string(),
            "compute the transfer offset from workgroup-uniform expressions: literals, buffer lengths, workgroup ID, or a load from a read-only or uniform buffer at a uniform index, written in the operand rather than hoisted into a binding. Avoid thread-divergent expressions like invocation ID.".to_string(),
        ));
    }
    if !super::uniformity::is_uniform_with_load_policy(size, scope, load_policy) {
        errors.push(err(
            "V139",
            ValidationPhase::Node,
            ValidationLocation::Program,
            "async transfer size expression is not workgroup-uniform".to_string(),
            "compute the transfer size from workgroup-uniform expressions: literals, buffer lengths, workgroup ID, or a load from a read-only or uniform buffer at a uniform index, written in the operand rather than hoisted into a binding. Avoid thread-divergent expressions like invocation ID.".to_string(),
        ));
    }
}

/// The buffers a collective node names, in operand order.
///
/// One match over the collective variants, so a rule that reads a collective's
/// buffers and a walk that records them as accesses cannot disagree about which
/// buffers a variant has. A non-collective node names none.
///
/// Exhaustive with no catch-all arm: a new collective variant would answer
/// "names no buffers" under one, and the access walk would then let a program
/// write a buffer the alias rules never saw. Adding a variant fails to compile
/// here instead.
#[must_use]
pub(crate) fn collective_buffers(node: &Node) -> [Option<&Ident>; 2] {
    match node {
        Node::AllReduce { buffer, .. } | Node::Broadcast { buffer, .. } => [Some(buffer), None],
        Node::AllGather { input, output, .. } | Node::ReduceScatter { input, output, .. } => {
            [Some(input), Some(output)]
        }
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::If { .. }
        | Node::Loop { .. }
        | Node::Return
        | Node::Block(_)
        | Node::Barrier { .. }
        | Node::LogicalBarrier { .. }
        | Node::Region { .. }
        | Node::IndirectDispatch { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::AsyncWait { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::TileLoad { .. }
        | Node::TileStore { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileElementwise { .. }
        | Node::TileDecl { .. }
        | Node::Opaque(_) => [None, None],
    }
}

/// V046: collectives need backend transport, device-visible buffers, and one
/// element type across an input/output pair.
pub(crate) fn check_collective(
    node: &Node,
    options: ValidationOptions<'_>,
    buffers: &BufferTable<'_>,
    errors: &mut Vec<ValidationError>,
) {
    if !options.supports_distributed_collectives() {
        errors.push(err(
            "V046",
            ValidationPhase::Node,
            ValidationLocation::Program,
            "distributed collective nodes require backend collective support".to_string(),
            "validate with BackendCapabilities { supports_distributed_collectives: true, .. } or lower collectives before this backend."
                .to_string(),
        ));
    }

    let [first, second] = collective_buffers(node);
    for name in [first, second].into_iter().flatten() {
        check_collective_buffer(name, buffers, errors);
    }

    let (Some(input), Some(output)) = (first, second) else {
        return;
    };
    if let (Some(input), Some(output)) = (buffers.get(input.as_str()), buffers.get(output.as_str()))
    {
        if input.element != output.element {
            errors.push(err(
                "V046",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                    "collective input/output element mismatch: `{}` is `{}`, `{}` is `{}`",
                    input.name(),
                    input.element,
                    output.name(),
                    output.element
                ),
                "use matching element types before collective lowering",
            ));
        }
    }
}

/// V046: a collective buffer must be declared and visible beyond the workgroup.
fn check_collective_buffer(
    name: &Ident,
    buffers: &BufferTable<'_>,
    errors: &mut Vec<ValidationError>,
) {
    let Some(buffer) = buffers.get(name.as_str()) else {
        errors.push(err(
            "V046",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!("collective references unknown buffer `{name}`"),
            "declare the collective buffer before validation.".to_string(),
        ));
        return;
    };
    if buffer.access == BufferAccess::Workgroup {
        errors.push(err(
            "V046",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!("collective buffer `{name}` is workgroup-local"),
            "use device/global storage visible to the distributed backend.".to_string(),
        ));
    }
}

/// V031: a downstream node extension must identify itself and pass its own
/// validation.
pub(crate) fn check_opaque_node_extension(
    extension: &dyn NodeExtension,
    errors: &mut Vec<ValidationError>,
) {
    if extension.extension_kind().is_empty() {
        errors.push(err(
            "V031",
            ValidationPhase::Node,
            ValidationLocation::Program,
            "opaque node extension has an empty extension_kind",
            "return a stable non-empty namespace from NodeExtension::extension_kind.",
        ));
    }
    if extension.debug_identity().is_empty() {
        errors.push(err(
            "V031",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!(
                "opaque node extension `{}` has an empty debug_identity",
                extension.extension_kind()
            ),
            "return a stable human-readable identity from NodeExtension::debug_identity",
        ));
    }
    if let Err(message) = extension.validate_extension() {
        errors.push(err(
            "V031",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!(
                "opaque node extension `{}`/`{}` failed validation: {message}",
                extension.extension_kind(),
                extension.debug_identity()
            ),
            "rewrite the program to satisfy this validation invariant",
        ));
    }
}

/// V036: a literal store index must fall inside a statically sized buffer.
pub(crate) fn check_constant_store_index(
    buffer_name: &str,
    buffer: &BufferDecl,
    index: &Expr,
    errors: &mut Vec<ValidationError>,
) {
    if buffer.count == 0 {
        return;
    }
    match index {
        Expr::LitU32(value) if *value >= buffer.count => {
            errors.push(err(
                "V036",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                    "store index {value} overflows buffer `{buffer_name}` with count {}",
                    buffer.count
                ),
                "keep constant store indices below the declared element count",
            ));
        }
        Expr::LitI32(value) if *value < 0 => {
            errors.push(err(
                "V036",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                    "store index {value} overflows buffer `{buffer_name}` with count {}",
                    buffer.count
                ),
                format!("keep constant store indices in 0..{}", buffer.count),
            ));
        }
        Expr::LitI32(value) => {
            let as_u32 = u32::try_from(*value).unwrap_or(u32::MAX);
            if as_u32 >= buffer.count {
                errors.push(err(
                    "V036",
                    ValidationPhase::Node,
                    ValidationLocation::Program,
                    format!(
                        "store index {value} overflows buffer `{buffer_name}` with count {}",
                        buffer.count
                    ),
                    "keep constant store indices below the declared element count",
                ));
            }
        }
        _ => {}
    }
}

/// Bind `name`, recording the value it displaced so the scope can be restored.
pub(crate) fn insert_binding(
    scope: &mut Scope,
    name: Ident,
    binding: Binding,
    scope_log: Option<&mut ScopeLog>,
) {
    let previous = scope.insert(name.clone(), binding);
    if let Some(scope_log) = scope_log {
        scope_log.push((name, previous));
    }
}

/// Undo a scope frame's bindings, newest first.
pub(crate) fn restore_scope(scope: &mut Scope, mut scope_log: ScopeLog) {
    while let Some((name, previous)) = scope_log.pop() {
        if let Some(binding) = previous {
            scope.insert(name, binding);
        } else {
            scope.remove(&name);
        }
    }
}

/// True if `value` and `element` are the SAME-WIDTH integer type differing only
/// in signedness (U32<->I32, U64<->I64).
///
/// A buffer element's signedness is observed only on LOAD (sign- vs zero-extend
/// on use); a STORE writes the raw little-endian word, so storing a U32-typed
/// value into an I32 buffer (or the 64-bit pair) is a bit-exact reinterpret
/// exactly `i32_slot = u32_val as i32` in Rust. This is load-bearing because the
/// typechecker types Mod/bitwise/shift results as U32 regardless of operand
/// signedness (Add/Sub/Mul/Div preserve the operand type via Frame::Bin), so a
/// valid `store(i32_buffer, rem(i32, i32))` would otherwise be rejected. Every
/// lower layer already reinterprets: a shader emitter coerces the store value to
/// the element type, a machine-code emitter stores a typeless 32-bit word, and
/// the reference oracle stores the value's raw bytes. NOT a widening or
/// narrowing coercion (those change bits): only the
/// same-width signed/unsigned reinterpret is bit-preserving and allowed here.
#[inline]
#[must_use]
pub(crate) fn same_width_int_reinterpret(value: &DataType, element: &DataType) -> bool {
    matches!(
        (value, element),
        (DataType::U32, DataType::I32)
            | (DataType::I32, DataType::U32)
            | (DataType::U64, DataType::I64)
            | (DataType::I64, DataType::U64)
    )
}

/// The single source of truth for whether a value of type `value` may be written
/// into a buffer whose element type is `element` (by `Node::Store` OR by an
/// assignment to a buffer binding, both write the same raw word, so they MUST
/// agree; they previously diverged, with the assign path silently accepting
/// `Bool <-> U32` that `Node::Store` rejected).
///
/// Permitted beyond an exact match, all bit-preserving or backend-coerced writes:
///   * `U32 <-> Bytes`: a packed-byte buffer round-trip.
///   * `U32 <-> Bool`: a comparison/flag (0/1) stored into a u32 buffer (the
///     emitter coerces Bool via `x != 0` / `select(1u, 0u)`); the `U32 -> Bool`
///     direction is inert because a `bool` storage-buffer element is not
///     host-shareable in a shader, but it is kept for symmetry with the assign path.
///   * `F32 -> F32`: identity (named explicitly so the `Bytes`/vector arms above
///     cannot be reached for a float).
///   * same-width signed/unsigned integer reinterpret (`U32<->I32`, `U64<->I64`).
/// A float-to-int, int-to-float, or differing-width write is NOT permitted (it
/// would change bits) and must use an explicit `Cast`.
#[inline]
#[must_use]
pub(crate) fn store_value_compatible(value: &DataType, element: &DataType) -> bool {
    value == element
        || matches!(
            (value, element),
            (DataType::U32, DataType::Bytes)
                | (DataType::Bytes, DataType::U32)
                | (DataType::U32, DataType::Bool)
                | (DataType::Bool, DataType::U32)
                | (DataType::F32, DataType::F32)
        )
        || same_width_int_reinterpret(value, element)
}

/// The element types a store into `element` may legally carry, for the V121 fix
/// hint.
#[inline]
pub(crate) fn store_value_targets(element: &DataType) -> String {
    let mut targets = vec![element.clone()];
    let legal = match element {
        DataType::U32 => vec![DataType::Bytes, DataType::I32],
        DataType::Bytes => vec![DataType::U32],
        DataType::I32 => vec![DataType::U32],
        DataType::U64 => vec![DataType::I64],
        DataType::I64 => vec![DataType::U64],
        _ => Vec::new(),
    };
    for target in legal {
        if !targets.contains(&target) {
            targets.push(target);
        }
    }

    targets
        .into_iter()
        .map(|target| format!("`{target}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// V138: Tile load validation and residency limit checks.
pub(crate) fn check_tile_load(
    tile_name: &Ident,
    tile_type: &crate::ir::Tile,
    buffer_name: &Ident,
    origin: &[Expr],
    buffers: &BufferTable<'_>,
    options: ValidationOptions<'_>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(buf) = buffers.get(buffer_name.as_str()) {
        if buf.access() == BufferAccess::WriteOnly {
            errors.push(err(
                "V138",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!("tile load from write-only buffer `{buffer_name}`"),
                "buffer must be ReadOnly or ReadWrite for tile loading.".to_string(),
            ));
        }
    } else {
        errors.push(err(
            "V138",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!("tile load from unknown buffer `{buffer_name}`"),
            "declare the buffer before loading tiles from it.".to_string(),
        ));
    }
    if origin.len() != tile_type.extents.len() {
        errors.push(err(
            "V138",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!(
                "tile load origin rank {} does not match tile rank {}",
                origin.len(),
                tile_type.extents.len()
            ),
            "provide one origin index per tile dimension.".to_string(),
        ));
    }
    check_tile_residency(tile_name, tile_type, options, errors);
}

/// V135: Check tile residency against target capabilities.
pub(crate) fn check_tile_residency(
    tile_name: &Ident,
    tile_type: &crate::ir::Tile,
    options: ValidationOptions<'_>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(caps) = options.backend_capabilities() {
        let bytes = tile_type.byte_size();
        match tile_type.residency {
            crate::ir::Residency::Workgroup => {
                if caps.max_shared_memory_bytes > 0 && bytes > caps.max_shared_memory_bytes as u64 {
                    errors.push(err(
                        "V135",
                        ValidationPhase::Node,
                        ValidationLocation::Program,
                        format!(
                            "total Workgroup residency of {bytes} bytes exceeds target profile limit {} in operation `{tile_name}`",
                            caps.max_shared_memory_bytes
                        ),
                        "reduce tile dimensions or shard workgroup memory.".to_string(),
                    ));
                }
            }
            crate::ir::Residency::Register => {
                if caps.regs_per_thread_max > 0 {
                    let words = (bytes + 3) / 4;
                    if words > caps.regs_per_thread_max as u64 {
                        errors.push(err(
                            "V135",
                            ValidationPhase::Node,
                            ValidationLocation::Program,
                            format!(
                                "live Register residency of {words} registers exceeds target profile limit {} in operation `{tile_name}`",
                                caps.regs_per_thread_max
                            ),
                            "reduce tile register footprint or spill to shared memory.".to_string(),
                        ));
                    }
                }
            }
            crate::ir::Residency::Subgroup => {
                if caps.subgroup_size > 0 && !tile_type.extents.is_empty() {
                    let total_elements = tile_type.element_count();
                    if total_elements % (caps.subgroup_size as usize) != 0
                        && (caps.subgroup_size as usize) % total_elements != 0
                    {
                        errors.push(err(
                            "V135",
                            ValidationPhase::Node,
                            ValidationLocation::Program,
                            format!(
                                "tile `{tile_name}` Subgroup residency with {total_elements} elements is incompatible with target subgroup size {}",
                                caps.subgroup_size
                            ),
                            "align tile dimensions to target subgroup size.".to_string(),
                        ));
                    }
                }
            }
            crate::ir::Residency::Global => {}
        }
    }
}

/// V136: Tile store validation.
pub(crate) fn check_tile_store(
    buffer_name: &Ident,
    _origin: &[Expr],
    _tile_name: &Ident,
    buffers: &BufferTable<'_>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(buf) = buffers.get(buffer_name.as_str()) {
        if buf.access() == BufferAccess::ReadOnly {
            errors.push(err(
                "V136",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!("tile store to read-only buffer `{buffer_name}`"),
                "buffer must be WriteOnly or ReadWrite for tile storing.".to_string(),
            ));
        }
    } else {
        errors.push(err(
            "V136",
            ValidationPhase::Node,
            ValidationLocation::Program,
            format!("tile store to unknown buffer `{buffer_name}`"),
            "declare the buffer before storing tiles to it.".to_string(),
        ));
    }
}

/// V137: Tile matmul validation.
pub(crate) fn check_tile_matmul(
    acc: &Ident,
    a: &Ident,
    b: &Ident,
    options: ValidationOptions<'_>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(caps) = options.backend_capabilities() {
        if !caps.supports_tensor_cores {
            errors.push(err(
                "V137",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                    "target profile does not support matrix tensor core instructions for operation `{acc} = {a} x {b}`"
                ),
                "lower matrix multiplication to scalar loops or run on a target with tensor core support.".to_string(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType, Expr, Ident};

    #[test]
    fn store_value_targets_u32_includes_bytes_and_i32() {
        let targets = store_value_targets(&DataType::U32);
        assert!(
            targets.contains("u32"),
            "target list should contain u32: {targets}"
        );
        assert!(
            targets.contains("bytes"),
            "target list should contain bytes: {targets}"
        );
        // i32 is the same-width signed sibling: a bit-exact reinterpret store.
        assert!(
            targets.contains("i32"),
            "target list should contain i32 (same-width reinterpret): {targets}"
        );
        assert!(
            !targets.contains("bool"),
            "target list must not allow bool/u32 store coercion: {targets}"
        );
        assert!(
            !targets.contains("f32"),
            "target list must not allow f32/u32 store coercion (different bit semantics): {targets}"
        );
    }

    #[test]
    fn store_value_targets_f32_is_self_only() {
        let targets = store_value_targets(&DataType::F32);
        assert!(targets.contains("f32"));
        assert!(!targets.contains("u32"));
    }

    #[test]
    fn same_width_int_reinterpret_only_same_width_signedness() {
        // The bit-exact same-width pairs are allowed.
        assert!(same_width_int_reinterpret(&DataType::U32, &DataType::I32));
        assert!(same_width_int_reinterpret(&DataType::I32, &DataType::U32));
        assert!(same_width_int_reinterpret(&DataType::U64, &DataType::I64));
        assert!(same_width_int_reinterpret(&DataType::I64, &DataType::U64));
        // Different-width or non-integer pairs are NOT (they would change bits).
        assert!(!same_width_int_reinterpret(&DataType::U32, &DataType::U64));
        assert!(!same_width_int_reinterpret(&DataType::U32, &DataType::U8));
        assert!(!same_width_int_reinterpret(&DataType::F32, &DataType::U32));
        assert!(!same_width_int_reinterpret(&DataType::U32, &DataType::F32));
        assert!(!same_width_int_reinterpret(&DataType::U32, &DataType::U32));
    }

    #[test]
    fn store_value_compatible_unifies_store_and_assign_rules() {
        // Exact match, byte round-trip, bool flag, f32 identity, same-width int.
        assert!(store_value_compatible(&DataType::U32, &DataType::U32));
        assert!(store_value_compatible(&DataType::U32, &DataType::Bytes));
        assert!(store_value_compatible(&DataType::Bytes, &DataType::U32));
        assert!(store_value_compatible(&DataType::Bool, &DataType::U32));
        assert!(store_value_compatible(&DataType::U32, &DataType::Bool));
        assert!(store_value_compatible(&DataType::F32, &DataType::F32));
        assert!(store_value_compatible(&DataType::I32, &DataType::U32));
        assert!(store_value_compatible(&DataType::U64, &DataType::I64));
        // Bit-changing writes are NOT permitted (need an explicit Cast).
        assert!(!store_value_compatible(&DataType::F32, &DataType::U32));
        assert!(!store_value_compatible(&DataType::U32, &DataType::F32));
        assert!(!store_value_compatible(&DataType::Bool, &DataType::F32));
        assert!(!store_value_compatible(&DataType::U32, &DataType::U64));
        assert!(!store_value_compatible(&DataType::U8, &DataType::U32));
    }

    #[test]
    fn check_constant_store_index_within_bounds_no_error() {
        let buf = BufferDecl::read_write("buf", 0, DataType::U32).with_count(4);
        let mut errors = Vec::new();
        check_constant_store_index("buf", &buf, &Expr::u32(3), &mut errors);
        assert!(errors.is_empty());
    }

    #[test]
    fn check_constant_store_index_at_boundary_errors() {
        let buf = BufferDecl::read_write("buf", 0, DataType::U32).with_count(4);
        let mut errors = Vec::new();
        check_constant_store_index("buf", &buf, &Expr::u32(4), &mut errors);
        assert_eq!(
            errors.len(),
            1,
            "index == count should overflow: {errors:?}"
        );
    }

    #[test]
    fn check_constant_store_index_negative_i32_errors() {
        let buf = BufferDecl::read_write("buf", 0, DataType::U32).with_count(4);
        let mut errors = Vec::new();
        check_constant_store_index("buf", &buf, &Expr::i32(-1), &mut errors);
        assert_eq!(errors.len(), 1, "negative index should error: {errors:?}");
    }

    #[test]
    fn check_constant_store_index_zero_count_skips() {
        let buf = BufferDecl::read_write("buf", 0, DataType::U32);
        let mut errors = Vec::new();
        check_constant_store_index("buf", &buf, &Expr::u32(999), &mut errors);
        assert!(
            errors.is_empty(),
            "count=0 means dynamic and must be accepted"
        );
    }

    #[test]
    fn check_constant_store_index_dynamic_index_skips() {
        let buf = BufferDecl::read_write("buf", 0, DataType::U32).with_count(4);
        let mut errors = Vec::new();
        check_constant_store_index("buf", &buf, &Expr::Var(Ident::from("i")), &mut errors);
        assert!(errors.is_empty(), "dynamic index must be accepted");
    }

    /// `collective_buffers` is the one match every collective rule and every
    /// collective access walk reads, so a variant it forgets is a buffer no
    /// rule validates. The gate over the whole `Node` surface lives in
    /// `vyre-foundation/tests/validator_node_kind_coverage.rs`; this pins the
    /// operand order the callers depend on.
    #[test]
    fn collective_buffers_reports_operands_in_order() {
        use crate::ir::{CollectiveOp, CommGroup, Node};

        let all_reduce = Node::AllReduce {
            buffer: "shared".into(),
            op: CollectiveOp::Sum,
            group: CommGroup::WORLD,
        };
        assert_eq!(
            collective_buffers(&all_reduce).map(|name| name.map(Ident::to_string)),
            [Some("shared".to_string()), None]
        );

        let reduce_scatter = Node::ReduceScatter {
            input: "in".into(),
            output: "out".into(),
            op: CollectiveOp::Sum,
            group: CommGroup::WORLD,
        };
        assert_eq!(
            collective_buffers(&reduce_scatter).map(|name| name.map(Ident::to_string)),
            [Some("in".to_string()), Some("out".to_string())]
        );

        assert_eq!(collective_buffers(&Node::Return), [None, None]);
    }
}
