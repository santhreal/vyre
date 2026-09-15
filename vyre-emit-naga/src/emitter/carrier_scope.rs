//! The carrier-scope protocol: how a value produced inside a child block
//! becomes readable from the block that reads it afterwards.
//!
//! naga rejects an SSA handle whose `Statement::Emit` sits in a block the
//! reader is not inside (`no definition in scope for identifier _eN`). Every
//! structured op that opens a child body therefore routes the values that
//! escape it through a function-scope `LocalVariable`: seed the local from the
//! parent's pre-op value, emit the child, then rebind each escaping id to a
//! fresh `Load` in the parent block. This module owns that protocol, the two
//! local pools that implement it, and the analysis that decides which ids
//! escape.

use rustc_hash::{FxHashMap, FxHashSet};

use naga::{Expression, LocalVariable, ScalarKind, Span, Statement, Type};
use vyre_lower::{KernelBody, KernelOp, KernelOpKind};

use super::BodyBuilder;
use crate::EmitError;

type LoopCarrierSnapshot = (
    FxHashSet<u32>,
    FxHashMap<u32, naga::Handle<LocalVariable>>,
    FxHashMap<u32, naga::Handle<LocalVariable>>,
);

/// Which pool of function-scope locals a value publishes through.
///
/// Both pools exist for the same naga scoping reason and use the same
/// store-then-reload mechanism. They differ in lifetime: a loop carrier has to
/// survive iteration to iteration, a block-scoped local only has to outlive
/// the child block that computed it.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum LocalPool {
    LoopCarrier,
    BlockScoped,
}

impl LocalPool {
    /// Prefix of the emitted local's name. Part of the emitted text, so it is
    /// fixed per pool rather than derived at the call site.
    fn local_name_prefix(self) -> &'static str {
        match self {
            Self::LoopCarrier => "vyre_loop_carry_",
            Self::BlockScoped => "vyre_block_scope_",
        }
    }

    /// Name this pool reports in the `VYRE_BIND_RESULT_LOG` publish trace.
    pub(super) fn trace_name(self) -> &'static str {
        match self {
            Self::LoopCarrier => "LoopCarrier",
            Self::BlockScoped => "BlockScoped",
        }
    }
}

/// Whether a carrier seed Store retypes the parent value first.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum CarrierSeed {
    /// Store the parent value unchanged. A structured `for` loop seeds this
    /// way: the loop's index type already decided the carrier's type.
    Verbatim,
    /// Coerce to the carrier local's own type first, so a parent value of a
    /// different scalar width does not trip naga's `InvalidStoreTypes` gate.
    CoercedToLocal,
}

/// How a structured op's operand list splits into plain value references and
/// child-body indices.
struct StructuredOperands {
    /// Operand positions holding plain value ids.
    values: &'static [usize],
    /// First operand position holding a child-body index.
    children_from: usize,
    /// Number of child-body operands, or `None` for every remaining operand.
    children: Option<usize>,
}

/// Operand layout of a structured op, or `None` for an op that references
/// every operand as a plain value and opens no child body.
///
/// Adding a structured op means adding a row here. Leaving it out routes the
/// op through the `None` case, which reads its child-body index as a value id.
fn structured_operands(kind: &KernelOpKind) -> Option<StructuredOperands> {
    match kind {
        KernelOpKind::StructuredIfThen | KernelOpKind::StructuredBlock => {
            Some(StructuredOperands {
                values: &[0],
                children_from: 1,
                children: Some(1),
            })
        }
        KernelOpKind::StructuredIfThenElse => Some(StructuredOperands {
            values: &[0],
            children_from: 1,
            children: None,
        }),
        KernelOpKind::StructuredForLoop { .. } => Some(StructuredOperands {
            values: &[0, 1],
            children_from: 2,
            children: Some(1),
        }),
        KernelOpKind::Region { .. } => Some(StructuredOperands {
            values: &[],
            children_from: 0,
            children: Some(1),
        }),
        _ => None,
    }
}

/// Position of `op` inside `body.ops`, matched by pointer identity because
/// numeric descriptor ids are not unique after structured lowering.
fn op_position(body: &KernelBody, op: &KernelOp) -> usize {
    body.ops
        .iter()
        .position(|candidate| std::ptr::eq(candidate, op))
        .unwrap_or(body.ops.len())
}

/// Restore the binding `key` shadowed while a nested scope was open, or drop
/// the key when the nested scope introduced it.
fn restore_shadowed<V>(
    map: &mut FxHashMap<vyre_lower::Name, V>,
    key: &vyre_lower::Name,
    previous: Option<V>,
) {
    match previous {
        Some(previous) => {
            map.insert(key.clone(), previous);
        }
        None => {
            map.remove(key);
        }
    }
}

impl BodyBuilder<'_> {
    pub(super) fn snapshot_loop_carriers(&self) -> LoopCarrierSnapshot {
        (
            self.loop_carrier_targets.clone(),
            self.loop_carrier_locals.clone(),
            self.block_scoped_locals.clone(),
        )
    }

    /// Restore the carrier-target gate after a structured op ends.
    ///
    /// `loop_carrier_targets` is restored (it is a per-scope gate that
    /// controls whether `bind_result` runs the carrier-publish path), and the
    /// carrier-local resolver map is restored to the parent scope. The
    /// allocated `naga::LocalVariable`s remain function-scoped, but their
    /// numeric ids must not keep shadowing parent SSA after the structured op
    /// closes. Lowering can reuse descriptor ids for unrelated outer values
    /// and inner loop temporaries; leaving the inner carrier map live made
    /// Sinkhorn's inner GEMM loop replace the outer lane/count ids after the
    /// loop. Post-op users that genuinely need the result are already rebound
    /// to a fresh Load by `publish_carriers`.
    pub(super) fn restore_loop_carriers(&mut self, snapshot: LoopCarrierSnapshot) {
        let (targets, locals, block_locals) = snapshot;
        self.loop_carrier_targets = targets;
        self.loop_carrier_locals = locals;
        self.block_scoped_locals = block_locals;
    }

    /// Ids produced inside a structured `for` loop's child body that the
    /// parent body references after the loop.
    pub(super) fn collect_loop_carried_ids(
        &self,
        parent: &KernelBody,
        loop_op: &KernelOp,
    ) -> FxHashSet<u32> {
        let Some(child_idx) = loop_op.operands.get(2).copied() else {
            return FxHashSet::default();
        };
        let loop_pos = parent.ops.iter().position(|op| {
            matches!(&op.kind, KernelOpKind::StructuredForLoop { .. })
                && op.operands.get(2).copied() == Some(child_idx)
        });
        let Some(loop_pos) = loop_pos else {
            return FxHashSet::default();
        };
        self.collect_child_carried_ids(parent, loop_pos, &[child_idx])
    }

    /// Ids produced inside any of the child bodies named by `child_indices`
    /// that the parent body references after the op at `op_pos`.
    ///
    /// Without the round-trip these ids drive, naga's WGSL writer emits `let
    /// _eN = ...;` inside the child block and the reader after it uses `_eN`
    /// from the outer scope, which wgpu rejects with `no definition in scope
    /// for identifier _eN`.
    pub(super) fn collect_child_carried_ids(
        &self,
        parent: &KernelBody,
        op_pos: usize,
        child_indices: &[u32],
    ) -> FxHashSet<u32> {
        let mut produced_inside = FxHashSet::default();
        for child_idx in child_indices {
            if let Some(child) = parent.child_bodies.get(*child_idx as usize) {
                collect_produced_ids(child, &mut produced_inside);
            }
        }

        let mut referenced_after = FxHashSet::default();
        for op in parent.ops.iter().skip(op_pos + 1) {
            collect_op_referenced_ids(op, parent, &mut referenced_after);
        }

        produced_inside
            .into_iter()
            .filter(|id| referenced_after.contains(id))
            .collect()
    }

    /// Run the whole carrier-scope protocol around `emit`, which emits the
    /// structured statement and its child blocks.
    ///
    /// `child_operands` names the operand positions holding this op's
    /// child-body indices, so the carrier analysis sees exactly the bodies
    /// `emit` is about to open.
    pub(in crate::emitter) fn with_carrier_scope(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
        child_operands: &[usize],
        emit: impl FnOnce(&mut Self) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let prior_carriers = self.snapshot_loop_carriers();
        let op_pos = op_position(body, op);
        let child_body_idxs: Vec<u32> = child_operands
            .iter()
            .filter_map(|i| op.operands.get(*i).copied())
            .collect();
        let targets = self.collect_child_carried_ids(body, op_pos, &child_body_idxs);
        let seeds = self.register_carrier_targets(&targets);
        self.store_carrier_seeds(&seeds, CarrierSeed::CoercedToLocal);
        emit(self)?;
        self.publish_carriers(&targets, prior_carriers);
        Ok(())
    }

    /// Register every id in `targets` as a carrier and resolve the value each
    /// already holds in the parent scope.
    ///
    /// Resolution has to happen before the caller appends any expression
    /// belonging to the child: `value_handle_for_id` synthesizes a fresh
    /// `Load` in the *current* block when the cached handle's `Statement::Emit`
    /// range has closed, and that Load has to land in the parent block where
    /// the seed Store will read it.
    pub(super) fn register_carrier_targets(
        &mut self,
        targets: &FxHashSet<u32>,
    ) -> Vec<(u32, naga::Handle<Expression>)> {
        let mut seeds = Vec::with_capacity(targets.len());
        for id in targets {
            self.loop_carrier_targets.insert(*id);
            if let Some(handle) = self.value_handle_for_id(*id) {
                seeds.push((*id, handle));
            }
        }
        seeds
    }

    /// Seed each registered carrier's local with its parent value, so
    /// iteration 0 (or a not-taken if-arm) reads the pre-op value instead of
    /// an uninitialised local.
    pub(super) fn store_carrier_seeds(
        &mut self,
        seeds: &[(u32, naga::Handle<Expression>)],
        seed: CarrierSeed,
    ) {
        for (id, init) in seeds {
            let local = self.allocate_carrier_local(*id, init);
            match seed {
                CarrierSeed::Verbatim => self.store_into_local(local, *init),
                CarrierSeed::CoercedToLocal => self.store_coerced_to_local(local, *init),
            }
        }
    }

    /// Rebind every carried id to a fresh `Load` from its local in the current
    /// block, then restore the parent scope's carrier state so a sibling
    /// structured op starts clean.
    pub(super) fn publish_carriers(
        &mut self,
        targets: &FxHashSet<u32>,
        prior_carriers: LoopCarrierSnapshot,
    ) {
        for id in targets {
            if let Some(local) = self.loop_carrier_locals.get(id).copied() {
                let load = self.load_local(local);
                self.values.insert(*id, load);
            }
        }
        self.restore_loop_carriers(prior_carriers);
    }

    /// Publish `value` for `id` through `pool`: Store it into the pool's
    /// function-scope local, then rebind `id` to a fresh `Load` so every later
    /// reader resolves through the local instead of an SSA handle whose
    /// `Statement::Emit` may sit in a block the reader is not inside.
    pub(super) fn publish_through_local(
        &mut self,
        pool: LocalPool,
        id: u32,
        value: naga::Handle<Expression>,
    ) {
        let local = match pool {
            LocalPool::LoopCarrier => self.allocate_carrier_local(id, &value),
            LocalPool::BlockScoped => self.allocate_block_scoped_local(id, &value),
        };
        self.store_coerced_to_local(local, value);
        let load = self.load_local(local);
        self.values.insert(id, load);
    }

    /// Append a pointer to `local` and a `Load` through it, both in the
    /// current block.
    pub(super) fn load_local(
        &mut self,
        local: naga::Handle<LocalVariable>,
    ) -> naga::Handle<Expression> {
        let pointer = self.append_expr(Expression::LocalVariable(local));
        self.append_expr(Expression::Load { pointer })
    }

    /// Load `local` in the current block, coerced to `expected_ty` when
    /// emission recorded a type for the id being resolved.
    pub(super) fn load_local_as(
        &mut self,
        local: naga::Handle<LocalVariable>,
        expected_ty: Option<naga::Handle<Type>>,
    ) -> naga::Handle<Expression> {
        let load = self.load_local(local);
        match expected_ty {
            Some(ty) => self.coerce_value_to_type(load, ty),
            None => load,
        }
    }

    /// Store `value` into `local` verbatim.
    pub(super) fn store_into_local(
        &mut self,
        local: naga::Handle<LocalVariable>,
        value: naga::Handle<Expression>,
    ) {
        let pointer = self.append_expr(Expression::LocalVariable(local));
        self.function
            .body
            .push(Statement::Store { pointer, value }, Span::UNDEFINED);
    }

    /// Store `value` into `local`, coerced to the local's own type first.
    ///
    /// A local is typed from the first init it saw; a later write to the same
    /// id can carry a different scalar kind, and naga rejects the Store with
    /// `InvalidStoreTypes` when the two disagree.
    pub(super) fn store_coerced_to_local(
        &mut self,
        local: naga::Handle<LocalVariable>,
        value: naga::Handle<Expression>,
    ) {
        let local_ty = self.function.local_variables[local].ty;
        let value = self.coerce_value_to_type(value, local_ty);
        self.store_into_local(local, value);
    }

    /// Allocate a function-scope local of `ty` and Store `value` into it. An
    /// absent `name` leaves the local anonymous, which is what naga's writer
    /// expects for a value with no descriptor result id to name it after.
    pub(super) fn allocate_local_seeded(
        &mut self,
        name: Option<String>,
        ty: naga::Handle<Type>,
        value: naga::Handle<Expression>,
    ) -> naga::Handle<LocalVariable> {
        let local = self.function.local_variables.append(
            LocalVariable {
                name,
                ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        self.store_into_local(local, value);
        local
    }

    /// The type emission recorded for `id`, kept only when naga accepts it as
    /// a `LocalVariable` type: the canonical scalars plus the
    /// `vec2<u32>`/`vec3<u32>` backings that U64/I64 lower to. Atomic, array
    /// and struct handles are rejected with `InvalidType`, so they are
    /// reported absent and the caller falls back to the value's scalar kind.
    fn recorded_local_type(&self, id: u32) -> Option<naga::Handle<Type>> {
        self.value_types.get(&id).copied().filter(|ty| {
            *ty == self.types.bool_ty
                || *ty == self.types.u32_ty
                || *ty == self.types.i32_ty
                || *ty == self.types.f32_ty
                || *ty == self.types.vec2_u32_ty
                || *ty == self.types.vec3_u32_ty
        })
    }

    /// Canonical `LocalVariable` type for whatever `handle` produces, decided
    /// from its scalar kind and falling back to `fallback` when the kind
    /// cannot be derived (nested `Select`/`Binary` operands often defeat it).
    fn local_type_for_expression(
        &self,
        handle: naga::Handle<Expression>,
        fallback: naga::Handle<Type>,
    ) -> naga::Handle<Type> {
        match self.scalar_kind_of_expression(handle, 0) {
            Some(
                kind @ (ScalarKind::Bool | ScalarKind::Sint | ScalarKind::Float | ScalarKind::Uint),
            ) => self.canonical_type_for_scalar_kind(kind),
            _ => fallback,
        }
    }

    /// The local `pool` currently holds for `id`.
    pub(super) fn scoped_local(
        &self,
        pool: LocalPool,
        id: u32,
    ) -> Option<naga::Handle<LocalVariable>> {
        match pool {
            LocalPool::LoopCarrier => self.loop_carrier_locals.get(&id).copied(),
            LocalPool::BlockScoped => self.block_scoped_locals.get(&id).copied(),
        }
    }

    /// Return the local `pool` holds for `id`, allocating a fresh one when the
    /// cached local has a different type.
    ///
    /// Lowering reuses an SSA id across sibling blocks for values of different
    /// scalar kind (a Bool comparison result, then a u32 state word in an
    /// NFA-scan shader). Returning the stale-typed local makes the caller's
    /// coerced Store validate as `InvalidStoreTypes` whenever the scalar-kind
    /// heuristic cannot re-derive the value's kind, because
    /// `coerce_value_to_type` then no-ops and leaves a u32 value stored into a
    /// bool local.
    fn intern_scoped_local(
        &mut self,
        pool: LocalPool,
        id: u32,
        ty: naga::Handle<Type>,
    ) -> naga::Handle<LocalVariable> {
        if let Some(existing) = self.scoped_local(pool, id) {
            if self.function.local_variables[existing].ty == ty {
                return existing;
            }
        }
        let local = self.function.local_variables.append(
            LocalVariable {
                name: Some(format!("{}{id}", pool.local_name_prefix())),
                ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        match pool {
            LocalPool::LoopCarrier => self.loop_carrier_locals.insert(id, local),
            LocalPool::BlockScoped => self.block_scoped_locals.insert(id, local),
        };
        local
    }

    /// Loop-carrier local for `id`. The recorded type wins here: it comes from
    /// `bind_result_typed`, which runs before `bind_result`.
    pub(super) fn allocate_carrier_local(
        &mut self,
        id: u32,
        init_handle: &naga::Handle<Expression>,
    ) -> naga::Handle<LocalVariable> {
        let ty = match self.recorded_local_type(id) {
            Some(ty) => ty,
            None => self.local_type_for_expression(*init_handle, self.types.u32_ty),
        };
        self.intern_scoped_local(LocalPool::LoopCarrier, id, ty)
    }

    /// Block-scoped local for `id`: a value produced inside a child block that
    /// only needs to be readable from a block after it, not carried across
    /// iterations.
    ///
    /// Here the value's own scalar kind wins and the recorded type is the
    /// fallback, so a 64-bit value gets its `vec2<u32>` backing rather than the
    /// u32 default that would fail the coerced Store.
    pub(super) fn allocate_block_scoped_local(
        &mut self,
        id: u32,
        init_handle: &naga::Handle<Expression>,
    ) -> naga::Handle<LocalVariable> {
        let fallback = self.recorded_local_type(id).unwrap_or(self.types.u32_ty);
        let ty = self.local_type_for_expression(*init_handle, fallback);
        self.intern_scoped_local(LocalPool::BlockScoped, id, ty)
    }
}

impl BodyBuilder<'_> {
    /// Allocate (idempotent) the function-scope local that backs the
    /// source-level loop carrier `name`.
    fn ensure_named_carrier_local(
        &mut self,
        name: &vyre_lower::Name,
        seed_handle: naga::Handle<Expression>,
    ) -> naga::Handle<LocalVariable> {
        if let Some(existing) = self.named_carrier_locals.get(name).copied() {
            return existing;
        }
        let ty = self.local_type_for_expression(seed_handle, self.types.u32_ty);
        let local = self.function.local_variables.append(
            LocalVariable {
                name: Some(format!("vyre_named_carry_{name}")),
                ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        self.named_carrier_locals.insert(name.clone(), local);
        self.named_carrier_types.insert(name.clone(), ty);
        local
    }

    pub(super) fn emit_loop_carrier_init(
        &mut self,
        op: &KernelOp,
        name: &vyre_lower::Name,
    ) -> Result<(), EmitError> {
        let seed = self.value_operand(op, 0)?;
        let local = self.ensure_named_carrier_local(name, seed);
        self.store_coerced_to_local(local, seed);
        Ok(())
    }

    pub(super) fn emit_loop_carrier_read(
        &mut self,
        op: &KernelOp,
        name: &vyre_lower::Name,
    ) -> Result<(), EmitError> {
        let local = *self.named_carrier_locals.get(name).ok_or_else(|| {
            EmitError::InvalidDescriptor(format!(
                "LoopCarrier `{name}` read before any LoopCarrierInit allocated its slot. \
                 Fix: lower a LoopCarrierInit op into the parent body before any \
                 LoopCarrier/LoopCarrierEnd op for this name."
            ))
        })?;
        let ty = self.function.local_variables[local].ty;
        let value = self.load_local(local);
        // Snapshot the carrier into its own local: consumers must see the
        // value as of this read, not whatever a later `LoopCarrierEnd` wrote.
        let snapshot = self.allocate_local_seeded(
            op.result
                .map(|id| format!("vyre_named_carry_snapshot_{id}")),
            ty,
            value,
        );
        let snapshot_value = self.load_local(snapshot);
        self.bind_result_typed(op, snapshot_value, ty)
    }

    pub(super) fn emit_loop_carrier_end(
        &mut self,
        op: &KernelOp,
        name: &vyre_lower::Name,
    ) -> Result<(), EmitError> {
        let value = self.value_operand(op, 0)?;
        let local = *self.named_carrier_locals.get(name).ok_or_else(|| {
            EmitError::InvalidDescriptor(format!(
                "LoopCarrierEnd `{name}` writes before any LoopCarrierInit allocated its slot."
            ))
        })?;
        self.store_coerced_to_local(local, value);
        Ok(())
    }

    /// Restore the loop-index bindings shadowed by a nested loop over the same
    /// variable name.
    pub(super) fn restore_loop_bindings(
        &mut self,
        loop_var: &vyre_lower::Name,
        previous_local: Option<naga::Handle<LocalVariable>>,
        previous_type: Option<naga::Handle<Type>>,
    ) {
        restore_shadowed(&mut self.loop_locals, loop_var, previous_local);
        restore_shadowed(&mut self.loop_types, loop_var, previous_type);
    }
}

fn collect_op_referenced_ids(op: &KernelOp, parent: &KernelBody, out: &mut FxHashSet<u32>) {
    let Some(layout) = structured_operands(&op.kind) else {
        out.extend(op.operands.iter().copied());
        return;
    };
    for position in layout.values {
        if let Some(&id) = op.operands.get(*position) {
            out.insert(id);
        }
    }
    let child_idxs = op.operands.iter().skip(layout.children_from);
    let child_idxs: Vec<u32> = match layout.children {
        Some(count) => child_idxs.take(count).copied().collect(),
        None => child_idxs.copied().collect(),
    };
    for child_idx in child_idxs {
        if let Some(child) = parent.child_bodies.get(child_idx as usize) {
            collect_body_referenced_ids(child, out);
        }
    }
}

fn collect_body_referenced_ids(body: &KernelBody, out: &mut FxHashSet<u32>) {
    for op in &body.ops {
        collect_op_referenced_ids(op, body, out);
    }
}

fn collect_produced_ids(body: &KernelBody, out: &mut FxHashSet<u32>) {
    for op in &body.ops {
        if let Some(result) = op.result {
            out.insert(result);
        }
    }
    for child in &body.child_bodies {
        collect_produced_ids(child, out);
    }
}
