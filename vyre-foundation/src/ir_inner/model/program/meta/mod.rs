use std::sync::atomic::Ordering;
use std::sync::Arc;

use vyre_spec::OpIntensity;

use crate::ir::Node;
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::op_signature::BufferAccess;
use crate::visit::walk_nodes_and_exprs;

use super::Program;

/// Bounded IR structure digest for the wire-hash fallback.
mod fallback_wire_hash;

/// Canonical buffer-declaration bytes, and the comparison that keys on them.
mod buffer_key;

pub(super) use buffer_key::buffer_decl_canonical_key;
pub(crate) use buffer_key::buffers_equal_ignoring_declaration_order;
use fallback_wire_hash::FallbackWireHasher;

/// Provenance for mutations that invalidate Program validation/cache state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ProgramMutationProvenance {
    /// Program has not been mutated since construction or successful validation.
    Clean = 0,
    /// The non-composable dispatch flag changed.
    NonComposableFlag = 1,
    /// The target workgroup dimensions changed.
    WorkgroupSize = 2,
    /// The substrate-neutral parallel-region dimensions changed.
    ParallelRegionSize = 3,
    /// A caller borrowed the mutable entry vector.
    EntryMutation = 4,
    /// Internal builder or decode path rewrote Program shape.
    InternalShapeMutation = 5,
    /// Mutation provenance is unknown, so validation must fail closed.
    Unknown = 255,
}

impl ProgramMutationProvenance {
    #[inline]
    const fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Clean,
            1 => Self::NonComposableFlag,
            2 => Self::WorkgroupSize,
            3 => Self::ParallelRegionSize,
            4 => Self::EntryMutation,
            5 => Self::InternalShapeMutation,
            _ => Self::Unknown,
        }
    }
}

impl Program {
    /// Re-apply the same top-level `Node::Region` contract as
    /// [`Program::wrapped`].
    ///
    /// The [`region_inline_engine`](crate::optimizer::passes::cleanup::region_inline_engine)
    /// pass flattens small Category-A regions so CSE/DCE can see a single
    /// function-shaped body, which can leave a statement-shaped entry list. The
    /// standard optimizer run ends with this helper so the program remains in
    /// a runnable, validator/reference-interpreter–compatible form while
    /// still benefiting from the inline pass.
    #[must_use]
    pub fn reconcile_runnable_top_level(self) -> Self {
        if self.is_top_level_region_wrapped() {
            return self;
        }
        // Move the entry Vec out via map_entry's Arc-aware path; one
        // Program rebuild instead of two scaffold rebuilds.
        self.map_entry(Self::wrap_entry)
    }

    /// Look up a buffer declaration by name.
    #[must_use]
    #[inline]
    pub fn buffer(&self, name: &str) -> Option<&super::BufferDecl> {
        self.buffer_index
            .get(name)
            .and_then(|&index| self.buffers.get(index))
    }

    /// Declared buffers.
    #[must_use]
    #[inline]
    pub fn buffers(&self) -> &[super::BufferDecl] {
        self.buffers.as_ref()
    }

    /// Access the buffer declaration Arc directly for identity checks.
    #[must_use]
    #[inline]
    #[cfg(test)]
    pub(crate) fn buffers_arc(&self) -> &Arc<[super::BufferDecl]> {
        &self.buffers
    }

    /// Compare two programs by observable IR structure.
    ///
    /// This walk intentionally ignores buffer declaration order and never
    /// consults arena-local allocation identity. Two programs are structurally
    /// equal when they declare the same buffers, workgroup size, optional entry
    /// op id, and entry body semantics.
    #[must_use]
    #[inline]
    pub fn structural_eq(&self, other: &Self) -> bool {
        // Identity short-circuit: Program::clone shares all the
        // inner Arcs, so comparing a cloned program against its
        // source (the common optimizer-pipeline pattern) is pure
        // refcount comparison.
        if std::ptr::eq(self, other)
            || (Arc::ptr_eq(&self.buffers, &other.buffers)
                && Arc::ptr_eq(&self.entry, &other.entry)
                && self.entry_op_id == other.entry_op_id
                && self.non_composable_with_self == other.non_composable_with_self
                && self.workgroup_size == other.workgroup_size)
        {
            return true;
        }
        self.entry_op_id == other.entry_op_id
            && self.non_composable_with_self == other.non_composable_with_self
            && buffers_equal_ignoring_declaration_order(&self.buffers, &other.buffers)
            && self.workgroup_size == other.workgroup_size
            && self.entry == other.entry
    }

    /// Workgroup dimensions.
    #[must_use]
    #[inline]
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// Substrate-neutral alias for [`workgroup_size`](Self::workgroup_size).
    ///
    /// Naming: "parallel region" avoids picking a single target substrate's
    /// word for one dispatch invocation grouping.
    #[must_use]
    #[inline]
    pub fn parallel_region_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// Return true when this program must not be fused with another copy
    /// of itself in the same megakernel.
    #[must_use]
    #[inline]
    pub fn is_non_composable_with_self(&self) -> bool {
        self.non_composable_with_self
    }

    /// Mark this program as non-composable with itself.
    #[must_use]
    #[inline]
    pub fn with_non_composable_with_self(mut self, flag: bool) -> Self {
        self.non_composable_with_self = flag;
        self.invalidate_caches_for(ProgramMutationProvenance::NonComposableFlag);
        self
    }

    /// Set the workgroup dimensions in place. Used by harnesses that
    /// need to clone-and-rewrite a program's workgroup size for fallback
    /// dispatch  -  the alternative was to reconstruct the entire Program,
    /// which is unnecessarily expensive when only one field changes.
    #[inline]
    pub fn set_workgroup_size(&mut self, workgroup_size: [u32; 3]) {
        self.workgroup_size = workgroup_size;
        self.invalidate_caches_for(ProgramMutationProvenance::WorkgroupSize);
    }

    /// Substrate-neutral alias for [`set_workgroup_size`](Self::set_workgroup_size).
    #[inline]
    pub fn set_parallel_region_size(&mut self, parallel_region_size: [u32; 3]) {
        self.workgroup_size = parallel_region_size;
        self.invalidate_caches_for(ProgramMutationProvenance::ParallelRegionSize);
    }

    /// Apply lowered launch geometry dimensions to this program.
    #[inline]
    pub fn set_launch_geometry(&mut self, geometry: &crate::geometry::LaunchGeometry) {
        self.set_workgroup_size(geometry.workgroup);
    }

    /// Return a clone of this program with lowered launch geometry applied.
    #[must_use]
    #[inline]
    pub fn with_launch_geometry(&self, geometry: &crate::geometry::LaunchGeometry) -> Self {
        self.with_rewritten_workgroup_size_and_entry(geometry.workgroup, self.entry.as_ref().clone())
    }

    /// Entry-point nodes.
    #[must_use]
    #[inline]
    pub fn entry(&self) -> &[Node] {
        self.entry.as_ref().as_slice()
    }

    /// Shared entry-point body Arc for identity checks.
    #[must_use]
    #[inline]
    pub fn entry_arc(&self) -> &Arc<Vec<Node>> {
        &self.entry
    }

    /// Return true when this Program is the canonical no-op shape produced by
    /// [`Program::empty`]: no buffers and a single empty root Region.
    #[must_use]
    #[inline]
    pub fn is_explicit_noop(&self) -> bool {
        self.buffers().is_empty()
            && matches!(self.entry(), [Node::Region { body, .. }] if body.is_empty())
    }

    /// Return true when the program satisfies the top-level region-chain
    /// invariant: at least one top-level node, and every top-level node is a
    /// `Node::Region`.
    #[must_use]
    #[inline]
    pub fn is_top_level_region_wrapped(&self) -> bool {
        !self.entry.is_empty()
            && self
                .entry()
                .iter()
                .all(|node| matches!(node, Node::Region { .. }))
    }

    /// Structural cause for a violated top-level Region invariant.
    #[must_use]
    pub fn top_level_region_violation_cause(&self) -> Option<String> {
        if self.entry().is_empty() {
            return Some("program entry has no top-level Region".to_string());
        }

        self.entry()
            .iter()
            .enumerate()
            .find(|(_, node)| !matches!(node, Node::Region { .. }))
            .map(|(index, node)| {
                format!(
                    "program entry node {index} is `{}` instead of `Node::Region`",
                    crate::ir_inner::model::node::node_op_id(node)
                )
            })
    }

    /// Actionable error text describing why the top-level region invariant
    /// failed, or `None` when the entry is valid.
    #[must_use]
    pub fn top_level_region_violation(&self) -> Option<String> {
        self.top_level_region_violation_cause().map(|cause| {
            format!(
                "{cause}. Fix: construct runnable programs with Program::wrapped(...) or wrap the body in Node::Region before validation, interpretation, or dispatch."
            )
        })
    }

    /// Mutable entry-point nodes for transformation passes.
    #[must_use]
    #[inline]
    pub fn entry_mut(&mut self) -> &mut Vec<Node> {
        self.invalidate_caches_for(ProgramMutationProvenance::EntryMutation);
        Arc::make_mut(&mut self.entry)
    }

    /// Stable BLAKE3 fingerprint of the canonical wire-format bytes.
    #[must_use]
    #[inline]
    pub fn fingerprint(&self) -> [u8; 32] {
        *self.fingerprint.get_or_init(|| {
            let hash = self.compute_wire_hash();
            let _ = self.hash.set(hash);
            *hash.as_bytes()
        })
    }

    /// VSA-style hypervector fingerprint of the canonical wire-format
    /// bytes. Each `u32` lane is one segment of the program's blake3
    /// hash; together they form an 8-lane hypervector suitable for
    /// approximate similarity search via hamming distance.
    ///
    /// Use as the canonical cache key for approximate-match caches
    /// (e.g. validation cache, AOT artifact dedup); use
    /// [`Self::fingerprint`] for exact-match lookups.
    ///
    /// Wires the substrate's #29 hypervector primitive into Program
    /// itself  -  every Program now carries its own VSA fingerprint
    /// without callers having to reach into the substrate explicitly.
    #[must_use]
    pub fn vsa_fingerprint(&self) -> Vec<u32> {
        self.fingerprint()
            .chunks_exact(core::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Indices of writable storage outputs in `buffers()` order.
    #[must_use]
    #[inline]
    pub fn output_buffer_indices(&self) -> &[u32] {
        self.output_buffer_index
            .get_or_init(|| {
                Arc::new(
                    self.buffers()
                        .iter()
                        .enumerate()
                        .filter_map(|(index, buffer)| {
                            matches!(
                                buffer.access(),
                                BufferAccess::ReadWrite | BufferAccess::WriteOnly
                            )
                            .then(|| u32::try_from(index).ok())
                            .flatten()
                        })
                        .collect(),
                )
            })
            .as_slice()
    }

    /// True when the entry walk discovers any indirect dispatch node.
    #[must_use]
    #[inline]
    pub fn has_indirect_dispatch(&self) -> bool {
        *self.has_indirect_dispatch.get_or_init(|| {
            // Fast-path: ProgramStats records every node kind seen during
            // its single-pass walk. If the IndirectDispatch bit is unset,
            // the tree definitely contains no IndirectDispatch nodes and
            // the explicit traversal below would redundantly visit every
            // node only to return false. Reading the bit is O(1).
            if !self
                .stats()
                .has_any_node_kind(super::stats::NODE_KIND_INDIRECT_DISPATCH)
            {
                return false;
            }
            let mut stack: smallvec::SmallVec<[&Node; 32]> = self.entry().iter().rev().collect();
            while let Some(node) = stack.pop() {
                match node {
                    Node::IndirectDispatch { .. } => return true,
                    Node::If {
                        then, otherwise, ..
                    } => {
                        stack.extend(otherwise.iter().rev());
                        stack.extend(then.iter().rev());
                    }
                    Node::Loop { body, .. } | Node::Block(body) => {
                        stack.extend(body.iter().rev());
                    }
                    Node::Region { body, .. } => {
                        stack.extend(body.iter().rev());
                    }
                    Node::TileElementwise { body, .. } => {
                        stack.extend(body.iter().rev());
                    }
                    Node::Let { .. }
                    | Node::Assign { .. }
                    | Node::Store { .. }
                    | Node::AllReduce { .. }
                    | Node::AllGather { .. }
                    | Node::ReduceScatter { .. }
                    | Node::Broadcast { .. }
                    | Node::Return
                    | Node::Barrier { .. }
                    | Node::AsyncLoad { .. }
                    | Node::AsyncStore { .. }
                    | Node::AsyncWait { .. }
                    | Node::Trap { .. }
                    | Node::Resume { .. }
                    | Node::TileLoad { .. }
                    | Node::TileStore { .. }
                    | Node::TileMatmul { .. }
                    | Node::TileReduce { .. }
                    | Node::TileDecl { .. }
                    | Node::Opaque(_) => {}
                }
            }
            false
        })
    }

    /// Check whether a named buffer exists.
    #[must_use]
    #[inline]
    pub fn has_buffer(&self, name: &str) -> bool {
        self.buffer_index.contains_key(name)
    }

    /// Number of declared buffers.
    #[must_use]
    #[inline]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    #[inline]
    pub(super) fn build_buffer_index(
        buffers: &[super::BufferDecl],
    ) -> rustc_hash::FxHashMap<Arc<str>, usize> {
        let mut index = rustc_hash::FxHashMap::default();
        index.reserve(buffers.len());
        for (buffer_index, buffer) in buffers.iter().enumerate() {
            index
                .entry(Arc::clone(&buffer.name))
                .or_insert(buffer_index);
        }
        index
    }

    /// Mark this program as successfully validated structurally.
    #[inline]
    pub fn mark_structurally_validated(&self) {
        self.structural_validation_fingerprint.store(
            self.current_validation_fingerprint_token(),
            Ordering::Release,
        );
        self.mutation_provenance
            .store(ProgramMutationProvenance::Clean as u8, Ordering::Release);
        self.structural_validated.store(true, Ordering::Release);
    }

    /// Return true once structural validation has succeeded for this program shape.
    #[must_use]
    #[inline]
    pub fn is_structurally_validated(&self) -> bool {
        if !self.structural_validated.load(Ordering::Acquire) {
            return false;
        }
        if self.validation_mutation_provenance() == ProgramMutationProvenance::Unknown {
            self.structural_validated.store(false, Ordering::Release);
            return false;
        }
        let recorded = self
            .structural_validation_fingerprint
            .load(Ordering::Acquire);
        if recorded == 0 || recorded != self.current_validation_fingerprint_token() {
            self.structural_validated.store(false, Ordering::Release);
            return false;
        }
        true
    }

    /// Last mutation provenance recorded for validation/cache invalidation.
    #[must_use]
    #[inline]
    pub fn validation_mutation_provenance(&self) -> ProgramMutationProvenance {
        ProgramMutationProvenance::from_code(self.mutation_provenance.load(Ordering::Acquire))
    }

    /// Mark the Program as having been mutated by a boundary that cannot name a
    /// concrete provenance. Validation fails closed until the Program is rebuilt
    /// through a known constructor or known mutation API.
    #[inline]
    pub fn mark_unknown_mutation_provenance(&mut self) {
        self.invalidate_caches_for(ProgramMutationProvenance::Unknown);
    }

    /// Mark this program as successfully validated for a specific backend.
    #[inline]
    pub fn mark_validated_on(&self, backend_id: &str) {
        if self.validation_mutation_provenance() == ProgramMutationProvenance::Unknown {
            return;
        }
        self.validation_set
            .get_or_init(|| Arc::new(dashmap::DashSet::new()))
            .insert(Arc::from(self.validation_cache_key(backend_id)));
    }

    /// Return true if this program has been validated for the given backend.
    #[must_use]
    #[inline]
    pub fn is_validated_on(&self, backend_id: &str) -> bool {
        self.validation_set
            .get()
            .is_some_and(|set| set.contains(self.validation_cache_key(backend_id).as_str()))
    }

    /// Validate the program and cache the successful result on the program.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IrError::Validation`] with every structured issue when
    /// the structural validator rejects the program.
    pub fn validate(&self) -> crate::error::IrResult<()> {
        if self.validation_mutation_provenance() == ProgramMutationProvenance::Unknown {
            return Err(crate::error::IrError::WireFormatValidation {
                message: "program validation cache was invalidated by unknown mutation provenance. Fix: rebuild the Program through Program::wrapped/from_wire or use a named Program mutation API before validating.".into(),
            });
        }
        if self.is_structurally_validated() {
            return Ok(());
        }
        let errors = crate::validate::validate(self);
        if errors.is_empty() {
            self.mark_structurally_validated();
            return Ok(());
        }
        Err(crate::error::IrError::Validation { issues: errors })
    }

    #[inline]
    /// Estimate the peak VRAM byte size of this Program.
    ///
    /// Innovation I.11: Static VRAM Pressure Analysis.
    /// Returns the total bytes required by all storage and uniform buffers
    /// declared in the Program. Optimizer passes use this to automatically
    /// partition workloads if they would exceed a backend-specific safety
    /// margin.
    #[must_use]
    pub fn estimate_peak_vram_bytes(&self) -> u64 {
        self.buffers
            .iter()
            .map(|buffer| {
                let Some(element_size) = buffer.element.size_bytes() else {
                    return u64::MAX;
                };
                u64::from(buffer.count)
                    .saturating_mul(u64::try_from(element_size).unwrap_or(u64::MAX))
            })
            .fold(0u64, u64::saturating_add)
    }

    /// Return the peak computational intensity found in any instruction.
    ///
    /// Two hand-written descents used to stand here, `node_intensity` over
    /// `Node` and `expr_intensity` over `Expr`, each naming the positions it
    /// recursed into and each ending in `_ => OpIntensity::Free`. The node one
    /// named neither async copy nor `Trap`, so the expressions in an async
    /// copy's `offset` and `size` and in a trap address were scored `Free` no
    /// matter what they contained, and a program whose only heavy work sat there
    /// reported as free. The expression one had the same hole for any variant it
    /// had not been told about.
    ///
    /// Descent belongs to
    /// [`for_each_expr`](crate::visit::for_each_expr), the one walk
    /// that reaches every operand of every node and every sub-expression of
    /// every operand. What is left here is the scoring rule, which is per
    /// expression and does not need to see its children: the walk offers the
    /// children separately and `max` over the whole visit is the same answer the
    /// recursive `max` produced, in one pass rather than one pass per depth.
    #[must_use]
    pub fn peak_intensity(&self) -> OpIntensity {
        let mut peak = OpIntensity::Free;
        crate::visit::for_each_expr(self.entry(), |expr| {
            peak = peak.max(Self::expr_intensity(expr));
        });
        peak
    }

    /// Intensity of `expr` itself, ignoring its children.
    fn expr_intensity(expr: &crate::ir::Expr) -> OpIntensity {
        use crate::ir::Expr;
        match expr {
            Expr::BinOp { op, .. } => op.intensity(),
            Expr::Atomic { .. }
            | Expr::SubgroupBallot { .. }
            | Expr::SubgroupShuffle { .. }
            | Expr::SubgroupReduce { .. } => OpIntensity::Heavy,
            _ => OpIntensity::Free,
        }
    }

    fn compute_wire_hash(&self) -> blake3::Hash {
        match self.canonical_wire_hash() {
            Ok(hash) => hash,
            Err(error) => {
                let structural = self.structural_fingerprint_fallback();
                let err_msg = error.to_string();
                let mut fallback = Vec::with_capacity(96 + err_msg.len() + structural.len());
                fallback.extend_from_slice(b"VYRE-PROGRAM-CANONICAL-WIRE-HASH-ERROR\0");
                fallback.extend_from_slice(err_msg.as_bytes());
                fallback.push(0);
                fallback.extend_from_slice(structural.as_bytes());
                blake3::hash(&fallback)
            }
        }
    }

    fn structural_fingerprint_fallback(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"VYRE-WIRE-FALLBACK-V4\0");
        if let Some(id) = self.entry_op_id.as_deref() {
            hasher.update(id.as_bytes());
        }
        hasher.update(b"\0");
        for axis in &self.workgroup_size {
            hasher.update(&axis.to_le_bytes());
        }
        hasher.update(&[u8::from(self.non_composable_with_self)]);
        let mut keys: Vec<Vec<u8>> = self
            .buffers()
            .iter()
            .map(buffer_decl_canonical_key)
            .collect();
        keys.sort_unstable();
        for key in keys {
            hasher.update(&key);
        }
        let mut visitor = FallbackWireHasher(&mut hasher);
        walk_nodes_and_exprs(self, &mut visitor);
        hasher.finalize().to_hex().to_string()
    }

    fn validation_cache_key(&self, backend_id: &str) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let fingerprint = self.current_validation_fingerprint();
        let mut key = String::with_capacity(backend_id.len() + 1 + 64);
        key.push_str(backend_id);
        key.push(':');
        for &byte in &fingerprint {
            key.push(HEX[(byte >> 4) as usize] as char);
            key.push(HEX[(byte & 0x0f) as usize] as char);
        }
        key
    }

    #[inline]
    pub(super) fn invalidate_caches(&mut self) {
        self.invalidate_caches_for(ProgramMutationProvenance::InternalShapeMutation);
    }

    #[inline]
    pub(super) fn invalidate_caches_for(&mut self, provenance: ProgramMutationProvenance) {
        self.structural_validated.store(false, Ordering::Release);
        self.structural_validation_fingerprint
            .store(0, Ordering::Release);
        self.mutation_provenance
            .store(provenance as u8, Ordering::Release);
        if let Some(set) = self.validation_set.get() {
            set.clear();
        }
        let _ = self.hash.take();
        let _ = self.fingerprint.take();
        let _ = self.normalized_cache_digest.take();
        drop(self.output_buffer_index.take());
        let _ = self.has_indirect_dispatch.take();
        drop(self.stats.take());
    }

    fn current_validation_fingerprint(&self) -> [u8; 32] {
        *self.compute_wire_hash().as_bytes()
    }

    fn current_validation_fingerprint_token(&self) -> u64 {
        let bytes = self.current_validation_fingerprint();
        let token = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        token.max(1)
    }

    #[inline]
    pub(super) fn wrap_entry(entry: Vec<Node>) -> Vec<Node> {
        if !Self::entry_needs_root_region(&entry) {
            return entry;
        }
        vec![Node::Region {
            generator: Ident::from(Self::ROOT_REGION_GENERATOR),
            source_region: None,
            body: Arc::new(entry),
        }]
    }

    #[inline]
    fn entry_needs_root_region(entry: &[Node]) -> bool {
        entry.is_empty()
            || entry
                .iter()
                .any(|node| !matches!(node, Node::Region { .. }))
    }
}
