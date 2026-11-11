//! The fact table and the questions it answers.
//!
//! Every query here is a scan of one column or a lookup in an index built on
//! first use. The columns are filled by the build walk in `build.rs`; nothing
//! in this file walks a `Node`.

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::OnceLock;

use super::kind::{kind_mask, BufferRefKind, NodeIndex, NodeKind};
use crate::ir::Ident;

/// One row per `Node::Region` observed during the build walk  -
/// the diagnostic / source-correlation metadata that the
/// `Region` enum variant inlines. passes that don't
/// care about source provenance can ignore this column entirely;
/// passes that do care (diagnostics, region-inlining, region
/// identity tracking) iterate the column once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegionMeta {
    /// `NodeIndex` of the `Node::Region` within the `SoA` fact table.
    pub node: NodeIndex,
    /// `Region.generator`  -  the op id / pass / extension that
    /// produced this region, used by diagnostics to attribute
    /// errors back to their source.
    pub generator: Ident,
    /// `Region.source_region`  -  the optional generator ref that
    /// links a derived region back to the original source span.
    pub source_region: Option<Ident>,
}

/// Columnar fact view of a `Program`. Construct via
/// `ProgramFacts::build(&program)` and query through the helpers.
#[derive(Debug, Default)]
pub struct ProgramFacts {
    pub(super) kinds: Vec<NodeKind>,
    pub(super) parent: Vec<Option<NodeIndex>>,
    /// Bitset of every `NodeKind` discriminant observed during
    /// `build`. Populated alongside `kinds` so `has_kind` and
    /// `has_any_kind_in_mask` are O(1) bit tests instead of an O(N)
    /// scan of `kinds`. Pass-`analyze_impl` predicates (which run
    /// before every transform on every iteration) hit this in the
    /// hot pipeline.
    pub(super) kinds_present: u32,
    pub(super) lets: Vec<(NodeIndex, Ident)>,
    pub(super) assigns: Vec<(NodeIndex, Ident)>,
    pub(super) loop_vars: Vec<(NodeIndex, Ident)>,
    pub(super) var_reads: Vec<(NodeIndex, Ident)>,
    pub(super) buffer_refs: Vec<(NodeIndex, Ident, BufferRefKind)>,
    pub(super) regions: Vec<RegionMeta>,
    pub(super) let_index: OnceLock<FxHashMap<Ident, Vec<NodeIndex>>>,
    pub(super) assign_index: OnceLock<FxHashMap<Ident, Vec<NodeIndex>>>,
    pub(super) var_read_index: OnceLock<FxHashMap<Ident, Vec<NodeIndex>>>,
    pub(super) buffer_index: OnceLock<FxHashMap<Ident, Vec<(NodeIndex, BufferRefKind)>>>,
    pub(super) region_index_by_node: OnceLock<FxHashMap<NodeIndex, usize>>,
    pub(super) region_index_by_generator: OnceLock<FxHashMap<Ident, Vec<usize>>>,
}

impl ProgramFacts {
    /// Total number of nodes (preorder count) in the program tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.kinds.len()
    }

    /// `NodeKind` at index `idx`. Panics if `idx` is out of range  -
    /// callers should always pull indices from this same fact table.
    #[must_use]
    pub fn kind_at(&self, idx: NodeIndex) -> NodeKind {
        self.kinds[idx.0 as usize]
    }

    /// Parent node index, or `None` if `idx` is a root entry-level
    /// sibling.
    #[must_use]
    pub fn parent_of(&self, idx: NodeIndex) -> Option<NodeIndex> {
        self.parent[idx.0 as usize]
    }

    /// `true` iff `node` is inside the subtree rooted at `ancestor`.
    /// A node is considered inside itself. This is O(depth) and uses
    /// the parent column, avoiding a recursive tree walk for scoped
    /// optimizer queries.
    #[must_use]
    pub fn is_descendant_of(&self, node: NodeIndex, ancestor: NodeIndex) -> bool {
        let mut current = Some(node);
        while let Some(idx) = current {
            if idx == ancestor {
                return true;
            }
            current = self.parent_of(idx);
        }
        false
    }

    /// Iterate every `(NodeIndex, NodeKind)` in preorder.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (NodeIndex, NodeKind)> + '_ {
        self.kinds.iter().copied().enumerate().map(|(i, kind)| {
            (
                NodeIndex(u32::try_from(i).map_or(u32::MAX, |value| value)),
                kind,
            )
        })
    }

    /// Iterate optimizer-semantic nodes in preorder, skipping
    /// `NodeKind::Region`. Region generator/source payload lives in
    /// the [`RegionMeta`] side table, so passes that only care about
    /// computation can scan this view without matching through debug
    /// wrappers.
    pub fn iter_regionless_nodes(&self) -> impl Iterator<Item = (NodeIndex, NodeKind)> + '_ {
        self.iter_nodes()
            .filter(|(_, kind)| *kind != NodeKind::Region)
    }

    /// Parent in the optimizer-semantic tree, skipping any enclosing
    /// `NodeKind::Region` wrappers. This lets passes treat Region as
    /// provenance metadata while preserving the canonical wire tree
    /// for diagnostics and serialization.
    #[must_use]
    pub fn regionless_parent_of(&self, idx: NodeIndex) -> Option<NodeIndex> {
        let mut parent = self.parent_of(idx);
        while let Some(candidate) = parent {
            if self.kind_at(candidate) != NodeKind::Region {
                return Some(candidate);
            }
            parent = self.parent_of(candidate);
        }
        None
    }

    /// `true` iff at least one node has the given kind. O(1) bit
    /// test against the cached `kinds_present` mask populated during
    /// `build`.
    #[must_use]
    #[inline]
    pub fn has_kind(&self, kind: NodeKind) -> bool {
        (self.kinds_present & kind_mask(kind)) != 0
    }

    /// `true` iff at least one node's kind is in `mask`. O(1).
    /// Compose with [`kind_mask`] when checking several kinds at
    /// once: `facts.has_any_kind_in_mask(kind_mask(NodeKind::Loop) | kind_mask(NodeKind::If))`.
    #[must_use]
    #[inline]
    pub fn has_any_kind_in_mask(&self, mask: u32) -> bool {
        (self.kinds_present & mask) != 0
    }

    /// Raw kind-presence bitset. Exposed so passes that need to
    /// short-circuit on multiple distinct kinds can grab the mask
    /// once and AND/OR/XOR locally without going through the
    /// helpers per-kind.
    #[must_use]
    #[inline]
    pub fn kinds_present(&self) -> u32 {
        self.kinds_present
    }

    /// Every `(NodeIndex, name)` where `Node::Let { name, .. }` was
    /// observed. The order is preorder.
    #[must_use]
    pub fn lets(&self) -> &[(NodeIndex, Ident)] {
        &self.lets
    }

    /// Every `(NodeIndex, name)` where `Node::Assign { name, .. }`
    /// was observed.
    #[must_use]
    pub fn assigns(&self) -> &[(NodeIndex, Ident)] {
        &self.assigns
    }

    /// Every `(NodeIndex, name)` where `Node::Loop { var, .. }`
    /// declared an induction variable.
    #[must_use]
    pub fn loop_vars(&self) -> &[(NodeIndex, Ident)] {
        &self.loop_vars
    }

    /// Every `(NodeIndex, name)` where `Expr::Var(name)` appears
    /// (including inside compound expressions).
    #[must_use]
    pub fn var_reads(&self) -> &[(NodeIndex, Ident)] {
        &self.var_reads
    }

    /// Every `(NodeIndex, buffer, kind)` where a buffer was touched.
    #[must_use]
    pub fn buffer_refs(&self) -> &[(NodeIndex, Ident, BufferRefKind)] {
        &self.buffer_refs
    }

    /// All node indices where `Let(name, _)` was observed.
    /// Builds the lookup index on first call; subsequent calls are
    /// O(1) hash lookup.
    #[must_use]
    pub fn let_sites_of(&self, name: &str) -> &[NodeIndex] {
        let map = self.let_index.get_or_init(|| build_index(&self.lets));
        map.get(name).map_or(&[], Vec::as_slice)
    }

    /// All node indices where `Assign(name, _)` was observed.
    #[must_use]
    pub fn assign_sites_of(&self, name: &str) -> &[NodeIndex] {
        let map = self.assign_index.get_or_init(|| build_index(&self.assigns));
        map.get(name).map_or(&[], Vec::as_slice)
    }

    /// All node indices that read `Expr::Var(name)`.
    #[must_use]
    pub fn var_read_sites_of(&self, name: &str) -> &[NodeIndex] {
        let map = self
            .var_read_index
            .get_or_init(|| build_index(&self.var_reads));
        map.get(name).map_or(&[], Vec::as_slice)
    }

    /// Every site that touches buffer `name`, paired with the kind
    /// of touch (Read / Write / Atomic / `AsyncDestination` /
    /// `AsyncSource` / `IndirectCount`).
    #[must_use]
    pub fn buffer_refs_of(&self, name: &str) -> &[(NodeIndex, BufferRefKind)] {
        let map = self.buffer_index.get_or_init(|| {
            let mut out: FxHashMap<Ident, Vec<(NodeIndex, BufferRefKind)>> = FxHashMap::default();
            for (idx, buffer, kind) in &self.buffer_refs {
                out.entry(buffer.duplicate_handle())
                    .or_default()
                    .push((*idx, *kind));
            }
            out
        });
        map.get(name).map_or(&[], Vec::as_slice)
    }

    /// Every `Node::Region` observed during the build walk, with
    /// its diagnostic `generator` ident and optional `source_region`
    /// ref. the side-table half of "treat Region /
    /// source metadata as side tables during optimization, restore
    /// for diagnostics."
    #[must_use]
    pub fn regions(&self) -> &[RegionMeta] {
        &self.regions
    }

    /// Look up the `RegionMeta` for the `Node::Region` at `idx`,
    /// or `None` if `idx` is not a Region or no Region was recorded
    /// at that index. O(1) hash lookup once the index is built.
    #[must_use]
    pub fn region_at(&self, idx: NodeIndex) -> Option<&RegionMeta> {
        let map = self.region_index_by_node.get_or_init(|| {
            let mut out: FxHashMap<NodeIndex, usize> = FxHashMap::default();
            for (i, meta) in self.regions.iter().enumerate() {
                out.insert(meta.node, i);
            }
            out
        });
        map.get(&idx).and_then(|&i| self.regions.get(i))
    }

    /// All `Node::Region` sites whose `generator` ident equals the
    /// argument. O(1) hash lookup once the index is built.
    pub fn regions_by_generator(&self, generator: &str) -> impl Iterator<Item = &RegionMeta> + '_ {
        let map = self.region_index_by_generator.get_or_init(|| {
            let mut out: FxHashMap<Ident, Vec<usize>> = FxHashMap::default();
            for (i, meta) in self.regions.iter().enumerate() {
                out.entry(meta.generator.duplicate_handle())
                    .or_default()
                    .push(i);
            }
            out
        });
        map.get(generator)
            .map_or(&[] as &[usize], std::vec::Vec::as_slice)
            .iter()
            .filter_map(move |&i| self.regions.get(i))
    }

    /// Convenience: `true` iff `name` is rebound anywhere  -  either
    /// as a `Let` shadow, an `Assign`, or a `Loop` induction
    /// variable. Used by passes that want to check "is this name
    /// stable across the whole program?" without writing the same
    /// scan three times.
    #[must_use]
    pub fn is_name_rebound(&self, name: &str) -> bool {
        let lets = self.let_sites_of(name);
        if lets.len() > 1 {
            return true;
        }
        if !self.assign_sites_of(name).is_empty() {
            return true;
        }
        self.loop_vars.iter().any(|(_, var)| var.as_str() == name)
    }

    /// points-to fact: `true` iff `buf_a` and `buf_b`
    /// can be proven to refer to disjoint memory.
    ///
    /// Soundness: in vyre's IR every `BufferDecl` is a distinct
    /// named allocation. Two distinct buffer names declared in the
    /// program's buffer table are guaranteed by construction not to
    /// alias (the runtime allocates a fresh region per `BufferDecl`).
    /// The same name aliases itself trivially. The fact lets
    /// alias-aware passes (load elision, store-to-load forwarding,
    /// dead-store elimination) assume non-aliasing without paying
    /// for the full downstream points-to analysis on the unique slice.
    ///
    /// Returns `true` iff `buf_a != buf_b` AND both names appear in
    /// the program's `buffer_refs` column (so they're real declared
    /// buffers, not phantom or extension-defined names).
    #[must_use]
    pub fn buffers_provably_distinct(&self, buf_a: &str, buf_b: &str) -> bool {
        if buf_a == buf_b {
            return false;
        }
        let a_seen = self.buffer_refs.iter().any(|(_, b, _)| b.as_str() == buf_a);
        let b_seen = self.buffer_refs.iter().any(|(_, b, _)| b.as_str() == buf_b);
        a_seen && b_seen
    }

    /// escape fact: `true` iff `name`'s contents are
    /// observable outside this kernel's execution.
    ///
    /// A buffer escapes the kernel scope when it appears as:
    ///   - the destination of any `Node::Store`, `Node::AsyncStore`,
    ///     `Node::AsyncLoad` (the host reads back the destination),
    ///   - the index target of any `Expr::Atomic` (atomic results are
    ///     visible to other workgroups + the host),
    ///   - the count buffer of any `Node::IndirectDispatch` (the
    ///     value is consumed by the dispatch grid).
    ///
    /// Buffers that are READ ONLY (no Write / Atomic / `AsyncDestination`
    /// / `IndirectCount` in the `buffer_refs` column) do not escape  -  their
    /// contents are an input the host produced, not a kernel-local
    /// scratch the host needs to read back.
    ///
    /// Used by scratch-reuse passes (megakernel arms can recycle the
    /// storage of a non-escaping buffer for the next arm).
    #[must_use]
    pub fn buffer_escapes(&self, name: &str) -> bool {
        self.buffer_refs.iter().any(|(_, b, kind)| {
            b.as_str() == name
                && matches!(
                    kind,
                    BufferRefKind::Write
                        | BufferRefKind::Atomic(_)
                        | BufferRefKind::AsyncDestination
                        | BufferRefKind::IndirectCount
                )
        })
    }

    /// All buffer names that escape the kernel scope (helper for
    /// scratch-reuse passes that want to enumerate the escaping
    /// set in one go).
    #[must_use]
    pub fn escaping_buffers(&self) -> FxHashSet<Ident> {
        let mut out: FxHashSet<Ident> = FxHashSet::default();
        for (_, name, kind) in &self.buffer_refs {
            if matches!(
                kind,
                BufferRefKind::Write
                    | BufferRefKind::Atomic(_)
                    | BufferRefKind::AsyncDestination
                    | BufferRefKind::IndirectCount
            ) {
                out.insert(name.duplicate_handle());
            }
        }
        out
    }
}

fn build_index(rows: &[(NodeIndex, Ident)]) -> FxHashMap<Ident, Vec<NodeIndex>> {
    let mut out: FxHashMap<Ident, Vec<NodeIndex>> = FxHashMap::default();
    for (idx, name) in rows {
        out.entry(name.duplicate_handle()).or_default().push(*idx);
    }
    out
}
