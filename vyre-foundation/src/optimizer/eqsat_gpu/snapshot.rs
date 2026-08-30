//! The columnar mirror of an e-graph.
//!
//! One row per e-node, `(eclass_id, language_op_id, children_offset,
//! children_len)`, with the child indices in a column of their own. A subgroup
//! reading thirty-two consecutive rows touches one cache line per column, which
//! is the whole reason the mirror exists.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::device_image::{append_words, GpuEGraphDeviceImage, GpuEGraphDeviceLayout};
use super::error::{
    u32_len, GpuEGraphDeviceImageError, GpuEGraphSnapshotError, GpuEGraphSnapshotIntegrityError,
};
use super::signature::egraph_row_signature;
use crate::optimizer::eqsat::{EGraph, ENodeLang};

/// GPU-resident snapshot row: one entry per node in the e-graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SnapshotRow {
    /// E-class id this node belongs to (post-canonicalisation).
    pub eclass_id: u32,
    /// Stable language-op id (e.g. `BinOp::Add` → 1, `Load` → 2).
    /// The `OpIdRegistry` maintains the assignment.
    pub language_op_id: u32,
    /// Offset into the snapshot's `children` column where this
    /// node's child eclass ids start.
    pub children_offset: u32,
    /// Number of children (consecutive in the `children` column).
    pub children_len: u32,
}

/// One discovered equivalence (e-class merge candidate) produced by a
/// saturation pass. The CPU merges these back into the `EGraph`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Equivalence {
    /// Left e-class id.
    pub left: u32,
    /// Right e-class id (to be merged with left).
    pub right: u32,
}

/// Columnar GPU-uploadable mirror of an e-graph.
#[derive(Clone, Debug, Default)]
pub struct GpuEGraphSnapshot {
    /// Per-node rows in `(eclass_id, language_op_id, offset, len)` form.
    pub rows: Vec<SnapshotRow>,
    /// Flat children column. `rows[i]` references children at
    /// `children[rows[i].children_offset..rows[i].children_offset + rows[i].children_len]`.
    pub children: Vec<u32>,
    /// Op-id assignment used by `language_op_id`. Stable for the
    /// life of the snapshot.
    pub op_ids: OpIdRegistry,
}

/// Stable language-op id assignment used inside snapshot rows.
#[derive(Clone, Debug, Default)]
pub struct OpIdRegistry {
    by_name: FxHashMap<Arc<str>, u32>,
    names: Vec<Arc<str>>,
}

impl OpIdRegistry {
    /// Intern a language-op name and return its stable id.
    /// Repeated calls with the same name return the same id.
    ///
    /// Returns `u32::MAX` only when the registry has exceeded the current
    /// 32-bit GPU column ABI. Use [`Self::try_intern`] when the exact overflow
    /// reason must be surfaced.
    pub fn intern(&mut self, name: &str) -> u32 {
        match self.try_intern(name) {
            Ok(id) => id,
            Err(_) => u32::MAX,
        }
    }

    /// Fallible form of [`Self::intern`] for GPU snapshot builders that must
    /// reject over-wide op dictionaries instead of silently saturating ids.
    pub fn try_intern(&mut self, name: &str) -> Result<u32, GpuEGraphSnapshotError> {
        if let Some(&id) = self.by_name.get(name) {
            return Ok(id);
        }
        let id = u32_len(self.names.len(), "op-id registry")?;
        let name: Arc<str> = Arc::from(name);
        self.names.push(Arc::clone(&name));
        self.by_name.insert(name, id);
        Ok(id)
    }

    /// Resolve an op-id back to its name, or `None` if unknown.
    #[must_use]
    pub fn name_of(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(AsRef::as_ref)
    }

    /// Number of registered op names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// `true` iff zero op names registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl GpuEGraphSnapshot {
    /// Build a snapshot from a sequence of `(eclass_id, op_name,
    /// children: &[u32])` triples. Caller-driven construction so
    /// this module doesn't depend on the exact `eqsat::EGraph`
    /// internal shape; the `EGraph` crate's adapter calls this
    /// builder to materialise the GPU mirror.
    ///
    /// Returns an empty snapshot if the input exceeds the current 32-bit GPU
    /// column ABI. Use [`Self::try_build`] for actionable overflow diagnostics.
    #[must_use]
    pub fn build<'a, I>(rows: I) -> Self
    where
        I: IntoIterator<Item = (u32, &'a str, &'a [u32])>,
    {
        Self::try_build(rows).unwrap_or_default()
    }

    /// Fallible form of [`Self::build`] that rejects snapshots too large for
    /// the current 32-bit GPU column ABI.
    pub fn try_build<'a, I>(rows: I) -> Result<Self, GpuEGraphSnapshotError>
    where
        I: IntoIterator<Item = (u32, &'a str, &'a [u32])>,
    {
        let mut snapshot = Self::default();
        let rows = rows.into_iter();
        let (lower_bound, _) = rows.size_hint();
        snapshot.rows.reserve(lower_bound);
        for (eclass_id, op_name, kids) in rows {
            let language_op_id = snapshot.op_ids.try_intern(op_name)?;
            let children_offset = u32_len(snapshot.children.len(), "GPU egraph children offset")?;
            let children_len = u32_len(kids.len(), "GPU egraph row child count")?;
            snapshot.children.extend_from_slice(kids);
            snapshot.rows.push(SnapshotRow {
                eclass_id,
                language_op_id,
                children_offset,
                children_len,
            });
        }
        Ok(snapshot)
    }

    /// Materialise a snapshot directly from the CPU `EGraph`.
    ///
    /// The caller supplies the stable operation-name projection because
    /// `ENodeLang` is intentionally domain-generic and does not require
    /// `Debug` or a string identity. Child ids are canonicalized during the
    /// copy so the GPU columns match the CPU graph's current union-find state.
    ///
    /// Returns an empty snapshot if the CPU e-graph exceeds the current 32-bit
    /// GPU column ABI. Use [`Self::try_from_egraph_with`] for actionable
    /// overflow diagnostics.
    #[must_use]
    pub fn from_egraph_with<L, F, S>(egraph: &EGraph<L>, mut op_name: F) -> Self
    where
        L: ENodeLang,
        F: FnMut(&L) -> S,
        S: AsRef<str>,
    {
        Self::try_from_egraph_with(egraph, &mut op_name).unwrap_or_default()
    }

    /// Fallible form of [`Self::from_egraph_with`] that rejects CPU e-graphs
    /// whose node or child-column counts exceed the current 32-bit GPU ABI.
    pub fn try_from_egraph_with<L, F, S>(
        egraph: &EGraph<L>,
        mut op_name: F,
    ) -> Result<Self, GpuEGraphSnapshotError>
    where
        L: ENodeLang,
        F: FnMut(&L) -> S,
        S: AsRef<str>,
    {
        let mut snapshot = Self::default();
        snapshot.rows.reserve(egraph.class_count());
        for (eclass_id, node) in egraph.iter_nodes() {
            let language_op_id = snapshot.op_ids.try_intern(op_name(node).as_ref())?;
            let children = node.children();
            let children_offset = u32_len(snapshot.children.len(), "GPU egraph children offset")?;
            let children_len = u32_len(children.len(), "GPU egraph row child count")?;
            snapshot
                .children
                .extend(children.iter().map(|child| egraph.find_immut(*child).0));
            snapshot.rows.push(SnapshotRow {
                eclass_id: egraph.find_immut(eclass_id).0,
                language_op_id,
                children_offset,
                children_len,
            });
        }
        Ok(snapshot)
    }

    /// Number of nodes in the snapshot.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.rows.len()
    }

    /// `true` iff the snapshot contains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Total number of children references across all rows.
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Children of the row at `row_idx`, or `None` if the snapshot row
    /// references an invalid range.
    #[must_use]
    pub fn children_of(&self, row_idx: usize) -> Option<&[u32]> {
        let row = self.rows.get(row_idx)?;
        let start = row.children_offset as usize;
        let end = start.checked_add(row.children_len as usize)?;
        self.children.get(start..end)
    }

    /// Group rows by their `eclass_id`, returning a map of
    /// `eclass_id → Vec<row_idx>`. Useful for the GPU saturation
    /// kernel's per-eclass passes.
    #[must_use]
    pub fn rows_by_eclass(&self) -> FxHashMap<u32, Vec<usize>> {
        let mut out: FxHashMap<u32, Vec<usize>> =
            FxHashMap::with_capacity_and_hasher(self.rows.len(), Default::default());
        for (i, row) in self.rows.iter().enumerate() {
            out.entry(row.eclass_id).or_default().push(i);
        }
        out
    }

    /// Validate that the columnar snapshot is safe to upload to a GPU kernel.
    ///
    /// This checks every row's operation id, child-column range, and child
    /// e-class references. It is intentionally stricter than construction:
    /// callers may still build partial test fixtures, but production upload
    /// paths can require this gate before device execution.
    ///
    /// # Errors
    ///
    /// Returns [`GpuEGraphSnapshotIntegrityError`] when a row references an
    /// unknown op id, points outside the child column, or names a child e-class
    /// not present in the snapshot.
    pub fn validate_integrity(&self) -> Result<(), GpuEGraphSnapshotIntegrityError> {
        let mut eclasses: FxHashSet<u32> =
            FxHashSet::with_capacity_and_hasher(self.rows.len(), Default::default());
        for row in &self.rows {
            eclasses.insert(row.eclass_id);
        }
        for (row_idx, row) in self.rows.iter().enumerate() {
            if self.op_ids.name_of(row.language_op_id).is_none() {
                return Err(GpuEGraphSnapshotIntegrityError::new(
                    "unknown language_op_id",
                    row_idx,
                    row.language_op_id,
                ));
            }
            let start = row.children_offset as usize;
            let end = start
                .checked_add(row.children_len as usize)
                .ok_or_else(|| {
                    GpuEGraphSnapshotIntegrityError::new(
                        "children range overflow",
                        row_idx,
                        row.children_len,
                    )
                })?;
            if end > self.children.len() {
                return Err(GpuEGraphSnapshotIntegrityError::new(
                    "children range end",
                    row_idx,
                    row.children_len,
                ));
            }
            for &child in &self.children[start..end] {
                if !eclasses.contains(&child) {
                    return Err(GpuEGraphSnapshotIntegrityError::new(
                        "dangling child eclass",
                        row_idx,
                        child,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Pack the validated snapshot into one backend-uploadable u32 slab.
    ///
    /// The image contains row metadata columns, a structural row-signature
    /// column, the flat child column, and a deterministic e-class-to-row prefix
    /// index. A driver uploads the returned [`GpuEGraphDeviceImage::words`]
    /// slice in one copy and passes the spans from
    /// [`GpuEGraphDeviceImage::layout`] as kernel parameters.
    ///
    /// # Errors
    ///
    /// Returns [`GpuEGraphDeviceImageError`] if the snapshot fails integrity
    /// validation or if a derived row/group index exceeds the u32 device ABI.
    pub fn try_pack_device_image(&self) -> Result<GpuEGraphDeviceImage, GpuEGraphDeviceImageError> {
        self.validate_integrity()?;

        let mut groups: FxHashMap<u32, Vec<u32>> =
            FxHashMap::with_capacity_and_hasher(self.rows.len(), Default::default());
        for (row_idx, row) in self.rows.iter().enumerate() {
            groups
                .entry(row.eclass_id)
                .or_default()
                .push(u32_len(row_idx, "GPU egraph grouped row index")?);
        }

        let mut group_eclass_ids = groups.keys().copied().collect::<Vec<_>>();
        group_eclass_ids.sort_unstable();

        let mut group_offsets = Vec::with_capacity(group_eclass_ids.len() + 1);
        let mut group_rows = Vec::with_capacity(self.rows.len());
        for eclass_id in &group_eclass_ids {
            group_offsets.push(u32_len(group_rows.len(), "GPU egraph group row offset")?);
            let Some(rows) = groups.get(eclass_id) else {
                return Err(GpuEGraphSnapshotIntegrityError::new(
                    "missing grouped eclass key",
                    0,
                    *eclass_id,
                )
                .into());
            };
            group_rows.extend_from_slice(rows);
        }
        group_offsets.push(u32_len(
            group_rows.len(),
            "GPU egraph group row terminal offset",
        )?);

        let row_signatures = self
            .rows
            .iter()
            .map(|row| {
                let start = row.children_offset as usize;
                let end = start + row.children_len as usize;
                egraph_row_signature(row, &self.children[start..end])
            })
            .collect::<Vec<_>>();
        let mut words = Vec::with_capacity(
            self.rows.len() * 5
                + self.children.len()
                + group_eclass_ids.len()
                + group_offsets.len()
                + group_rows.len(),
        );
        let row_eclass_ids = append_words(&mut words, self.rows.iter().map(|row| row.eclass_id));
        let row_language_op_ids =
            append_words(&mut words, self.rows.iter().map(|row| row.language_op_id));
        let row_children_offsets =
            append_words(&mut words, self.rows.iter().map(|row| row.children_offset));
        let row_children_lens =
            append_words(&mut words, self.rows.iter().map(|row| row.children_len));
        let row_signatures = append_words(&mut words, row_signatures);
        let children = append_words(&mut words, self.children.iter().copied());
        let group_eclass_ids_span = append_words(&mut words, group_eclass_ids);
        let group_offsets = append_words(&mut words, group_offsets);
        let group_rows = append_words(&mut words, group_rows);

        Ok(GpuEGraphDeviceImage {
            words,
            layout: GpuEGraphDeviceLayout {
                row_count: self.rows.len(),
                child_count: self.children.len(),
                eclass_group_count: groups.len(),
                row_eclass_ids,
                row_language_op_ids,
                row_children_offsets,
                row_children_lens,
                row_signatures,
                children,
                group_eclass_ids: group_eclass_ids_span,
                group_offsets,
                group_rows,
            },
        })
    }
}
