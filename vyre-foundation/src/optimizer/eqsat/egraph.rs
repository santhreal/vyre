//! Hashcons, union-find, and rebuild: the `EGraph` operations themselves.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::class_index::{
    eclass_index, reserve_hashcons, reserve_vec_exact, try_dedup_enodes_by_hash,
    try_eclass_id_from_index,
};
use super::{log_egraph_compat_error, EChildren, EClass, EClassId, EGraph, EGraphError, ENodeLang};

impl<L: ENodeLang> Default for EGraph<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: ENodeLang> EGraph<L> {
    /// Create an empty `EGraph`.
    #[must_use]
    pub fn new() -> Self {
        Self::empty_unreserved()
    }

    /// Create an `EGraph` with capacity for an expected number of `EClasses`.
    #[must_use]
    pub fn with_capacity(class_capacity: usize) -> Self {
        match Self::try_with_capacity(class_capacity) {
            Ok(egraph) => egraph,
            Err(error) => {
                log_egraph_compat_error("egraph with_capacity", &error);
                Self::empty_unreserved()
            }
        }
    }

    /// Fallible variant of [`Self::with_capacity`].
    pub fn try_with_capacity(class_capacity: usize) -> Result<Self, EGraphError> {
        let mut classes = Vec::new();
        reserve_vec_exact(&mut classes, class_capacity, "egraph class storage")?;
        let mut hashcons = FxHashMap::default();
        reserve_hashcons(&mut hashcons, class_capacity, "egraph hashcons storage")?;
        let mut parent = Vec::new();
        reserve_vec_exact(
            &mut parent,
            class_capacity,
            "egraph union-find parent storage",
        )?;
        let mut pending = Vec::new();
        reserve_vec_exact(&mut pending, class_capacity, "egraph rebuild queue storage")?;
        Ok(Self {
            classes,
            hashcons,
            parent,
            pending,
        })
    }

    fn empty_unreserved() -> Self {
        Self {
            classes: Vec::new(),
            hashcons: FxHashMap::default(),
            parent: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// Number of `EClasses` currently in the graph.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Find the canonical class representative via path compression.
    pub fn find(&mut self, id: EClassId) -> EClassId {
        match self.try_find(id) {
            Ok(found) => found,
            Err(error) => {
                log_egraph_compat_error("egraph find", &error);
                id
            }
        }
    }

    /// Fallible variant of [`Self::find`].
    pub fn try_find(&mut self, id: EClassId) -> Result<EClassId, EGraphError> {
        let mut cur = id;
        loop {
            let cur_idx = eclass_index(cur, self.parent.len(), "egraph find")?;
            let parent = self.parent[cur_idx];
            if parent == cur {
                break;
            }
            cur = parent;
        }
        // Path compression.
        let mut walk = id;
        loop {
            let walk_idx = eclass_index(walk, self.parent.len(), "egraph path compression")?;
            let next = self.parent[walk_idx];
            if next == cur {
                break;
            }
            self.parent[walk_idx] = cur;
            walk = next;
        }
        Ok(cur)
    }

    /// Find a canonical class without path compression  -  for read-only
    /// use during iteration.
    #[must_use]
    pub fn find_immut(&self, id: EClassId) -> EClassId {
        match self.try_find_immut(id) {
            Ok(found) => found,
            Err(error) => {
                log_egraph_compat_error("egraph immutable find", &error);
                id
            }
        }
    }

    /// Fallible variant of [`Self::find_immut`].
    pub fn try_find_immut(&self, id: EClassId) -> Result<EClassId, EGraphError> {
        let mut cur = id;
        loop {
            let cur_idx = eclass_index(cur, self.parent.len(), "egraph immutable find")?;
            let parent = self.parent[cur_idx];
            if parent == cur {
                break;
            }
            cur = parent;
        }
        Ok(cur)
    }

    /// Canonicalize a node by replacing each child with its current
    /// canonical `EClass`.
    fn canonicalize(&self, node: &L) -> L {
        match self.try_canonicalize(node) {
            Ok(canonical) => canonical,
            Err(error) => {
                log_egraph_compat_error("egraph canonicalize", &error);
                node.clone()
            }
        }
    }

    fn try_canonicalize(&self, node: &L) -> Result<L, EGraphError> {
        let canon_children: EChildren = node
            .children()
            .into_iter()
            .map(|c| self.try_find_immut(c))
            .collect::<Result<_, _>>()?;
        Ok(node.with_children(&canon_children))
    }

    /// Add a node to the `EGraph`. If an equivalent node already exists,
    /// return its `EClassId`; otherwise create a new `EClass`.
    pub fn add(&mut self, node: L) -> EClassId {
        match self.try_add(node) {
            Ok(id) => id,
            Err(error) => {
                log_egraph_compat_error("egraph add", &error);
                EClassId(0)
            }
        }
    }

    /// Fallible variant of [`Self::add`].
    #[expect(
        clippy::needless_pass_by_value,
        reason = "public insertion API consumes language nodes; canonicalized misses store an owned node"
    )]
    pub fn try_add(&mut self, node: L) -> Result<EClassId, EGraphError> {
        let canon = self.try_canonicalize(&node)?;
        if let Some(&existing) = self.hashcons.get(&canon) {
            return self.try_find(existing);
        }
        let new_id = try_eclass_id_from_index(self.classes.len())?;
        let canon_children = canon.children();
        reserve_vec_exact(&mut self.parent, 1, "egraph parent insertion")?;
        reserve_vec_exact(&mut self.classes, 1, "egraph class insertion")?;
        reserve_hashcons(&mut self.hashcons, 1, "egraph hashcons insertion")?;
        let mut child_indices: SmallVec<[(usize, EClassId); 4]> = SmallVec::new();
        for child in &canon_children {
            let child_canon = self.try_find(*child)?;
            let child_idx = eclass_index(
                child_canon,
                self.classes.len(),
                "egraph child parent registration",
            )?;
            child_indices.push((child_idx, child_canon));
        }
        for (position, (child_idx, _)) in child_indices.iter().enumerate() {
            if child_indices[..position]
                .iter()
                .any(|(seen_idx, _)| seen_idx == child_idx)
            {
                continue;
            }
            let occurrences = child_indices
                .iter()
                .filter(|(seen_idx, _)| seen_idx == child_idx)
                .count();
            reserve_vec_exact(
                &mut self.classes[*child_idx].parents,
                occurrences,
                "egraph child parent registration",
            )?;
        }
        let mut nodes = Vec::new();
        reserve_vec_exact(&mut nodes, 1, "egraph singleton enode storage")?;
        nodes.push(canon.clone());
        self.parent.push(new_id);
        // Register `new_id` as a parent of each child class.
        for (child_idx, _) in child_indices {
            self.classes[child_idx].parents.push(new_id);
        }
        self.classes.push(EClass {
            nodes,
            parents: Vec::new(),
        });
        self.hashcons.insert(canon, new_id);
        Ok(new_id)
    }

    /// Equate two `EClasses`. The returned id is the canonical class for
    /// both inputs after the union. Calls to `add()` on equivalent nodes
    /// will return this same id.
    ///
    /// Caller must invoke `rebuild()` after a batch of `union()` calls
    /// to re-canonicalize the hashcons + propagate equivalences upward
    /// through parent pointers.
    pub fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        match self.try_union(a, b) {
            Ok(id) => id,
            Err(error) => {
                log_egraph_compat_error("egraph union", &error);
                a
            }
        }
    }

    /// Fallible variant of [`Self::union`].
    pub fn try_union(&mut self, a: EClassId, b: EClassId) -> Result<EClassId, EGraphError> {
        let a_root = self.try_find(a)?;
        let b_root = self.try_find(b)?;
        if a_root == b_root {
            return Ok(a_root);
        }
        // Union with the smaller-id-as-root convention for determinism.
        let (winner, loser) = if a_root.0 < b_root.0 {
            (a_root, b_root)
        } else {
            (b_root, a_root)
        };
        let winner_idx = eclass_index(winner, self.classes.len(), "egraph union winner")?;
        let loser_idx = eclass_index(loser, self.classes.len(), "egraph union loser")?;
        let loser_nodes_len = self.classes[loser_idx].nodes.len();
        let loser_parents_len = self.classes[loser_idx].parents.len();
        reserve_vec_exact(
            &mut self.classes[winner_idx].nodes,
            loser_nodes_len,
            "egraph union node merge",
        )?;
        reserve_vec_exact(
            &mut self.classes[winner_idx].parents,
            loser_parents_len,
            "egraph union parent merge",
        )?;
        reserve_vec_exact(&mut self.pending, 1, "egraph rebuild queue push")?;
        self.parent[loser_idx] = winner;
        // Merge nodes + parent lists into the winning class.
        let loser_class = std::mem::replace(
            &mut self.classes[loser_idx],
            EClass {
                nodes: Vec::new(),
                parents: Vec::new(),
            },
        );
        self.classes[winner_idx].nodes.extend(loser_class.nodes);
        self.classes[winner_idx].parents.extend(loser_class.parents);
        // Schedule the winner for rebuild  -  its parents may now be
        // canonicalizable.
        self.pending.push(winner);
        Ok(winner)
    }

    /// Re-canonicalize the hashcons after a batch of `union()` calls.
    /// Returns the number of additional unions discovered transitively.
    pub fn rebuild(&mut self) -> usize {
        match self.try_rebuild() {
            Ok(count) => count,
            Err(error) => {
                log_egraph_compat_error("egraph rebuild", &error);
                0
            }
        }
    }

    /// Fallible variant of [`Self::rebuild`].
    pub fn try_rebuild(&mut self) -> Result<usize, EGraphError> {
        let mut new_unions = 0;
        while let Some(class_id) = self.pending.pop() {
            let canonical = self.try_find(class_id)?;
            let canonical_idx = eclass_index(canonical, self.classes.len(), "egraph rebuild")?;
            let nodes_len = self.classes[canonical_idx].nodes.len();
            let mut canon_nodes = Vec::new();
            reserve_vec_exact(
                &mut canon_nodes,
                nodes_len,
                "egraph rebuild canonical node staging",
            )?;
            reserve_hashcons(
                &mut self.hashcons,
                nodes_len,
                "egraph rebuild hashcons staging",
            )?;
            // Re-canonicalize every node in the canonical class.
            let nodes = std::mem::take(&mut self.classes[canonical_idx].nodes);
            for node in nodes {
                let new_canon = self.try_canonicalize(&node)?;
                // Re-insert into hashcons; collisions trigger more unions.
                if let Some(&existing) = self.hashcons.get(&new_canon) {
                    let existing_canon = self.try_find(existing)?;
                    if existing_canon != canonical {
                        let unified = self.try_union(existing_canon, canonical)?;
                        new_unions += 1;
                        if unified != canonical {
                            // The winner changed  -  re-find at top of loop.
                            reserve_vec_exact(
                                &mut self.pending,
                                1,
                                "egraph rebuild winner reschedule",
                            )?;
                            self.pending.push(unified);
                        }
                    }
                }
                self.hashcons.insert(new_canon.clone(), canonical);
                canon_nodes.push(new_canon);
            }
            try_dedup_enodes_by_hash(&mut canon_nodes)?;
            self.classes[canonical_idx].nodes = canon_nodes;
        }
        Ok(new_unions)
    }

    /// Iterate every (`EClassId`, `ENode`) pair currently in the graph.
    /// Useful for rule application and extraction.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (EClassId, &L)> {
        self.classes
            .iter()
            .enumerate()
            .filter_map(|(idx, class)| {
                let class_id = match try_eclass_id_from_index(idx) {
                    Ok(class_id) => class_id,
                    Err(error) => {
                        log_egraph_compat_error("egraph iter_nodes class id", &error);
                        return None;
                    }
                };
                (self.parent[idx] == class_id).then_some((class_id, class))
            })
            .flat_map(|(class_id, class)| class.nodes.iter().map(move |n| (class_id, n)))
    }

    /// Read-only access to a class by id.
    #[must_use]
    pub fn class(&self, id: EClassId) -> Option<&EClass<L>> {
        match self.try_class(id) {
            Ok(class) => class,
            Err(error) => {
                log_egraph_compat_error("egraph class lookup", &error);
                None
            }
        }
    }

    /// Fallible variant of [`Self::class`].
    pub fn try_class(&self, id: EClassId) -> Result<Option<&EClass<L>>, EGraphError> {
        let canon = self.try_find_immut(id)?;
        let idx = eclass_index(canon, self.classes.len(), "egraph class lookup")?;
        Ok(self.classes.get(idx))
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use super::super::arith_fixture::Arith;
    use super::super::{EClassId, EGraph, EGraphError};

    #[test]
    fn empty_egraph_has_zero_classes() {
        let egraph: EGraph<Arith> = EGraph::new();
        assert_eq!(egraph.class_count(), 0);
    }

    #[test]
    fn add_const_creates_one_class() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _ = egraph.add(Arith::Const(7));
        assert_eq!(egraph.class_count(), 1);
    }

    #[test]
    fn add_same_const_twice_returns_same_class() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(7));
        let b = egraph.add(Arith::Const(7));
        assert_eq!(a, b);
        assert_eq!(egraph.class_count(), 1);
    }

    #[test]
    fn add_distinct_consts_creates_distinct_classes() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(7));
        let b = egraph.add(Arith::Const(8));
        assert_ne!(a, b);
        assert_eq!(egraph.class_count(), 2);
    }

    #[test]
    fn add_compound_node_creates_proper_class() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(1));
        let b = egraph.add(Arith::Const(2));
        let sum = egraph.add(Arith::Add(a, b));
        assert_eq!(egraph.class_count(), 3);
        assert_ne!(sum, a);
        assert_ne!(sum, b);
    }

    #[test]
    fn union_merges_two_classes() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(1));
        let b = egraph.add(Arith::Const(2));
        let unified = egraph.union(a, b);
        assert_eq!(egraph.find(a), unified);
        assert_eq!(egraph.find(b), unified);
    }

    #[test]
    fn union_is_idempotent() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(1));
        let b = egraph.add(Arith::Const(2));
        let first = egraph.union(a, b);
        let second = egraph.union(a, b);
        assert_eq!(first, second);
    }

    #[test]
    fn rebuild_canonicalizes_compound_nodes_after_union() {
        // Build (1 + 2). Union 1 and 2. After rebuild, two adds that look
        // structurally different should canonicalize to the same form.
        let mut egraph: EGraph<Arith> = EGraph::new();
        let one = egraph.add(Arith::Const(1));
        let two = egraph.add(Arith::Const(2));
        let _add_12 = egraph.add(Arith::Add(one, two));
        let _add_22 = egraph.add(Arith::Add(two, two));
        egraph.union(one, two);
        let _ = egraph.rebuild();
        // After rebuild, Add(1,2) and Add(2,2) canonicalize to the same
        // pair of children → same EClass.
        let post_one = egraph.find(one);
        let post_two = egraph.find(two);
        assert_eq!(post_one, post_two, "1 and 2 must be in the same class");
    }

    #[test]
    fn find_immut_returns_canonical_after_union() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(1));
        let b = egraph.add(Arith::Const(2));
        egraph.union(a, b);
        // find_immut must agree with find.
        let canon_a = egraph.find_immut(a);
        let canon_b = egraph.find_immut(b);
        assert_eq!(canon_a, canon_b);
    }

    #[test]
    fn class_lookup_returns_canonical_class() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(7));
        let class = egraph.class(a).expect("Fix: class must exist");
        assert!(matches!(class.nodes[0], Arith::Const(7)));
    }

    #[test]
    fn rebuild_propagates_through_parents() {
        // Build Add(1, 2). Union 1 and 2. After rebuild, Add(1,2) should
        // canonicalize to Add(1,1) (or whichever survived).
        let mut egraph: EGraph<Arith> = EGraph::new();
        let one = egraph.add(Arith::Const(1));
        let two = egraph.add(Arith::Const(2));
        let add_12 = egraph.add(Arith::Add(one, two));
        egraph.union(one, two);
        let _ = egraph.rebuild();
        // The Add(1,2) class should still be findable, and its node should
        // now reference the unified child class.
        let class = egraph.class(add_12).expect("Fix: class must still exist");
        match &class.nodes[0] {
            Arith::Add(a, b) => {
                let canon_a = egraph.find_immut(*a);
                let canon_b = egraph.find_immut(*b);
                assert_eq!(
                    canon_a, canon_b,
                    "Add(1,2)'s children must canonicalize to the same class after union"
                );
            }
            other => panic!("expected Add; got {other:?}"),
        }
    }

    #[test]
    fn iter_nodes_visits_only_canonical_classes() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(1));
        let b = egraph.add(Arith::Const(2));
        egraph.union(a, b);
        let _ = egraph.rebuild();
        // iter_nodes yields one entry per (class, node) pair. After union,
        // the loser class is filtered out (its parent points elsewhere),
        // but the merged winner class holds both Const(1) and Const(2)
        // nodes. So the canonical-class set has size 1, but the (class,
        // node) entry count is 2.
        let unique_classes: FxHashSet<EClassId> = egraph.iter_nodes().map(|(cid, _)| cid).collect();
        assert_eq!(
            unique_classes.len(),
            1,
            "post-union iter must visit exactly one canonical class id"
        );
        let total_nodes = egraph.iter_nodes().count();
        assert_eq!(
            total_nodes, 2,
            "the merged class still holds both original nodes (Const(1) + Const(2))"
        );
    }

    #[test]
    fn fallible_find_reports_foreign_class_id() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let err = egraph
            .try_find(EClassId(0))
            .expect_err("empty graph must reject foreign class id 0");
        assert!(
            matches!(
                err,
                EGraphError::ClassIdOutOfBounds {
                    context: "egraph find",
                    id: EClassId(0),
                    len: 0
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fallible_add_rejects_foreign_child_ids() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let err = egraph
            .try_add(Arith::Add(EClassId(0), EClassId(0)))
            .expect_err("foreign children must be rejected before insertion");
        assert!(
            matches!(
                err,
                EGraphError::ClassIdOutOfBounds {
                    context: "egraph immutable find",
                    id: EClassId(0),
                    len: 0
                }
            ),
            "unexpected error: {err}"
        );
        assert_eq!(
            egraph.class_count(),
            0,
            "failed fallible insertion must not allocate a partial class"
        );
    }

    #[test]
    fn fallible_add_handles_duplicate_children_without_late_allocation_path() {
        let mut egraph: EGraph<Arith> = EGraph::try_with_capacity(2)
            .expect("Fix: unit-test oracle precondition - small egraph reservation must succeed");
        let one = egraph
            .try_add(Arith::Const(1))
            .expect("Fix: unit-test oracle precondition - const insert must succeed");
        let add = egraph
            .try_add(Arith::Add(one, one))
            .expect("Fix: unit-test oracle precondition - duplicate child registration must be pre-reserved");
        let class = egraph
            .try_class(add)
            .expect("Fix: unit-test oracle precondition - class lookup must be valid")
            .expect("Fix: unit-test oracle precondition - class must exist");
        assert!(matches!(class.nodes[0], Arith::Add(_, _)));
    }
}
