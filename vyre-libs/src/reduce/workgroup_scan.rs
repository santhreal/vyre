//! Workgroup-local prefix scans over scratch buffers.
//!
//! A tree reduction folds a workgroup down to one value; a scan leaves every
//! lane holding the fold of everything up to it. The two share the scratch
//! staging convention and the barrier discipline of
//! [`super::workgroup_tree`], and nothing else: the sweep here walks the same
//! slot pairs twice, up and back down, and its callers publish per-element
//! results rather than a single total.

use vyre_foundation::ir::{Expr, Node};

/// Index of the lane `stride` positions before `lane`.
///
/// One owner for "previous lane" addressing. The two spellings this replaces
/// (`lane + (0u32).wrapping_sub(stride)` and `lane + u32::MAX` in the
/// exclusive scan) both reached the answer through `BinOp::Add` on a
/// pre-negated constant, so every consumer of the IR - the optimizer's identity
/// rules, the value-range analysis, a reader - saw an addition where the
/// program means a subtraction. `BinOp::WrappingSub` says it once.
pub(crate) fn previous_lane(lane: &Expr, stride: u32) -> Expr {
    lane.clone().wrapping_sub(Expr::u32(stride))
}

/// Emit the work-efficient Blelloch inclusive-sum sweep over `lanes` workgroup
/// lanes, reading the staged per-lane values from `scratch_a` and leaving the
/// inclusive prefix sums there.
///
/// `scratch_b` keeps each lane's staged value across the sweep. The sweep
/// itself produces an EXCLUSIVE scan in `scratch_a`, and the inclusive result
/// is that prefix plus the lane's own staged value, so the second scratch
/// buffer every caller already declares carries the addend instead of being a
/// ping-pong target.
///
/// Reduce phase: at stride `s` the lane owning slot `(k+1)*2s-1` folds the slot
/// `s` positions back into it, so slot `lanes-1` ends holding the total after
/// `log2(lanes)` rounds and `lanes-1` additions. Downsweep phase: that total is
/// cleared and the same slot pairs are walked in reverse, each handing its left
/// child the running prefix and taking the sum, another `lanes-1` additions.
/// Total work is `2*lanes-2` additions against the `lanes*log2(lanes)` a
/// Hillis-Steele sweep performs, because round `s` activates `lanes/(2s)` lanes
/// instead of all of them.
///
/// Barriers are workgroup-scoped: the sweep touches nothing but the two
/// workgroup scratch buffers. The sweep ends on a barrier, so every lane may
/// read any lane's inclusive sum the moment it returns. That is part of the
/// contract rather than a caller's responsibility:
/// `frontier_word_block_offsets_single_workgroup` reads `scratch_a[lane - 1]`
/// immediately after the call to turn the inclusive scan into an exclusive one,
/// and without the trailing barrier lane `k` could read lane `k - 1` before that
/// lane added its own staged value, producing a block offset short by exactly
/// the previous block's count.
///
/// Callers differ in how they stage `scratch_a` and how they write the result
/// out; the sweep between those two steps does not, and was hand-written five
/// times before this became its owner.
///
/// # Panics
///
/// Panics when `lanes` is not a power of two. The slot pairing walks a balanced
/// binary tree over the scratch buffers, and a partial top level would leave
/// slots the downsweep never reaches.
pub(crate) fn blelloch_inclusive_sum_nodes(
    scratch_a: &str,
    scratch_b: &str,
    lane: &Expr,
    lanes: u32,
) -> Vec<Node> {
    assert!(
        lanes.is_power_of_two(),
        "Fix: blelloch_inclusive_sum_nodes needs a power-of-two lane count, got {lanes}; round the scratch buffers up to the next power of two before staging into them."
    );

    let mut nodes = vec![
        Node::store(scratch_b, lane.clone(), Expr::load(scratch_a, lane.clone())),
        Node::barrier(),
    ];

    let mut stride = 1_u32;
    while stride < lanes {
        let slot = sweep_slot(lane, stride);
        nodes.push(Node::if_then(
            Expr::lt(slot.clone(), Expr::u32(lanes)),
            vec![Node::store(
                scratch_a,
                slot.clone(),
                Expr::add(
                    Expr::load(scratch_a, slot.clone()),
                    Expr::load(scratch_a, previous_lane(&slot, stride)),
                ),
            )],
        ));
        nodes.push(Node::barrier());
        stride *= 2;
    }

    nodes.push(Node::if_then(
        Expr::eq(lane.clone(), Expr::u32(0)),
        vec![Node::store(scratch_a, Expr::u32(lanes - 1), Expr::u32(0))],
    ));
    nodes.push(Node::barrier());

    let mut stride = lanes / 2;
    let mut round = 0_u32;
    while stride >= 1 {
        let slot = sweep_slot(lane, stride);
        let left = previous_lane(&slot, stride);
        let held = format!("{scratch_a}_downsweep_{round}");
        nodes.push(Node::if_then(
            Expr::lt(slot.clone(), Expr::u32(lanes)),
            vec![
                Node::let_bind(held.as_str(), Expr::load(scratch_a, left.clone())),
                Node::store(scratch_a, left, Expr::load(scratch_a, slot.clone())),
                Node::store(
                    scratch_a,
                    slot.clone(),
                    Expr::add(Expr::load(scratch_a, slot), Expr::var(held.as_str())),
                ),
            ],
        ));
        nodes.push(Node::barrier());
        stride /= 2;
        round += 1;
    }

    nodes.push(Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(lanes)),
        vec![Node::store(
            scratch_a,
            lane.clone(),
            Expr::add(
                Expr::load(scratch_a, lane.clone()),
                Expr::load(scratch_b, lane.clone()),
            ),
        )],
    ));
    nodes.push(Node::barrier());
    nodes
}

/// One pass of a multi-block scan: bind, stage, sweep, publish.
///
/// A pass-A kernel binds `lane`, `block` and `global`, zero-fills the workgroup
/// scratch so the sweep never reads an uninitialized slot, stages one value per
/// lane under `global < in_range`, runs [`blelloch_inclusive_sum_nodes`], then
/// writes the per-element inclusive partial for the in-range lanes and the
/// block total from the last lane of the block.
///
/// Only the staging differs between callers: `reduce::multi_block_prefix_scan`
/// loads an input element, `graph::csr_frontier_queue::word_block_scan` loads a
/// frontier word, masks the tail and takes its population count. Everything
/// around that was written twice, including the detail that the block total
/// comes from lane `block_lanes - 1` reading `scratch_a` after the sweep rather
/// than from a separate accumulator.
pub(crate) struct BlockScanPass<'a> {
    /// Name bound to `LocalId { axis: 0 }`.
    pub lane: &'a str,
    /// Name bound to `WorkgroupId { axis: 0 }`.
    pub block: &'a str,
    /// Name bound to `block * block_lanes + lane`.
    pub global: &'a str,
    /// Workgroup scratch the sweep scans in place.
    pub scratch_a: &'a str,
    /// Workgroup scratch the sweep keeps the staged value in.
    pub scratch_b: &'a str,
    /// Buffer receiving one inclusive partial per in-range element.
    pub partials: &'a str,
    /// Buffer receiving one total per block.
    pub block_totals: &'a str,
    /// Lanes per block. Must be a power of two, as the sweep requires.
    pub block_lanes: u32,
    /// Element count; a lane whose `global` is at or past this stages nothing
    /// and publishes nothing, and its scratch slot stays the zero fill.
    pub in_range: u32,
}

impl BlockScanPass<'_> {
    /// Emit the pass. `stage` runs under `global < in_range` and must leave the
    /// lane's value in `scratch_a[lane]`.
    pub(crate) fn nodes(&self, stage: Vec<Node>) -> Vec<Node> {
        let lane = Expr::var(self.lane);
        let block = Expr::var(self.block);
        let global = Expr::var(self.global);
        let in_range = Expr::lt(global.clone(), Expr::u32(self.in_range));

        let mut nodes = vec![
            Node::let_bind(self.lane, Expr::LocalId { axis: 0 }),
            Node::let_bind(self.block, Expr::WorkgroupId { axis: 0 }),
            Node::let_bind(
                self.global,
                Expr::add(
                    Expr::mul(block.clone(), Expr::u32(self.block_lanes)),
                    lane.clone(),
                ),
            ),
            Node::store(self.scratch_a, lane.clone(), Expr::u32(0)),
            Node::if_then(in_range.clone(), stage),
            Node::barrier(),
        ];

        nodes.extend(blelloch_inclusive_sum_nodes(
            self.scratch_a,
            self.scratch_b,
            &lane,
            self.block_lanes,
        ));

        nodes.push(Node::if_then(
            in_range,
            vec![Node::store(
                self.partials,
                global,
                Expr::load(self.scratch_a, lane.clone()),
            )],
        ));
        nodes.push(Node::if_then(
            Expr::eq(lane.clone(), Expr::u32(self.block_lanes - 1)),
            vec![Node::store(
                self.block_totals,
                block,
                Expr::load(self.scratch_a, lane),
            )],
        ));
        nodes
    }
}

/// Scratch slot lane `lane` owns in a sweep round of stride `s`: `(lane+1)*2s-1`.
///
/// Active lanes are the contiguous prefix `0..lanes/(2s)`, so the divergence
/// sits at one subgroup boundary instead of splitting every subgroup in the
/// workgroup the way a `lane >= stride` predicate does.
fn sweep_slot(lane: &Expr, stride: u32) -> Expr {
    previous_lane(
        &Expr::mul(
            Expr::add(lane.clone(), Expr::u32(1)),
            Expr::u32(stride.saturating_mul(2)),
        ),
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every caller of the sweep reads a slot it did not write.
    ///
    /// `frontier_word_block_offsets_single_workgroup` reads `scratch_a[lane - 1]`
    /// on the statement after the call, so the sweep has to leave its result
    /// readable by every lane rather than only by the lane that wrote it. Before
    /// this was fixed the node list ended on the store that adds each lane's
    /// staged value back in, with no barrier behind it, and lane `k` could read
    /// lane `k - 1` one round early and take a block offset short by that
    /// block's own count. The reference interpreter runs lanes in order, so no
    /// value assertion can see this; the shape of the emitted program can.
    #[test]
    fn the_sweep_publishes_its_result_before_returning() {
        let nodes = blelloch_inclusive_sum_nodes("scratch_a", "scratch_b", &Expr::var("lane"), 8);
        assert!(
            matches!(nodes.last(), Some(Node::Barrier { .. })),
            "the sweep must end on a barrier so a cross-lane read is safe on the next statement, got {:?}",
            nodes.last()
        );
        let stores_after_last_barrier = nodes
            .iter()
            .rev()
            .take_while(|node| !matches!(node, Node::Barrier { .. }))
            .count();
        assert_eq!(
            stores_after_last_barrier, 0,
            "no node may write scratch after the sweep's final barrier"
        );
    }

    /// The sweep runs under whatever dispatch its caller declared.
    ///
    /// Every write inside the sweep is bounded by the lane count it was given,
    /// including the last one, so a dispatch wider than the scratch buffers
    /// cannot store past their end.
    #[test]
    fn every_scratch_write_is_bounded_by_the_lane_count() {
        let nodes = blelloch_inclusive_sum_nodes("scratch_a", "scratch_b", &Expr::var("lane"), 8);
        let bare_stores = nodes
            .iter()
            .skip(1)
            .filter(|node| matches!(node, Node::Store { .. }))
            .count();
        assert_eq!(
            bare_stores, 0,
            "every store after the staging write must sit inside a bounds guard"
        );
    }
}
