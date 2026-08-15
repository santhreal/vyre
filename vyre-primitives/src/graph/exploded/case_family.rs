//! The generated exploded-IFDS graph family every sweep over this primitive walks.
//!
//! Two crates derived the same graphs from a case index with the same eight
//! lines of arithmetic: the owner's allocating-versus-reusable oracle sweep and
//! a consumer's encode-dispatch-decode sweep. Nothing tied the two copies
//! together, so they had already drifted in how far they ran, 1024 cases against
//! 512, with no reason recorded on either side, and a fix to one generator's
//! edge rule would have narrowed the other's coverage while both kept passing.
//!
//! The family is a property of the exploded supergraph, not of either sweep, so
//! it is named here once. The generator is deliberately a pure function of the
//! case index rather than a random source: a failing case number reproduces the
//! graph exactly, from any crate, with no seed to carry.

/// One generated exploded-IFDS graph.
///
/// The four rule lists are what a positional call cannot distinguish:
/// `intra_edges`, `flow_gen` and `flow_kill` are all `[(u32, u32, u32)]`, so
/// transposing two of them at a call site compiles and inverts GEN against
/// KILL. Owning them as named fields is what makes such a swap a diff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExplodedIfdsCase {
    /// Procedures in the module.
    pub num_procs: u32,
    /// Basic blocks in every procedure.
    pub blocks_per_proc: u32,
    /// Dataflow facts tracked per procedure.
    pub facts_per_proc: u32,
    /// Intraprocedural control edges, `(proc, src_block, dst_block)`.
    pub intra_edges: Vec<(u32, u32, u32)>,
    /// Call edges, `(src_proc, src_block, dst_proc, dst_block)`.
    pub inter_edges: Vec<(u32, u32, u32, u32)>,
    /// GEN bits, `(proc, block, fact)`.
    pub flow_gen: Vec<(u32, u32, u32)>,
    /// KILL bits, `(proc, block, fact)`.
    pub flow_kill: Vec<(u32, u32, u32)>,
}

/// Cases a full sweep runs.
///
/// The generator repeats with a period of 150, so this covers every distinct
/// graph the family can produce almost seven times over. A shorter sweep is not
/// cheaper in coverage terms and a longer one buys nothing.
pub const EXPLODED_IFDS_CASE_COUNT: usize = 1024;

/// The generated graph at `case`.
///
/// Total order: `case` alone determines the extents, every edge and every flow
/// bit, so a sweep is reproducible and a reported case number is a complete
/// repro.
#[must_use]
pub fn exploded_ifds_case(case: usize) -> ExplodedIfdsCase {
    let num_procs = 1 + (case % 3) as u32;
    let blocks_per_proc = 1 + ((case / 3) % 5) as u32;
    let facts_per_proc = 1 + ((case / 15) % 5) as u32;
    let mut intra_edges = Vec::new();
    let mut inter_edges = Vec::new();
    let mut flow_gen = Vec::new();
    let mut flow_kill = Vec::new();

    for proc_id in 0..num_procs {
        for block in 0..blocks_per_proc {
            let next_block = (block + 1) % blocks_per_proc;
            let mixed = case
                .wrapping_mul(37)
                .wrapping_add((proc_id as usize).wrapping_mul(11))
                .wrapping_add((block as usize).wrapping_mul(7));
            if blocks_per_proc > 1 && mixed % 2 == 0 {
                intra_edges.push((proc_id, block, next_block));
            }
            let fact = (mixed as u32) % facts_per_proc;
            if mixed % 3 == 0 {
                flow_gen.push((proc_id, block, fact));
            }
            if mixed % 5 == 0 && fact != 0 {
                flow_kill.push((proc_id, block, fact));
            }
        }
    }
    for proc_id in 0..num_procs.saturating_sub(1) {
        if (case + proc_id as usize) % 2 == 0 {
            inter_edges.push((proc_id, 0, proc_id + 1, 0));
        }
    }

    ExplodedIfdsCase {
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
    }
}
