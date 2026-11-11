//! The declared exploded-supergraph (IFDS) CPU-reference case table.
//!
//! Two suites drove the same generated case stream against the same CSR
//! contract: the primitive reference in `vyre_libs::graph::exploded` and
//! the substrate consumer in `vyre_libs::graph::dispatch::exploded`. Each
//! carried its own copy of the construction, and the copies had already drifted
//! in breadth: the primitive suite ran 1024 mixed-flow cases and the substrate
//! suite ran 512 of the same stream, so the substrate dispatch path was never
//! asked about the upper half of a corpus its own file claimed to define. Two
//! copies of a case stream are two corpora that only look identical.
//!
//! This module owns which cases exist and what a correct CSR for them looks
//! like ([`ExplodedIfdsCase::assert_csr`]). What each crate asserts stays in
//! that crate: an arm names the builder it pins.
//!
//! [`arm_coverage`] is why a case group cannot be declared and then quietly
//! skipped: each arm records the groups it asserted and the ledger reads this
//! table back at run time, so a group added here with no arm in a crate turns
//! that crate's suite red instead of widening the table for nobody.

use crate::case_table::ArmCoverage;

/// Minimum declared group count, the floor [`arm_coverage`] enforces.
///
/// The table is enumerated by a function, so its failure mode is returning
/// almost nothing: an arm that covers one group out of one is trivially
/// complete. The floor makes a broken table fail instead of reporting a clean
/// sweep of an empty set.
const MIN_DECLARED_GROUPS: usize = 4;

/// Minimum cases one arm must actually assert, across all groups.
const MIN_ASSERTED_CASES: usize = 2_000;

/// Cases in the mixed intra/inter/GEN/KILL stream.
const MIXED_FLOW_CASES: usize = 1024;

/// One declared exploded-IFDS reference case.
///
/// The rule tuples are in the shape every builder in the stack takes: intra
/// edges are `(proc, src_block, dst_block)`, inter edges are
/// `(src_proc, src_block, dst_proc, dst_block)`, and GEN/KILL rules are
/// `(proc, block, fact)`.
#[derive(Clone, Debug)]
pub struct ExplodedIfdsCase {
    /// Case identity for a failure message.
    pub label: String,
    /// Procedures in the supergraph.
    pub num_procs: u32,
    /// Blocks per procedure.
    pub blocks_per_proc: u32,
    /// Dataflow facts per procedure.
    pub facts_per_proc: u32,
    /// Intra-procedural control edges.
    pub intra_edges: Vec<(u32, u32, u32)>,
    /// Inter-procedural call edges.
    pub inter_edges: Vec<(u32, u32, u32, u32)>,
    /// GEN rules, injecting a fact at a block.
    pub flow_gen: Vec<(u32, u32, u32)>,
    /// KILL rules, suppressing a fact at a block.
    pub flow_kill: Vec<(u32, u32, u32)>,
    /// Dense `(src, dst)` edges the built CSR must contain.
    pub required_edges: Vec<(u32, u32)>,
    /// Dense `(src, dst)` edges the built CSR must not contain.
    pub forbidden_edges: Vec<(u32, u32)>,
}

/// One declared case group. `name` is the coverage key an arm records.
pub struct ExplodedIfdsCaseGroup {
    /// Coverage key. Stable: an arm matches on it.
    pub name: &'static str,
    /// Every case in the group.
    pub cases: Vec<ExplodedIfdsCase>,
}

impl ExplodedIfdsCase {
    /// Total dense node count, which is also `row_ptr.len() - 1`.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.num_procs as usize * self.blocks_per_proc as usize * self.facts_per_proc as usize
    }

    /// Dense index of `(proc, block, fact)` in this case's shape.
    #[must_use]
    pub fn dense_index(&self, proc_id: u32, block: u32, fact: u32) -> u32 {
        (proc_id * self.blocks_per_proc + block) * self.facts_per_proc + fact
    }

    /// Assert `(row_ptr, col_idx)` is a well-formed CSR for this case and
    /// carries every required edge and no forbidden edge.
    ///
    /// Column order within a row is deliberately not asserted: the substrate
    /// dispatch path canonicalizes rows and the primitive reference does not, so
    /// an order assertion here would pin one arm's incidental layout onto both.
    ///
    /// `arm` names the builder under test, so a failure says which of the two
    /// crates diverged.
    ///
    /// # Panics
    /// Panics on a row-count mismatch, a non-monotone or out-of-range `row_ptr`,
    /// an out-of-range column, or a required/forbidden edge violation.
    pub fn assert_csr(&self, arm: &str, row_ptr: &[u32], col_idx: &[u32]) {
        let nodes = self.node_count();
        assert_eq!(
            row_ptr.len(),
            nodes + 1,
            "Fix: {arm} emitted {} row offsets for {} node(s) at {}.",
            row_ptr.len(),
            nodes,
            self.label
        );
        assert_eq!(
            row_ptr[nodes] as usize,
            col_idx.len(),
            "Fix: {arm} terminal row offset does not close the column array at {}.",
            self.label
        );
        for (node, window) in row_ptr.windows(2).enumerate() {
            assert!(
                window[0] <= window[1],
                "Fix: {arm} row offsets went backwards at node {node} of {}.",
                self.label
            );
        }
        for &dst in col_idx {
            assert!(
                (dst as usize) < nodes,
                "Fix: {arm} emitted column {dst} outside the {nodes}-node domain at {}.",
                self.label
            );
        }
        for &(src, dst) in &self.required_edges {
            assert!(
                self.row(row_ptr, col_idx, src).contains(&dst),
                "Fix: {arm} dropped required edge {src} -> {dst} at {}.",
                self.label
            );
        }
        for &(src, dst) in &self.forbidden_edges {
            assert!(
                !self.row(row_ptr, col_idx, src).contains(&dst),
                "Fix: {arm} emitted forbidden edge {src} -> {dst} at {}.",
                self.label
            );
        }
    }

    fn row<'a>(&self, row_ptr: &[u32], col_idx: &'a [u32], src: u32) -> &'a [u32] {
        let start = row_ptr[src as usize] as usize;
        let end = row_ptr[src as usize + 1] as usize;
        &col_idx[start..end]
    }
}

/// Every declared exploded-IFDS case group.
///
/// Bounds are the union of what the two copied suites drove, so no arm loses a
/// case to the merge.
#[must_use]
pub fn declared_groups() -> Vec<ExplodedIfdsCaseGroup> {
    vec![
        ExplodedIfdsCaseGroup {
            name: "mixed_flow_stream",
            cases: mixed_flow_stream(),
        },
        ExplodedIfdsCaseGroup {
            name: "dense_chain",
            cases: dense_chain(),
        },
        ExplodedIfdsCaseGroup {
            name: "flow_rule_edges",
            cases: flow_rule_edges(),
        },
        ExplodedIfdsCaseGroup {
            name: "empty_domain",
            cases: vec![ExplodedIfdsCase {
                label: "empty domain 0/0/0".to_string(),
                num_procs: 0,
                blocks_per_proc: 0,
                facts_per_proc: 0,
                intra_edges: Vec::new(),
                inter_edges: Vec::new(),
                flow_gen: Vec::new(),
                flow_kill: Vec::new(),
                required_edges: Vec::new(),
                forbidden_edges: Vec::new(),
            }],
        },
    ]
}

/// Small shapes with intra edges, inter edges, GEN and KILL all interleaved by
/// one mixing function, so consecutive cases differ in which rule kinds fire.
fn mixed_flow_stream() -> Vec<ExplodedIfdsCase> {
    (0..MIXED_FLOW_CASES)
        .map(|case| {
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
                label: format!("mixed flow case {case}"),
                num_procs,
                blocks_per_proc,
                facts_per_proc,
                intra_edges,
                inter_edges,
                flow_gen,
                flow_kill,
                required_edges: Vec::new(),
                forbidden_edges: Vec::new(),
            }
        })
        .collect()
}

/// Every procedure a straight-line chain of blocks, with a call edge from each
/// procedure's exit into the next procedure's entry. Wide in the three
/// dimensions the node encoding packs, so it is where a shape or offset bug in
/// the CSR layout shows up.
fn dense_chain() -> Vec<ExplodedIfdsCase> {
    let mut cases = Vec::new();
    for num_procs in 1u32..=4 {
        for blocks_per_proc in 1u32..=16 {
            for facts_per_proc in 1u32..=16 {
                let intra_edges: Vec<(u32, u32, u32)> = (0..num_procs)
                    .flat_map(|proc_id| {
                        (0..blocks_per_proc.saturating_sub(1))
                            .map(move |block| (proc_id, block, block + 1))
                    })
                    .collect();
                let inter_edges: Vec<(u32, u32, u32, u32)> = (0..num_procs.saturating_sub(1))
                    .map(|proc_id| (proc_id, blocks_per_proc - 1, proc_id + 1, 0))
                    .collect();
                let flow_gen: Vec<(u32, u32, u32)> = if facts_per_proc > 1 {
                    (0..num_procs)
                        .flat_map(|proc_id| {
                            (0..blocks_per_proc)
                                .map(move |block| {
                                    (proc_id, block, (block % (facts_per_proc - 1)) + 1)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let flow_kill: Vec<(u32, u32, u32)> = (0..num_procs)
                    .flat_map(|proc_id| {
                        (0..blocks_per_proc)
                            .filter_map(move |block| {
                                (facts_per_proc > 2 && block % 3 == 0)
                                    .then_some((proc_id, block, 1))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();

                cases.push(ExplodedIfdsCase {
                    label: format!(
                        "dense chain {num_procs} proc / {blocks_per_proc} block / {facts_per_proc} fact"
                    ),
                    num_procs,
                    blocks_per_proc,
                    facts_per_proc,
                    intra_edges,
                    inter_edges,
                    flow_gen,
                    flow_kill,
                    required_edges: Vec::new(),
                    forbidden_edges: Vec::new(),
                });
            }
        }
    }
    cases
}

/// Hand-checked rule semantics: KILL suppresses, GEN injects, and an inter
/// edge propagates every fact. The expected edges are spelled as dense indices
/// so both arms are held to the same answer rather than to their own arithmetic.
fn flow_rule_edges() -> Vec<ExplodedIfdsCase> {
    let mut cases = Vec::new();

    // (0, 0, 1) is killed, so no edge to (0, 1, 1).
    let mut kill = base_case("KILL suppresses fact propagation", 1, 2, 2);
    kill.intra_edges = vec![(0, 0, 1)];
    kill.flow_kill = vec![(0, 0, 1)];
    kill.forbidden_edges = vec![(kill.dense_index(0, 0, 1), kill.dense_index(0, 1, 1))];
    cases.push(kill);

    // GEN at (0, 0, 1) injects the fact along the intra edge from the 0-fact.
    let mut gen = base_case("GEN injects a new fact", 1, 2, 2);
    gen.intra_edges = vec![(0, 0, 1)];
    gen.flow_gen = vec![(0, 0, 1)];
    gen.required_edges = vec![(gen.dense_index(0, 0, 0), gen.dense_index(0, 1, 1))];
    cases.push(gen);

    // An inter edge is the IFDS upper bound: every fact crosses it.
    let mut inter = base_case("inter edge propagates every fact", 2, 2, 2);
    inter.inter_edges = vec![(0, 0, 1, 1)];
    inter.required_edges = vec![
        (inter.dense_index(0, 0, 0), inter.dense_index(1, 1, 0)),
        (inter.dense_index(0, 0, 1), inter.dense_index(1, 1, 1)),
    ];
    cases.push(inter);

    // Two procedures, one intra edge each, no flow rules: the shape assertion
    // every arm makes about row counts.
    let mut plain = base_case("two procedures, intra edges only", 2, 2, 2);
    plain.intra_edges = vec![(0, 0, 1), (1, 0, 1)];
    cases.push(plain);

    // All four rule kinds at once, the closure bar between the two layers.
    let mut every_rule = base_case("all four rule kinds", 2, 2, 2);
    every_rule.intra_edges = vec![(0, 0, 1), (1, 0, 1)];
    every_rule.inter_edges = vec![(0, 1, 1, 0)];
    every_rule.flow_gen = vec![(0, 0, 1)];
    every_rule.flow_kill = vec![(1, 0, 0)];
    cases.push(every_rule);

    // Four facts with GEN and KILL on different facts of one block.
    let mut wide_facts = base_case("GEN and KILL on distinct facts", 1, 2, 4);
    wide_facts.intra_edges = vec![(0, 0, 1)];
    wide_facts.flow_gen = vec![(0, 0, 2)];
    wide_facts.flow_kill = vec![(0, 0, 3)];
    wide_facts.required_edges = vec![(
        wide_facts.dense_index(0, 0, 0),
        wide_facts.dense_index(0, 1, 2),
    )];
    wide_facts.forbidden_edges = vec![(
        wide_facts.dense_index(0, 0, 3),
        wide_facts.dense_index(0, 1, 3),
    )];
    cases.push(wide_facts);

    // Single node, no rules: the smallest valid domain.
    cases.push(base_case("single node, no rules", 1, 1, 1));

    cases
}

fn base_case(
    label: &str,
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
) -> ExplodedIfdsCase {
    ExplodedIfdsCase {
        label: label.to_string(),
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges: Vec::new(),
        inter_edges: Vec::new(),
        flow_gen: Vec::new(),
        flow_kill: Vec::new(),
        required_edges: Vec::new(),
        forbidden_edges: Vec::new(),
    }
}

/// This crate's ledger over the declared exploded-IFDS groups.
///
/// The declared set is read from [`declared_groups`] on each call, so it is
/// whatever the table says on this run.
#[must_use]
pub fn arm_coverage() -> ArmCoverage {
    ArmCoverage::new(
        "exploded-IFDS",
        "vyre_test_support::exploded_ifds_cases",
        declared_groups().iter().map(|group| group.name).collect(),
        MIN_DECLARED_GROUPS,
        MIN_ASSERTED_CASES,
    )
}
