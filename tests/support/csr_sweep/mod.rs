//! The one generated CSR shape stream every graph sweep matrix draws from.
//!
//! A sweep over "a bounded random CSR with per-edge kind bits and a seeded
//! frontier" was written five times: twice in `vyre-primitives`' shared sweep
//! support, a third time inline in one of its own matrices, a fourth without a
//! frontier for the motif matrix, and once more in `vyre-libs` on a different
//! random generator. The copies did not agree about anything that matters to
//! coverage. One seeded a single frontier bit and allowed every edge kind, so
//! neither the multi-source contention path nor the kind-intersection filter
//! ever fired in the families that used it. One drew its allow mask from a
//! table including zero. Only one set the padding bits above `node_count` in
//! the last frontier word, which is the input a word-granular kernel gets wrong
//! and a node-granular oracle does not.
//!
//! This module owns the stream, and a group is data rather than a function:
//! [`CsrSweepGroup`] states the degree bound, how the frontier is seeded, how
//! the edge filter is chosen and whether edge kinds carry noise, and
//! [`generate`] is the only place that reads a random number. Adding a hostile
//! shape is a row here, and [`declared_groups`] is what a crate's coverage gate
//! reads at run time, so a new row turns red in every crate that no longer
//! sweeps every declared group.

#![allow(dead_code)]

#[path = "../sweep_rng.rs"]
pub(crate) mod rng;

pub(crate) use rng::Rng;

use vyre_libs::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};

/// How a group seeds its frontier.
///
/// The three settings are the coverage difference between the copies this
/// module replaces, not a taxonomy: a single bit walks one source lane, half the
/// nodes make many lanes contend for one output word, and a quarter plus tail
/// padding adds bits that name no node at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrontierSeeding {
    /// No frontier at all, for families that supply their own or match a motif.
    None,
    /// One bit, at a random node.
    SingleNode,
    /// Roughly half the nodes, drawn independently.
    HalfTheNodes,
    /// Roughly a quarter of the nodes, plus every padding bit above
    /// `node_count` in the last word when the count is not word-aligned.
    QuarterWithPaddedTail,
}

/// How a group chooses the edge-kind allow mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EdgeFilter {
    /// Every kind traversable. Leaves the intersection test unexercised.
    AllKinds,
    /// Two randomly chosen kind bits, so the per-edge intersection fires both
    /// ways. Never empty, which would make every case vacuous.
    TwoKinds,
    /// Drawn from a fixed table that includes the empty mask, so a group also
    /// sweeps the case where no edge is traversable.
    Table,
}

/// One row of the sweep: a named shape family with its generator parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CsrSweepGroup {
    /// Group name. This is what a coverage gate matches a sweep against.
    pub(crate) name: &'static str,
    /// Exclusive upper bound on per-node out-degree.
    pub(crate) max_degree: u32,
    /// How the seed frontier is populated.
    pub(crate) frontier: FrontierSeeding,
    /// How the allow mask is chosen.
    pub(crate) filter: EdgeFilter,
    /// Give some edges a second kind bit, so a mask is not always a power of
    /// two and an implementation cannot pass by comparing for equality.
    pub(crate) kind_noise: bool,
}

/// One generated CSR graph with its seed frontier and edge filter.
#[derive(Clone, Debug)]
pub(crate) struct CsrSweepCase {
    /// Node count, in `1..=96`.
    pub(crate) node_count: u32,
    /// Row starts, `node_count + 1` entries.
    pub(crate) offsets: Vec<u32>,
    /// Destination node of each edge.
    pub(crate) targets: Vec<u32>,
    /// Edge-kind bitmask of each edge.
    pub(crate) masks: Vec<u32>,
    /// Seed frontier, [`CsrSweepCase::words`] long, empty for
    /// [`FrontierSeeding::None`].
    pub(crate) frontier: Vec<u32>,
    /// Traversable edge kinds.
    pub(crate) allow_mask: u32,
}

/// One case as the positional tuple a sweep loop binds: node count, the three
/// CSR arrays, the frontier and the allow mask.
///
/// A sweep binds all six at once. Returning them as a tuple keeps that binding
/// one line per call site: a named-field destructuring of six fields is itself
/// eight lines, and repeating it at every site would trade one duplicated
/// generator for one duplicated pattern match.
pub(crate) type CsrSweepParts = (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32);

impl CsrSweepCase {
    /// Consume this case into [`CsrSweepParts`].
    pub(crate) fn into_parts(self) -> CsrSweepParts {
        (
            self.node_count,
            self.offsets,
            self.targets,
            self.masks,
            self.frontier,
            self.allow_mask,
        )
    }

    /// Frontier words this case's node count needs.
    pub(crate) fn words(&self) -> usize {
        self.node_count.div_ceil(32) as usize
    }

    /// Edges in this case.
    pub(crate) fn edge_count(&self) -> usize {
        self.targets.len()
    }

    /// Borrow the CSR arrays as the view every graph entry point reads.
    pub(crate) fn view(&self) -> CsrGraphView<'_> {
        CsrGraphView {
            node_count: self.node_count,
            edge_offsets: &self.offsets,
            edge_targets: &self.targets,
            edge_kind_mask: &self.masks,
        }
    }

    /// Closure inputs for this case, bounded by `max_iters`.
    pub(crate) fn inputs(&self, max_iters: u32) -> CsrClosureInputs<'_> {
        CsrClosureInputs {
            graph: self.view(),
            allow_mask: self.allow_mask,
            max_iters,
        }
    }
}

/// Allow masks [`EdgeFilter::Table`] draws from. The empty mask is deliberate:
/// a closure that ignores it saturates instead of standing still.
const FILTER_TABLE: [u32; 5] = [0, 1, 0b10, 0b101, u32::MAX];

/// Edge kinds a generated mask can name. Five is what every copy used, and it
/// keeps a two-bit mask well inside one word.
const KIND_COUNT: u32 = 5;

/// Largest node count a case can have. Bounded so a sweep of thousands of cases
/// stays a unit-test cost, and above one word so a padded tail exists.
const MAX_NODES: u32 = 96;

/// The declared sweep groups.
///
/// A coverage gate reads this at run time instead of listing names, so adding a
/// row here fails every crate that does not sweep it.
pub(crate) fn declared_groups() -> &'static [CsrSweepGroup] {
    &[
        CsrSweepGroup {
            name: "single_source_all_kinds",
            max_degree: 6,
            frontier: FrontierSeeding::SingleNode,
            filter: EdgeFilter::AllKinds,
            kind_noise: false,
        },
        CsrSweepGroup {
            name: "multi_source_restricted_kinds",
            max_degree: 7,
            frontier: FrontierSeeding::HalfTheNodes,
            filter: EdgeFilter::TwoKinds,
            kind_noise: false,
        },
        CsrSweepGroup {
            name: "padded_tail_masked_kinds",
            max_degree: 5,
            frontier: FrontierSeeding::QuarterWithPaddedTail,
            filter: EdgeFilter::Table,
            kind_noise: true,
        },
        CsrSweepGroup {
            name: "topology_only_all_kinds",
            max_degree: 6,
            frontier: FrontierSeeding::None,
            filter: EdgeFilter::AllKinds,
            kind_noise: false,
        },
    ]
}

/// The group named `name`.
///
/// # Panics
/// Panics when no group has that name, which means a sweep was written against
/// a row that no longer exists.
pub(crate) fn group(name: &str) -> &'static CsrSweepGroup {
    declared_groups()
        .iter()
        .find(|group| group.name == name)
        .unwrap_or_else(|| {
            panic!(
                "Fix: no CSR sweep group is named {name}; declared groups are {:?}.",
                group_names()
            )
        })
}

/// Every declared group name, in declaration order.
pub(crate) fn group_names() -> Vec<&'static str> {
    declared_groups().iter().map(|group| group.name).collect()
}

/// Generate one case of `group` from `seed`.
///
/// The whole case is a function of `seed`, so a failure is reproducible from the
/// seed in its message alone.
pub(crate) fn generate(group: &CsrSweepGroup, seed: u64) -> CsrSweepCase {
    let mut rng = Rng::new(seed | 1);
    let node_count = 1 + rng.range(MAX_NODES);
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    offsets.push(0u32);
    for _ in 0..node_count {
        let degree = rng.range(group.max_degree);
        for _ in 0..degree {
            targets.push(rng.range(node_count));
            let kind = 1u32 << rng.range(KIND_COUNT);
            let noise = if group.kind_noise && rng.next_u32() & 7 == 0 {
                1u32 << rng.range(KIND_COUNT)
            } else {
                0
            };
            masks.push(kind | noise);
        }
        offsets.push(targets.len() as u32);
    }

    let words = node_count.div_ceil(32) as usize;
    let mut frontier = match group.frontier {
        FrontierSeeding::None => Vec::new(),
        _ => vec![0u32; words],
    };
    match group.frontier {
        FrontierSeeding::None => {}
        FrontierSeeding::SingleNode => {
            let start = rng.range(node_count);
            frontier[(start / 32) as usize] |= 1u32 << (start % 32);
        }
        FrontierSeeding::HalfTheNodes => {
            for node in 0..node_count {
                if rng.next_u32() & 1 == 0 {
                    frontier[(node / 32) as usize] |= 1u32 << (node % 32);
                }
            }
        }
        FrontierSeeding::QuarterWithPaddedTail => {
            for node in 0..node_count {
                if rng.next_u32() & 3 == 0 {
                    frontier[(node / 32) as usize] |= 1u32 << (node % 32);
                }
            }
            let used = node_count % 32;
            if used != 0 {
                frontier[words - 1] |= !((1u32 << used) - 1);
            }
        }
    }

    let allow_mask = match group.filter {
        EdgeFilter::AllKinds => u32::MAX,
        EdgeFilter::TwoKinds => 1u32 << rng.range(KIND_COUNT) | 1u32 << rng.range(KIND_COUNT),
        EdgeFilter::Table => rng.pick(&FILTER_TABLE),
    };

    CsrSweepCase {
        node_count,
        offsets,
        targets,
        masks,
        frontier,
        allow_mask,
    }
}

/// `count` cases of the group named `name`, paired with the case index a failure
/// reports.
///
/// `seed` keeps one family's shapes distinct from every other family's and
/// `stride` decorrelates successive cases within it. Both are coverage: change
/// either and the family moves onto different graphs.
pub(crate) fn cases(
    name: &str,
    count: u64,
    seed: u64,
    stride: u64,
) -> impl Iterator<Item = (u64, CsrSweepCase)> {
    let group = group(name);
    (0..count).map(move |case| (case, generate(group, seed ^ case.wrapping_mul(stride))))
}

/// Fail unless every declared group is swept somewhere in `package`'s tests.
///
/// The declared set comes from [`declared_groups`] and the swept set is read out
/// of the crate's own test sources at run time, so neither side is a list this
/// function carries. A group added above therefore turns red in every crate
/// whose matrices do not draw from it, which is the hole five copies of this
/// stream had: a hostile shape existed in one copy and no other, and nothing
/// failed.
///
/// # Panics
/// Panics naming each declared group `package` never draws, and when the crate's
/// test directory holds no sweep at all.
pub(crate) fn assert_every_group_is_swept(package: &str) {
    let tests = vyre_test_support::monorepo::vyre_crate_directory(package).join("tests");
    let mut sources = Vec::new();
    vyre_test_support::collect_rust_files(&tests, &mut sources);
    assert!(
        !sources.is_empty(),
        "Fix: {package} has no test sources under {}; the sweep coverage gate cannot read what it \
         proves.",
        tests.display()
    );

    let text = sources
        .iter()
        .map(|path| std::fs::read_to_string(path).expect("Fix: read a test source"))
        .collect::<String>();

    let unswept: Vec<&str> = group_names()
        .into_iter()
        .filter(|name| !text.contains(&format!("\"{name}\"")))
        .collect();
    assert!(
        unswept.is_empty(),
        "Fix: {package} declares no sweep for CSR shape group(s) {unswept:?}. Either draw them \
         with csr_sweep::cases in one of its matrices, or delete the row from \
         tests/support/csr_sweep/mod.rs so no crate claims coverage nobody has."
    );
}

/// `count` cases of the group named `name` as flat tuples, paired with the case
/// index a failure reports.
pub(crate) fn tuples(
    name: &str,
    count: u64,
    seed: u64,
    stride: u64,
) -> impl Iterator<Item = (u64, u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32)> {
    cases(name, count, seed, stride).map(|(case, shape)| {
        let (node_count, offsets, targets, masks, frontier, allow_mask) = shape.into_parts();
        (
            case, node_count, offsets, targets, masks, frontier, allow_mask,
        )
    })
}

/// One masked forward step, computed without touching any production code path.
///
/// Every sweep family that measures a closure needs this, and two crates had
/// byte-identical copies of it, which makes the "independent oracle" claim in
/// their headers false: a mistake in the shared reasoning appeared in both.
/// Owning it once keeps it independent of the implementation, which is the
/// property that matters, and stops the two copies from drifting apart.
///
/// A set bit at `src` propagates to every `edge_targets[e]` whose
/// `edge_kind_mask[e]` intersects `allow_mask`. Bits at or above `node_count`
/// are ignored, so a padded frontier tail cannot manufacture a source.
pub(crate) fn oracle_forward_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let mut out = vec![0u32; node_count.div_ceil(32) as usize];
    for src in 0..node_count {
        let word = (src / 32) as usize;
        if word >= frontier_in.len() || frontier_in[word] & (1u32 << (src % 32)) == 0 {
            continue;
        }
        for edge in edge_offsets[src as usize] as usize..edge_offsets[src as usize + 1] as usize {
            if edge_kind_mask[edge] & allow_mask == 0 {
                continue;
            }
            let dst = edge_targets[edge];
            if dst < node_count {
                out[(dst / 32) as usize] |= 1u32 << (dst % 32);
            }
        }
    }
    out
}

/// The masked forward closure: `oracle_forward_step` to fixpoint, bounded by
/// `max_iters`, with the changed flag both persistent-BFS kernels also return.
///
/// Two crates carried this loop verbatim, which made the primitive arm and the
/// substrate arm share one reference instead of checking each other.
pub(crate) fn oracle_persistent_closure(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, u32) {
    let words = node_count.div_ceil(32) as usize;
    let mut out = frontier_in.to_vec();
    out.resize(words, 0);
    let mut changed = 0;
    for _ in 0..max_iters {
        let step = oracle_forward_step(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            &out,
            allow_mask,
        );
        let mut step_changed = false;
        for word in 0..words {
            let before = out[word];
            out[word] |= step[word];
            if out[word] != before {
                step_changed = true;
            }
        }
        if step_changed {
            changed = 1;
        } else {
            break;
        }
    }
    (out, changed)
}
