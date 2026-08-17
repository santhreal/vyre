//! Which queue materializer a resident sparse-queue step selects, and the exact
//! step sequence each one launches before the traverse step.
//!
//! `ResidentCsrQueueMaterializer` has two variants, and the deterministic
//! word-prefix one splits again on whether per-block frontier word offsets are
//! computed inside the scatter pass or by an extra pass before it. That is three
//! queue-materialization step sequences, and this file used to spell one case
//! per sequence with its own copy of the fixture and the assertions. The copies
//! drifted: only the wide atomic case asserted that the atomic path allocates no
//! word-prefix scratch, only the first three asserted which handles were
//! uploaded, only the last two asserted the plan cache, and none of them sat on
//! the block-offset boundary that decides between the two word-prefix
//! sequences.
//!
//! The variants are a declared table now, and one assertion body walks it.

use super::super::*;
use super::recording_dispatcher::{traversal_graph, Frontier, SparseQueueRun};
use crate::graph::csr_frontier_queue::scratch::WORD_PREFIX_INLINE_BLOCK_OFFSET_MAX_BLOCKS;

/// The queue-materialization steps a variant launches before traverse.
///
/// Each arm names the `ResidentCsrQueueMaterializer` variant it belongs to; the
/// gate below matches those names against the enum in the production source.
enum QueueMaterialization {
    /// `AtomicWordScan`: initialize `queue_len` on device, then scan packed
    /// words and append active bits atomically while clearing `frontier_out`.
    AtomicWordScan,
    /// `DeterministicWordPrefix` with few enough blocks to fold the block
    /// offsets into the scatter: clear `frontier_out`, popcount-scan words into
    /// partials and block totals, then scatter into queue order.
    WordPrefixInlineOffsets,
    /// `DeterministicWordPrefix` with too many blocks for that: one extra pass
    /// converts block totals into offsets before the scatter.
    WordPrefixPrecomputedOffsets,
}

impl QueueMaterialization {
    fn materializer(&self) -> &'static str {
        match self {
            QueueMaterialization::AtomicWordScan => "AtomicWordScan",
            QueueMaterialization::WordPrefixInlineOffsets
            | QueueMaterialization::WordPrefixPrecomputedOffsets => "DeterministicWordPrefix",
        }
    }

    /// Only the deterministic word-prefix sequences popcount-scan packed words,
    /// so only they allocate the partials and block-totals buffers.
    fn allocates_word_prefix(&self) -> bool {
        !matches!(self, QueueMaterialization::AtomicWordScan)
    }

    /// The handle sets each queue-materialization step binds, in launch order.
    fn expected_steps(&self, run: &SparseQueueRun) -> Vec<Vec<u64>> {
        let [frontier_in, frontier_out, queue_len] = run.frontier_scratch();
        let queue = run.active_queue();
        match self {
            QueueMaterialization::AtomicWordScan => vec![
                vec![queue_len],
                vec![frontier_in, queue, queue_len, frontier_out],
            ],
            QueueMaterialization::WordPrefixInlineOffsets => {
                let (partials, block_totals) = run.word_prefix();
                vec![
                    vec![frontier_out],
                    vec![frontier_in, partials, block_totals],
                    vec![frontier_in, partials, block_totals, queue, queue_len],
                ]
            }
            QueueMaterialization::WordPrefixPrecomputedOffsets => {
                let (partials, block_totals) = run.word_prefix();
                vec![
                    vec![frontier_out],
                    vec![frontier_in, partials, block_totals],
                    vec![block_totals],
                    vec![frontier_in, partials, block_totals, queue, queue_len],
                ]
            }
        }
    }
}

/// One graph width and frontier fill, and the materialization it must select.
struct MaterializerCase {
    label: &'static str,
    node_count: u32,
    frontier: Frontier,
    materialization: QueueMaterialization,
}

/// Packed frontier words `node_count` nodes occupy.
fn words_for(node_count: u32) -> usize {
    crate::bitset::bitset_words(node_count) as usize
}

/// The widest graph whose frontier still fits inside the inline block-offset
/// budget, and the narrowest one that does not. `frontier_word_prefix_scratch`
/// puts `FRONTIER_WORD_SCAN_BLOCK_LANES` words in a block, so the boundary sits
/// at `WORD_PREFIX_INLINE_BLOCK_OFFSET_MAX_BLOCKS` blocks of packed words, and
/// each of those words covers 32 nodes.
const NODES_PER_WORD: u32 = 32;

fn last_inline_offset_node_count() -> u32 {
    1024 * WORD_PREFIX_INLINE_BLOCK_OFFSET_MAX_BLOCKS * NODES_PER_WORD
}

fn cases() -> Vec<MaterializerCase> {
    let last_inline = last_inline_offset_node_count();
    vec![
        MaterializerCase {
            label: "narrowest graph",
            node_count: 1,
            frontier: Frontier::SingleSource,
            materialization: QueueMaterialization::AtomicWordScan,
        },
        MaterializerCase {
            label: "wide graph, one nonzero frontier word",
            node_count: 8_193,
            frontier: Frontier::SingleSource,
            materialization: QueueMaterialization::AtomicWordScan,
        },
        MaterializerCase {
            label: "narrowest dense frontier that earns the word-prefix scan",
            node_count: 8_193,
            frontier: Frontier::AllWords,
            materialization: QueueMaterialization::WordPrefixInlineOffsets,
        },
        MaterializerCase {
            label: "widest dense frontier that still inlines block offsets",
            node_count: last_inline,
            frontier: Frontier::AllWords,
            materialization: QueueMaterialization::WordPrefixInlineOffsets,
        },
        MaterializerCase {
            label: "narrowest dense frontier that needs a block-offset pass",
            node_count: last_inline + NODES_PER_WORD,
            frontier: Frontier::AllWords,
            materialization: QueueMaterialization::WordPrefixPrecomputedOffsets,
        },
    ]
}

/// Every case runs against a graph with no edges, so the traverse step is the
/// row-serial consumer and appends exactly one step after materialization.
#[test]
fn every_queue_materialization_launches_its_own_step_sequence() {
    for case in cases() {
        let label = case.label;
        let words = words_for(case.node_count);
        let graph = ResidentAdaptiveTraversalGraph {
            node_count: case.node_count,
            edge_count: 0,
            max_row_degree: 0,
            words,
            ..traversal_graph()
        };

        let run = SparseQueueRun::over_graph(&graph, &case.frontier).unwrap_or_else(|error| {
            panic!("Fix: recording dispatcher must complete the {label} sparse-queue step: {error}")
        });

        assert_eq!(
            run.allocated_word_prefix(),
            case.materialization.allocates_word_prefix(),
            "Fix: the {label} sparse-queue step must allocate word-prefix scratch only for the deterministic word-prefix materializer."
        );

        assert_eq!(
            run.uploads(),
            vec![run.frontier_scratch()[0]],
            "Fix: the {label} sparse-queue step must upload only the input frontier; queue length and output are initialized on device."
        );

        let mut expected = case.materialization.expected_steps(&run);
        let materialization_steps = expected.len();
        expected.push(traverse_step(&run, &graph));
        assert_eq!(
            run.steps(),
            expected,
            "Fix: the {label} sparse-queue step must launch {materialization_steps} queue-materialization steps then one traverse step, binding exactly these handles."
        );

        assert_eq!(
            run.plan_cache(),
            AdaptiveTraversalPlanCacheSnapshot {
                entries: expected.len(),
                hits: 0,
                misses: expected.len() as u64,
            },
            "Fix: the {label} sparse-queue step must build every launched Program through the plan cache, once each."
        );

        assert_eq!(
            run.frontier_out,
            vec![0; words],
            "Fix: the {label} sparse-queue step must read back one packed word per frontier word."
        );
    }
}

/// The row-serial queue consumer: the active queue and its length, the three CSR
/// graph buffers, and `frontier_out`.
fn traverse_step(run: &SparseQueueRun, graph: &ResidentAdaptiveTraversalGraph) -> Vec<u64> {
    let [_, frontier_out, queue_len] = run.frontier_scratch();
    vec![
        run.active_queue(),
        queue_len,
        graph.handles[0],
        graph.handles[1],
        graph.handles[2],
        frontier_out,
    ]
}

/// `ResidentCsrQueueMaterializer` variant names, read from the production source
/// at run time so a variant added there without a case above fails here.
fn published_materializer_variants() -> Vec<String> {
    let path = vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-libs/src/graph/csr_frontier_queue/scratch.rs");
    let source = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("Fix: resident CSR queue scratch source must be readable at {path:?}: {error}")
    });

    let start = source
        .find("pub(crate) enum ResidentCsrQueueMaterializer {")
        .unwrap_or_else(|| {
            panic!("Fix: {path:?} must declare `pub(crate) enum ResidentCsrQueueMaterializer`; this gate reads its variants from that declaration.")
        });
    let body = &source[start..];
    let end = body.find("\n}").expect(
        "Fix: the ResidentCsrQueueMaterializer declaration must close with a `}` at column zero.",
    );

    let mut variants = Vec::new();
    for line in body[..end].lines().skip(1) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        variants.push(trimmed.trim_end_matches(',').to_string());
    }

    assert!(
        !variants.is_empty(),
        "Fix: ResidentCsrQueueMaterializer variant scan of {path:?} found no variants; this gate cannot derive its member set."
    );
    variants
}

#[test]
fn every_resident_queue_materializer_variant_has_a_step_sequence_case() {
    let published = published_materializer_variants();
    let covered = cases()
        .iter()
        .map(|case| case.materialization.materializer().to_string())
        .collect::<Vec<_>>();

    for variant in &published {
        assert!(
            covered.iter().any(|name| name == variant),
            "Fix: ResidentCsrQueueMaterializer::{variant} has no case in this file, so nothing pins the step sequence it launches. Add a MaterializerCase whose graph width and frontier select it."
        );
    }

    for name in &covered {
        assert!(
            published.contains(name),
            "Fix: this file has a case for ResidentCsrQueueMaterializer::{name}, which the enum no longer declares. Delete the stale case."
        );
    }
}
