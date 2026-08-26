//! Contract test for adapter-backed cache-invalidation behavior.
//!
//! The `libs-compositions` feature routes cache invalidation through semantic
//! execution. Production cache invalidation has no hidden CPU fallback.

use vyre_driver::cache_invalidation::{impacted_entries_into, CacheInvalidationScratch};
use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_megakernel::{
    CompileObjective, DeviceFacts, Digest, ExternalFacts, SearchBudget, SemanticExecutionError,
    SemanticExecutionOutput, SemanticExecutionPolicy, SemanticExecutionRequest, SemanticExecutor,
};

fn policy() -> SemanticExecutionPolicy {
    SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([0; 32]), std::collections::BTreeMap::new()),
        DeviceFacts::unknown(),
        CompileObjective::MinimizeLatency,
        SearchBudget::new(8, 64, 0, 0, 1_000),
        1_000_000,
    )
}

#[test]
fn default_path_marks_impacted_lineage_entries() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut out = vec![99u32; 5];
    let mut scratch = CacheInvalidationScratch::default();
    let mut rule_adj = vec![0u32; 9];
    rule_adj[0 * 3 + 1] = 1;
    let mut state = vec![0u32; 9];
    state[2 * 3 + 1] = 1;

    impacted_entries_into(
        &dispatcher,
        &policy(),
        &[1, 0, 1],
        &rule_adj,
        &state,
        &[0; 9],
        3,
        10,
        &[0, 1, 2, 99],
        &mut out,
        &mut scratch,
    )
    .expect("reference dispatcher must execute GPU cache invalidation composition");

    assert_eq!(out.len(), 4, "output length must match lineage_cells.len()");
    assert_eq!(
        out,
        vec![1, 1, 1, 0],
        "default cache invalidation must mark direct, transitive, and provenance-linked entries"
    );
}

#[test]
fn default_path_handles_empty_lineage_cells() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut out = vec![99u32; 3];
    let mut scratch = CacheInvalidationScratch::default();

    impacted_entries_into(
        &dispatcher,
        &policy(),
        &[],
        &[],
        &[],
        &[],
        0,
        0,
        &[],
        &mut out,
        &mut scratch,
    )
    .expect("empty invalidation has no GPU work");

    assert!(
        out.is_empty(),
        "empty lineage_cells must produce empty output"
    );
}

#[test]
fn default_path_handles_max_u32_n_without_panic() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut out = vec![99u32; 2];
    let mut scratch = CacheInvalidationScratch::default();

    // n = u32::MAX with tiny arrays: the default path must not attempt
    // to index or allocate based on n.
    let err = impacted_entries_into(
        &dispatcher,
        &policy(),
        &[1],
        &[0],
        &[0],
        &[0],
        u32::MAX,
        u32::MAX,
        &[0; 2],
        &mut out,
        &mut scratch,
    )
    .expect_err("oversized n must fail loudly before indexing or allocating");

    assert!(
        err.to_string().contains("Fix:"),
        "oversized n error must be actionable"
    );
}

#[test]
fn default_path_reuses_scratch_without_growing() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut out = vec![99u32; 3];
    let mut scratch = CacheInvalidationScratch::default();

    impacted_entries_into(
        &dispatcher,
        &policy(),
        &[1],
        &[0],
        &[0],
        &[0],
        1,
        1,
        &[0; 3],
        &mut out,
        &mut scratch,
    )
    .expect("reference dispatcher must execute GPU cache invalidation composition");
    assert_eq!(out, vec![1, 1, 1]);

    impacted_entries_into(
        &dispatcher,
        &policy(),
        &[1],
        &[0],
        &[0],
        &[0],
        1,
        1,
        &[0; 5],
        &mut out,
        &mut scratch,
    )
    .expect("reference dispatcher must execute GPU cache invalidation composition");
    assert_eq!(out, vec![1, 1, 1, 1, 1]);
}
#[test]
fn default_path_handles_pure_transitive_provenance_and_unimpacted() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut out = vec![99u32; 5];
    let mut scratch = CacheInvalidationScratch::default();

    // 4 nodes: 0, 1, 2, 3
    // Rule graph: 0 -> 1
    let mut rule_adj = vec![0u32; 16];
    rule_adj[0 * 4 + 1] = 1;
    // Provenance: 2 -> 1
    let mut state = vec![0u32; 16];
    state[2 * 4 + 1] = 1;
    let join_rules = vec![0u32; 16];

    // Only node 0 intervened
    let intervention_mask = vec![1, 0, 0, 0];

    impacted_entries_into(
        &dispatcher,
        &policy(),
        &intervention_mask,
        &rule_adj,
        &state,
        &join_rules,
        4,
        10,
        &[0, 1, 2, 3, 99],
        &mut out,
        &mut scratch,
    )
    .expect("reference dispatcher must execute GPU cache invalidation composition");

    // 0 is direct, 1 is transitive rule impact, 2 is provenance impact, 3 is unimpacted, 99 is invalid lineage
    assert_eq!(out, vec![1, 1, 1, 0, 0]);
}

/// Returns one truncated buffer for every declared graph output.
struct MalformedOutputExecutor;

impl SemanticExecutor for MalformedOutputExecutor {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let outputs = request
            .logical()
            .graph()
            .nodes()
            .iter()
            .flat_map(|node| node.outputs.iter().copied())
            .map(|value| (value, vec![0u8; 1]))
            .collect();
        Ok(SemanticExecutionOutput {
            artifact: Digest([1; 32]),
            payload: Digest([2; 32]),
            outputs,
        })
    }
}

#[test]
fn default_path_fails_closed_on_malformed_backend_output_without_stale_caller_output() {
    let dispatcher = MalformedOutputExecutor;
    let mut out = vec![99u32; 4];
    let mut scratch = CacheInvalidationScratch::default();

    let err = impacted_entries_into(
        &dispatcher,
        &policy(),
        &[1, 0],
        &[0, 1, 0, 0],
        &[0; 4],
        &[0; 4],
        2,
        4,
        &[0, 1],
        &mut out,
        &mut scratch,
    )
    .expect_err("malformed backend output must fail loudly");

    assert!(
        out.is_empty(),
        "malformed backend output must clear stale caller output"
    );
    assert!(
        err.to_string().contains("Fix:"),
        "backend error must include actionable diagnostic"
    );
}
