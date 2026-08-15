//! `search.binary.u32.1m`  -  divergent binary search over a sorted table.

use super::byte_pack::u32_bytes;
use crate::api::case::{BenchCase, BenchContext, BenchError};
use crate::cases::harness::{
    verify_exact, CaseOps, ContractDescription, HarnessCase, WorkloadDescription,
};
use crate::cases::reference_sample::{
    measure_against_reference, referenced_program, HostReferencePayload, HostReferenced,
};
use rayon::prelude::*;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const KEY_COUNT: u32 = 1 << 20;
const QUERY_COUNT: u32 = 1 << 20;
const MISS: u32 = u32::MAX;
const SEARCH_STEPS: u32 = 21;

struct BinarySearchPrepared {
    dispatch: HostReferencePayload,
    keys: Vec<u32>,
    queries: Vec<u32>,
}

impl HostReferenced for BinarySearchPrepared {
    fn dispatch(&self) -> &HostReferencePayload {
        &self.dispatch
    }

    fn reference(&self) -> Result<Vec<u8>, BenchError> {
        Ok(cpu_binary_search_results(&self.keys, &self.queries))
    }
}

static WORKLOAD: WorkloadDescription = WorkloadDescription::honest(
    "search.binary.u32.1m",
    "Binary Search U32 1M",
    "Divergent binary search: 1M queries against a sorted 1M-entry u32 table",
    &["honest", "cpu-favorable", "branchy", "cache"],
    (KEY_COUNT as u64 + QUERY_COUNT as u64 * 2) * 4,
    Some(ContractDescription {
        primitive: "Divergent binary search",
        baseline_crate: "std+rayon",
        baseline_name: "Rust slice::binary_search with Rayon parallel query partitioning",
        min_speedup_x: 3.0,
    }),
);

static OPS: CaseOps<BinarySearchPrepared> = CaseOps {
    build: prepare_binary_search,
    measure: measure_against_reference::<BinarySearchPrepared>,
    verify: verify_exact,
    program: referenced_program::<BinarySearchPrepared>,
    fingerprint: None,
    bytes_touched: bytes_touched,
};

static CASE: HarnessCase<BinarySearchPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

/// Reads are the uploaded key and query tables; writes are one result per query.
fn bytes_touched(prepared: &BinarySearchPrepared) -> (u64, u64) {
    (
        prepared.dispatch.input_bytes_total,
        u64::from(QUERY_COUNT) * 4,
    )
}

fn binary_search_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("keys", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(KEY_COUNT),
            BufferDecl::storage("queries", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(QUERY_COUNT),
            BufferDecl::output("results", 2, DataType::U32).with_count(QUERY_COUNT),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(QUERY_COUNT)),
                vec![
                    Node::let_bind("query", Expr::load("queries", Expr::var("tid"))),
                    Node::let_bind("low", Expr::u32(0)),
                    Node::let_bind("high", Expr::u32(KEY_COUNT)),
                    Node::Loop {
                        var: "step".into(),
                        from: Expr::u32(0),
                        to: Expr::u32(SEARCH_STEPS),
                        body: vec![
                            Node::let_bind(
                                "mid",
                                Expr::shr(
                                    Expr::add(Expr::var("low"), Expr::var("high")),
                                    Expr::u32(1),
                                ),
                            ),
                            Node::let_bind("mid_key", Expr::load("keys", Expr::var("mid"))),
                            Node::let_bind(
                                "go_right",
                                Expr::lt(Expr::var("mid_key"), Expr::var("query")),
                            ),
                            Node::let_bind(
                                "next_low",
                                Expr::select(
                                    Expr::var("go_right"),
                                    Expr::add(Expr::var("mid"), Expr::u32(1)),
                                    Expr::var("low"),
                                ),
                            ),
                            Node::let_bind(
                                "next_high",
                                Expr::select(
                                    Expr::var("go_right"),
                                    Expr::var("high"),
                                    Expr::var("mid"),
                                ),
                            ),
                            Node::assign("low", Expr::var("next_low")),
                            Node::assign("high", Expr::var("next_high")),
                        ],
                    },
                    Node::if_then_else(
                        Expr::lt(Expr::var("low"), Expr::u32(KEY_COUNT)),
                        vec![
                            Node::let_bind("candidate", Expr::load("keys", Expr::var("low"))),
                            Node::if_then_else(
                                Expr::eq(Expr::var("candidate"), Expr::var("query")),
                                vec![Node::store("results", Expr::var("tid"), Expr::var("low"))],
                                vec![Node::store("results", Expr::var("tid"), Expr::u32(MISS))],
                            ),
                        ],
                        vec![Node::store("results", Expr::var("tid"), Expr::u32(MISS))],
                    ),
                ],
            ),
        ],
    )
}

fn prepare_binary_search(ctx: &mut BenchContext) -> Result<BinarySearchPrepared, BenchError> {
    let keys: Vec<u32> = (0..KEY_COUNT)
        .map(|value| value.saturating_mul(2))
        .collect();
    let queries = build_queries();
    let inputs = vec![u32_bytes(&keys), u32_bytes(&queries)];

    Ok(BinarySearchPrepared {
        dispatch: HostReferencePayload::program_ordered_resident(
            ctx,
            binary_search_program(),
            inputs,
            "binary search bench",
        )?,
        keys,
        queries,
    })
}

fn build_queries() -> Vec<u32> {
    (0..QUERY_COUNT)
        .map(|idx| {
            let mixed = idx.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            let slot = mixed & (KEY_COUNT - 1);
            if idx % 10 < 7 {
                slot.saturating_mul(2)
            } else {
                slot.saturating_mul(2).saturating_add(1)
            }
        })
        .collect()
}

fn cpu_binary_search_results(keys: &[u32], queries: &[u32]) -> Vec<u8> {
    let results: Vec<u32> = queries
        .par_iter()
        .map(|query| {
            keys.binary_search(query)
                .map(|index| index as u32)
                .unwrap_or(MISS)
        })
        .collect();
    u32_bytes(&results)
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
