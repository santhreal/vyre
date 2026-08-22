//! `hashtable.openaddr.probe.10m`  -  Open-addressing hash table probe.
//!
//! Probes a prebuilt 10M-key table with 1M random lookups. GPU uses
//! open-addressing with linear probing on a power-of-2 table. CPU baseline uses
//! a prebuilt hashbrown table (robin-hood hashing, SIMD probing).
//!
//! This is CPU-favorable territory: hash tables are latency-bound with
//! pointer-chasing patterns that exploit CPU caches. The GPU must overcome
//! random-access memory latency via massive parallelism.

use crate::api::case::{BenchCase, BenchContext, BenchError};
use crate::cases::harness::{
    verify_exact, CaseOps, ContractDescription, HarnessCase, WorkloadDescription,
};
use crate::cases::reference_sample::{
    measure_against_reference, referenced_program, HostReferencePayload, HostReferenced,
};
use hashbrown::HashMap;
use rand::{RngExt, SeedableRng};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const KEY_COUNT: u32 = 10_000_000;
const PROBE_COUNT: u32 = 1_000_000;
const TABLE_SIZE: u32 = 16_777_216; // 2^24, load factor ~0.6

struct HashtableProbePrepared {
    dispatch: HostReferencePayload,
    probe_keys: Vec<u32>,
    cpu_table: HashMap<u32, u32>,
}

impl HostReferenced for HashtableProbePrepared {
    fn dispatch(&self) -> &HostReferencePayload {
        &self.dispatch
    }

    fn reference(&self) -> Result<Vec<u8>, BenchError> {
        Ok(self
            .probe_keys
            .iter()
            .flat_map(|key| self.cpu_table.get(key).copied().unwrap_or(0).to_le_bytes())
            .collect::<Vec<u8>>())
    }
}

static WORKLOAD: WorkloadDescription = WorkloadDescription::honest(
    "hashtable.openaddr.probe.10m",
    "Hashtable Probe 10M",
    "Open-addressing hash table: probe 1M random lookups against a prebuilt 10M-key table",
    &["honest", "latency-bound", "random-access"],
    TABLE_SIZE as u64 * 8 + PROBE_COUNT as u64 * 4,
    Some(ContractDescription {
        primitive: "Hash table probe",
        baseline_crate: "hashbrown",
        baseline_name: "hashbrown 0.17.0 prebuilt SwissTable probe",
        min_speedup_x: 10.0,
    }),
);

static OPS: CaseOps<HashtableProbePrepared> = CaseOps {
    build: prepare_hashtable_probe,
    measure: measure_against_reference::<HashtableProbePrepared>,
    verify: verify_exact,
    program: referenced_program::<HashtableProbePrepared>,
    fingerprint: None,
    bytes_touched,
};

static CASE: HarnessCase<HashtableProbePrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

/// Reads are the table's key and value slots plus the probe keys; writes are one
/// result per probe.
fn bytes_touched(_prepared: &HashtableProbePrepared) -> (u64, u64) {
    (
        TABLE_SIZE as u64 * 8 + PROBE_COUNT as u64 * 4,
        PROBE_COUNT as u64 * 4,
    )
}

/// Linear-probed open-addressing lookup: one thread per probe key.
///
/// Bindings are the table's key slots, its value slots, the probe keys, and one
/// result word per probe.
fn hashtable_probe_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("table_keys", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(TABLE_SIZE),
            BufferDecl::storage("table_vals", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(TABLE_SIZE),
            BufferDecl::storage("probe_keys", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(PROBE_COUNT),
            BufferDecl::output("results", 3, DataType::U32).with_count(PROBE_COUNT),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(PROBE_COUNT)),
                vec![
                    Node::let_bind("key", Expr::load("probe_keys", Expr::var("tid"))),
                    // Knuth multiplicative hash, masked for the power-of-2 table.
                    Node::let_bind(
                        "hash",
                        Expr::bitand(
                            Expr::mul(Expr::var("key"), Expr::u32(2_654_435_761)),
                            Expr::u32(TABLE_SIZE - 1),
                        ),
                    ),
                    Node::let_bind("result", Expr::u32(0)),
                    Node::Loop {
                        var: "probe".into(),
                        from: Expr::u32(0),
                        to: Expr::u32(64),
                        body: vec![
                            Node::let_bind(
                                "slot",
                                Expr::bitand(
                                    Expr::add(Expr::var("hash"), Expr::var("probe")),
                                    Expr::u32(TABLE_SIZE - 1),
                                ),
                            ),
                            Node::let_bind("slot_key", Expr::load("table_keys", Expr::var("slot"))),
                            Node::if_then(
                                Expr::eq(Expr::var("slot_key"), Expr::var("key")),
                                vec![Node::assign(
                                    "result",
                                    Expr::load("table_vals", Expr::var("slot")),
                                )],
                            ),
                        ],
                    },
                    Node::store("results", Expr::var("tid"), Expr::var("result")),
                ],
            ),
        ],
    )
}

fn prepare_hashtable_probe(ctx: &mut BenchContext) -> Result<HashtableProbePrepared, BenchError> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEAD_BEEF);

    let mut table_keys = vec![0u32; TABLE_SIZE as usize];
    let mut table_vals = vec![0u32; TABLE_SIZE as usize];
    let mask = TABLE_SIZE - 1;
    let mut cpu_table: HashMap<u32, u32> = HashMap::with_capacity(KEY_COUNT as usize);

    let mut inserted_keys = Vec::with_capacity(KEY_COUNT as usize);
    for i in 0..KEY_COUNT {
        let key = rng.random_range(1..u32::MAX); // 0 = empty sentinel
        let val = i + 1;
        let mut slot = key.wrapping_mul(2_654_435_761) & mask;
        for _ in 0..64 {
            if table_keys[slot as usize] == 0 {
                table_keys[slot as usize] = key;
                table_vals[slot as usize] = val;
                inserted_keys.push(key);
                cpu_table.insert(key, val);
                break;
            }
            slot = (slot + 1) & mask;
        }
    }

    let mut probe_keys = vec![0u32; PROBE_COUNT as usize];
    for probe_key in &mut probe_keys {
        if rng.random_bool(0.8) && !inserted_keys.is_empty() {
            *probe_key = inserted_keys[rng.random_range(0..inserted_keys.len())];
        } else {
            *probe_key = rng.random_range(1..u32::MAX);
        }
    }

    let inputs = vec![
        vyre_primitives::wire::pack_u32_slice(&table_keys),
        vyre_primitives::wire::pack_u32_slice(&table_vals),
        vyre_primitives::wire::pack_u32_slice(&probe_keys),
    ];

    Ok(HashtableProbePrepared {
        dispatch: HostReferencePayload::program_ordered_resident(
            ctx,
            hashtable_probe_program(),
            inputs,
            "hashtable probe bench",
        )?,
        probe_keys,
        cpu_table,
    })
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
