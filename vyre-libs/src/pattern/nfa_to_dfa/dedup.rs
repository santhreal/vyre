//! Content-addressed deduplication of compiled dense DFAs.

use std::collections::HashMap;

use crate::hash::fnv1a::{fnv1a64_initial_state, fnv1a64_update_byte};
use crate::pattern::dfa_compile::CompiledDfa;

/// Stable content fingerprint for a compiled dense DFA.
///
/// The fingerprint intentionally covers only wire-relevant DFA fields, not
/// allocation identity or capacity. Scan pipelines can use it as a
/// content-addressed dedup key: identical automata compiled in different
/// places hash to the same value, while changes to transitions, accept
/// metadata, output records, state count, or maximum match length perturb the
/// key.
#[must_use]
pub fn dfa_fingerprint(dfa: &CompiledDfa) -> u64 {
    let mut hash = fnv1a64_initial_state();
    hash_u32(&mut hash, dfa.state_count);
    hash_u32(&mut hash, dfa.max_pattern_len);
    hash_u32_slice(&mut hash, &dfa.transitions);
    hash_u32_slice(&mut hash, &dfa.accept);
    hash_u32_slice(&mut hash, &dfa.output_offsets);
    hash_u32_slice(&mut hash, &dfa.output_records);
    hash
}

/// Wire-relevant byte size of a compiled dense DFA.
#[must_use]
pub fn dfa_wire_bytes(dfa: &CompiledDfa) -> usize {
    std::mem::size_of::<u32>()
        * (2 + dfa.transitions.len()
            + dfa.accept.len()
            + dfa.output_offsets.len()
            + dfa.output_records.len())
}

/// Result of inserting a DFA into a content-addressed dedup table.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DfaDedupResult {
    /// Stable content fingerprint for the DFA.
    pub fingerprint: u64,
    /// Canonical slot in the dedup table.
    pub canonical_index: usize,
    /// True when the DFA was not already present.
    pub inserted: bool,
}

/// Summary for a batch DFA canonicalization pass.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct DfaDedupStats {
    /// Number of input DFA plans submitted in this batch.
    pub input_count: usize,
    /// Number of DFA plans inserted as new canonical entries.
    pub inserted_count: usize,
    /// Number of input DFA plans resolved to existing canonical entries.
    pub duplicate_count: usize,
    /// Total number of canonical DFA plans retained after the batch.
    pub table_len_after: usize,
    /// Total wire-relevant bytes submitted in this batch.
    pub input_wire_bytes: usize,
    /// Wire-relevant bytes inserted as new canonical DFA plans in this batch.
    pub inserted_wire_bytes: usize,
    /// Wire-relevant bytes saved by resolving duplicates to canonical plans.
    pub saved_wire_bytes: usize,
}

/// Result of batch canonicalizing DFA plans.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct DfaDedupBatch {
    /// One canonicalization result per submitted DFA, in input order.
    pub results: Vec<DfaDedupResult>,
    /// Aggregate batch statistics.
    pub stats: DfaDedupStats,
}

impl DfaDedupBatch {
    /// Saved wire bytes in parts-per-million of submitted wire bytes.
    ///
    /// PPM keeps this metric deterministic and allocation-free, avoiding float
    /// drift across platforms while still giving planners a compact reuse
    /// efficiency signal.
    #[must_use]
    pub fn saved_wire_ppm(&self) -> u32 {
        saved_wire_ppm(self.stats.saved_wire_bytes, self.stats.input_wire_bytes)
    }
}

/// Collision-safe content-addressed table for compiled dense DFAs.
///
/// The fingerprint is a fast stable key, not a uniqueness proof. Buckets keep
/// every canonical DFA sharing the same fingerprint and compare full DFA
/// content before deduplicating, so a hash collision cannot alias two distinct
/// automata.
#[derive(Debug, Default, Clone)]
pub struct DfaDedupTable {
    buckets: HashMap<u64, Vec<usize>>,
    entries: Vec<CompiledDfa>,
}

impl DfaDedupTable {
    /// Number of unique DFA plans retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the table holds no DFA plans.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Read a canonical DFA by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&CompiledDfa> {
        self.entries.get(index)
    }

    /// Wire-relevant bytes retained by the canonical DFA table.
    #[must_use]
    pub fn canonical_wire_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(dfa_wire_bytes)
            .fold(0usize, usize::saturating_add)
    }

    /// Insert `dfa`, returning its stable canonical slot.
    pub fn insert(&mut self, dfa: CompiledDfa) -> DfaDedupResult {
        let fingerprint = dfa_fingerprint(&dfa);
        if let Some(bucket) = self.buckets.get(&fingerprint) {
            for &candidate in bucket {
                if self
                    .entries
                    .get(candidate)
                    .map(|existing| dfa_content_eq(existing, &dfa))
                    .unwrap_or(false)
                {
                    return DfaDedupResult {
                        fingerprint,
                        canonical_index: candidate,
                        inserted: false,
                    };
                }
            }
        }

        let canonical_index = self.entries.len();
        self.entries.push(dfa);
        self.buckets
            .entry(fingerprint)
            .or_default()
            .push(canonical_index);
        DfaDedupResult {
            fingerprint,
            canonical_index,
            inserted: true,
        }
    }

    /// Insert many DFA plans and return stable canonical indices in input order.
    ///
    /// This is the high-throughput path for scan compilers that emit many
    /// automata in one planning wave. It avoids each caller re-implementing
    /// duplicate accounting and makes dedup evidence explicit.
    pub fn insert_many<I>(&mut self, dfas: I) -> DfaDedupBatch
    where
        I: IntoIterator<Item = CompiledDfa>,
    {
        let mut results = Vec::new();
        let mut inserted_count = 0usize;
        let mut duplicate_count = 0usize;
        let mut input_wire_bytes = 0usize;
        let mut inserted_wire_bytes = 0usize;
        let mut saved_wire_bytes = 0usize;
        for dfa in dfas {
            let wire_bytes = dfa_wire_bytes(&dfa);
            input_wire_bytes = input_wire_bytes.saturating_add(wire_bytes);
            let result = self.insert(dfa);
            if result.inserted {
                inserted_count += 1;
                inserted_wire_bytes = inserted_wire_bytes.saturating_add(wire_bytes);
            } else {
                duplicate_count += 1;
                saved_wire_bytes = saved_wire_bytes.saturating_add(wire_bytes);
            }
            results.push(result);
        }
        DfaDedupBatch {
            stats: DfaDedupStats {
                input_count: results.len(),
                inserted_count,
                duplicate_count,
                table_len_after: self.len(),
                input_wire_bytes,
                inserted_wire_bytes,
                saved_wire_bytes,
            },
            results,
        }
    }

    /// Merge another canonical DFA table into this one.
    ///
    /// This is the cross-shard path: independent scan planners can build local
    /// canonical tables, then merge them into a global content-addressed table
    /// without recompiling automata. Returned results map each source-table
    /// canonical DFA, in source order, to this table's canonical slot.
    pub fn merge_from(&mut self, other: &DfaDedupTable) -> DfaDedupBatch {
        self.insert_many(other.entries.iter().cloned())
    }
}

fn saved_wire_ppm(saved_wire_bytes: usize, input_wire_bytes: usize) -> u32 {
    if input_wire_bytes == 0 {
        return 0;
    }
    let ppm = (saved_wire_bytes as u128).saturating_mul(1_000_000) / (input_wire_bytes as u128);
    u32::try_from(ppm).unwrap_or(u32::MAX)
}

fn dfa_content_eq(left: &CompiledDfa, right: &CompiledDfa) -> bool {
    left.state_count == right.state_count
        && left.max_pattern_len == right.max_pattern_len
        && left.transitions == right.transitions
        && left.accept == right.accept
        && left.output_offsets == right.output_offsets
        && left.output_records == right.output_records
}

fn hash_u32_slice(hash: &mut u64, values: &[u32]) {
    hash_u64(hash, values.len() as u64);
    for &value in values {
        hash_u32(hash, value);
    }
}

fn hash_u32(hash: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *hash = fnv1a64_update_byte(*hash, byte);
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash = fnv1a64_update_byte(*hash, byte);
    }
}
