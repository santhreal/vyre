//! Single deterministic query engine for compiler data, analysis, and artifact generation.
//!
//! Provides a unified query system owning parsing/validation, semantic facts,
//! equivalence facts, legality, cost inputs, lowering, emission, and artifact queries
//! with declared dependencies, cycle detection, revision tracking, cancellation,
//! memory accounting, and precise invalidation.

use core::fmt;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::arenas::SubstrateArena;
use super::cache::{StaleCacheError, VersionedCacheKey, SUBSTRATE_CACHE_SCHEMA_VERSION};
use super::ids::Revision;
use super::views::CompilerLevelStage;

/// Canonical query identifier key spanning all compiler query families.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QueryKey {
    /// Validation and structural check of a program.
    ParseValidate {
        /// Program content digest.
        program_digest: [u8; 32],
    },
    /// Whole-program graph validation and topology verification.
    ProgramGraphValidate {
        /// Graph content digest.
        graph_digest: [u8; 32],
    },
    /// Semantic facts inference (liveness, use sets, effects).
    SemanticFacts {
        /// Node index within graph.
        node_id: u32,
        /// Node program digest.
        program_digest: [u8; 32],
    },
    /// Type checking and inference.
    TypeCheck {
        /// Program content digest.
        program_digest: [u8; 32],
    },
    /// Shape fact propagation and bounds derivation.
    ShapeFacts {
        /// Program content digest.
        program_digest: [u8; 32],
    },
    /// Algebraic law and rewrite equivalence query.
    EquivalenceFacts {
        /// Semantic operation identifier.
        op_name: String,
    },
    /// Schedule and fusion legality verification.
    Legality {
        /// Graph content digest.
        graph_digest: [u8; 32],
        /// Target device fingerprint.
        target_fingerprint: u64,
    },
    /// Cost model inputs and occupancy metrics.
    CostInput {
        /// Graph content digest.
        graph_digest: [u8; 32],
        /// Target device fingerprint.
        target_fingerprint: u64,
    },
    /// Physical kernel lowering.
    Lowering {
        /// Graph node identifier.
        node_id: u32,
        /// Target device fingerprint.
        target_fingerprint: u64,
    },
    /// Target bytecode or shader text emission.
    Emission {
        /// Graph node identifier.
        node_id: u32,
        /// Format identity supplied by the target materializer.
        target_format: String,
        /// Target device fingerprint.
        target_fingerprint: u64,
    },
    /// Megakernel whole-program artifact compilation.
    Artifact {
        /// Graph content digest.
        graph_digest: [u8; 32],
        /// Target device fingerprint.
        target_fingerprint: u64,
    },
}

impl QueryKey {
    /// Associated compiler level stage for this query family.
    #[must_use]
    pub fn compiler_level(&self) -> CompilerLevelStage {
        match self {
            Self::ParseValidate { .. }
            | Self::ProgramGraphValidate { .. }
            | Self::TypeCheck { .. } => CompilerLevelStage::WholeProgramGraph,
            Self::SemanticFacts { .. }
            | Self::ShapeFacts { .. }
            | Self::EquivalenceFacts { .. } => CompilerLevelStage::LogicalRegion,
            Self::Legality { .. } | Self::CostInput { .. } => CompilerLevelStage::SelectedSchedule,
            Self::Lowering { .. } => CompilerLevelStage::PhysicalKernel,
            Self::Emission { .. } | Self::Artifact { .. } => CompilerLevelStage::TargetPayload,
        }
    }
}

impl fmt::Display for QueryKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseValidate { program_digest } => {
                write!(
                    f,
                    "parse_validate({:02x}{:02x})",
                    program_digest[0], program_digest[1]
                )
            }
            Self::ProgramGraphValidate { graph_digest } => {
                write!(
                    f,
                    "graph_validate({:02x}{:02x})",
                    graph_digest[0], graph_digest[1]
                )
            }
            Self::SemanticFacts { node_id, .. } => write!(f, "semantic_facts(node:{node_id})"),
            Self::TypeCheck { .. } => write!(f, "type_check"),
            Self::ShapeFacts { .. } => write!(f, "shape_facts"),
            Self::EquivalenceFacts { op_name } => write!(f, "equivalence({op_name})"),
            Self::Legality { .. } => write!(f, "legality"),
            Self::CostInput { .. } => write!(f, "cost_input"),
            Self::Lowering { node_id, .. } => write!(f, "lowering(node:{node_id})"),
            Self::Emission {
                node_id,
                target_format,
                ..
            } => {
                write!(f, "emission(node:{node_id}, {target_format})")
            }
            Self::Artifact { .. } => write!(f, "artifact"),
        }
    }
}

/// Errors occurring during query execution, cycle detection, or cancellation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum QueryError {
    /// Dependency cycle detected in recursive query evaluations.
    #[error("Cyclic query dependency detected on {query}: path={path:?}")]
    Cycle {
        /// Query causing the cycle.
        query: QueryKey,
        /// Active query call path leading to the cycle.
        path: Vec<QueryKey>,
    },
    /// Query was cancelled in-flight by cancellation token.
    #[error("Query execution was cancelled")]
    Cancelled,
    /// Memory allocation budget exceeded by query cache or intermediate data.
    #[error("Query memory budget exceeded: requested {requested_bytes} bytes, limit is {budget_bytes} bytes")]
    MemoryExceeded {
        /// Requested byte allocation.
        requested_bytes: usize,
        /// Memory budget ceiling in bytes.
        budget_bytes: usize,
    },
    /// Stale or corrupted cache entry.
    #[error("Stale cache error: {0}")]
    StaleCache(#[from] StaleCacheError),
    /// Query engine lock was poisoned by a previous thread panic.
    #[error("query engine lock for `{lock_name}` was poisoned. Fix: rebuild the query engine")]
    LockPoisoned {
        /// Name of the poisoned lock.
        lock_name: String,
    },
    /// General query evaluation failure.
    #[error("Query execution failed: {0}")]
    ExecutionFailed(String),
}

/// Cooperative cancellation token passed through query executions.
#[derive(Debug, Default)]
pub struct CancellationToken {
    cancelled: AtomicBool,
}

impl CancellationToken {
    /// Create a new active cancellation token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal cancellation to all observing queries.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Check if cancelled, returning `QueryError::Cancelled` if set.
    pub fn check_cancelled(&self) -> Result<(), QueryError> {
        if self.is_cancelled() {
            Err(QueryError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Memory accounting for query cache and intermediate substrate allocations.
#[derive(Debug)]
pub struct MemoryAccounting {
    query_cache_bytes: AtomicUsize,
    max_budget_bytes: usize,
}

impl Default for MemoryAccounting {
    fn default() -> Self {
        Self {
            query_cache_bytes: AtomicUsize::new(0),
            // Default 1GB memory budget.
            max_budget_bytes: 1024 * 1024 * 1024,
        }
    }
}

impl MemoryAccounting {
    /// Create memory accounting with an explicit budget limit in bytes.
    #[must_use]
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            query_cache_bytes: AtomicUsize::new(0),
            max_budget_bytes: budget_bytes,
        }
    }

    /// Check if allocating `additional_bytes` would exceed budget.
    pub fn check_budget(&self, additional_bytes: usize) -> Result<(), QueryError> {
        let current = self.query_cache_bytes.load(Ordering::Relaxed);
        if current + additional_bytes > self.max_budget_bytes {
            return Err(QueryError::MemoryExceeded {
                requested_bytes: current + additional_bytes,
                budget_bytes: self.max_budget_bytes,
            });
        }
        Ok(())
    }

    /// Record an allocation of bytes.
    pub fn record_alloc(&self, bytes: usize) {
        self.query_cache_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a deallocation of bytes.
    pub fn record_dealloc(&self, bytes: usize) {
        self.query_cache_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Current allocated cache memory in bytes.
    #[must_use]
    pub fn current_bytes(&self) -> usize {
        self.query_cache_bytes.load(Ordering::Relaxed)
    }

    /// Budget ceiling in bytes.
    #[must_use]
    pub fn budget_bytes(&self) -> usize {
        self.max_budget_bytes
    }
}

/// Output payload and receipt returned by a deterministic query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryOutput {
    /// Serialized output data or artifact bytes.
    pub output_bytes: Vec<u8>,
    /// Deterministic diagnostic messages in canonical sorted order.
    pub diagnostics: Vec<String>,
    /// Cryptographic receipt digest over output bytes and diagnostics.
    pub receipt_hash: [u8; 32],
    /// Versioned cache key associated with this query output.
    pub cache_key: VersionedCacheKey,
}

impl QueryOutput {
    /// Construct a new query output, computing canonical receipt hash.
    #[must_use]
    pub fn new(
        cache_key: VersionedCacheKey,
        output_bytes: Vec<u8>,
        mut diagnostics: Vec<String>,
    ) -> Self {
        diagnostics.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&output_bytes);
        for diag in &diagnostics {
            hasher.update(diag.as_bytes());
        }
        let receipt_hash = *hasher.finalize().as_bytes();
        Self {
            output_bytes,
            diagnostics,
            receipt_hash,
            cache_key,
        }
    }
}

/// Trait defining an executable compiler query.
pub trait Query: Send + Sync {
    /// Output produced by this query.
    type Output: Clone + Send + Sync + 'static;

    /// Canonical query key identifying this query.
    fn key(&self) -> QueryKey;

    /// Direct dependency query keys declared by this query.
    fn declared_dependencies(&self) -> Vec<QueryKey> {
        Vec::new()
    }

    /// Execute the query computation.
    fn compute(
        &self,
        engine: &QueryEngine,
        token: &CancellationToken,
    ) -> Result<Self::Output, QueryError>;

    /// Project the output to canonical bytes for caching and receipt verification.
    fn to_query_output(&self, output: &Self::Output, key: VersionedCacheKey) -> QueryOutput;
}

#[derive(Debug, Clone)]
struct CachedQueryResult {
    output: QueryOutput,
    revision: Revision,
    dependencies: Vec<QueryKey>,
}

/// Single deterministic query engine coordinating compiler data, analysis, and caching.
pub struct QueryEngine {
    revision: AtomicU64,
    cache: RwLock<FxHashMap<QueryKey, CachedQueryResult>>,
    reverse_deps: RwLock<FxHashMap<QueryKey, BTreeSet<QueryKey>>>,
    active_stack: Mutex<Vec<QueryKey>>,
    substrate: Arc<SubstrateArena>,
    memory: MemoryAccounting,
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryEngine {
    /// Create a new query engine with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            revision: AtomicU64::new(1),
            cache: RwLock::new(FxHashMap::default()),
            reverse_deps: RwLock::new(FxHashMap::default()),
            active_stack: Mutex::new(Vec::new()),
            substrate: Arc::new(SubstrateArena::new()),
            memory: MemoryAccounting::default(),
        }
    }

    /// Create a query engine with a specific memory budget.
    #[must_use]
    pub fn with_memory_budget(budget_bytes: usize) -> Self {
        Self {
            revision: AtomicU64::new(1),
            cache: RwLock::new(FxHashMap::default()),
            reverse_deps: RwLock::new(FxHashMap::default()),
            active_stack: Mutex::new(Vec::new()),
            substrate: Arc::new(SubstrateArena::new()),
            memory: MemoryAccounting::with_budget(budget_bytes),
        }
    }

    /// Access the shared substrate arena.
    #[must_use]
    pub fn substrate(&self) -> &Arc<SubstrateArena> {
        &self.substrate
    }

    /// Access memory accounting metrics.
    #[must_use]
    pub fn memory_accounting(&self) -> &MemoryAccounting {
        &self.memory
    }

    /// Current monotonic revision of the engine.
    #[must_use]
    pub fn current_revision(&self) -> Revision {
        Revision(self.revision.load(Ordering::Relaxed))
    }

    /// Advance the engine revision, signaling graph mutation or invalidation.
    pub fn advance_revision(&self) -> Revision {
        Revision(self.revision.fetch_add(1, Ordering::SeqCst) + 1)
    }

    /// Execute a query deterministically with cycle detection, cancellation, and caching.
    pub fn execute<Q: Query>(
        &self,
        query: &Q,
        token: &CancellationToken,
    ) -> Result<QueryOutput, QueryError> {
        token.check_cancelled()?;

        let key = query.key();

        // Check query cache.
        {
            let cache_guard = crate::failure_domain::govern_rwlock_read(
                &self.cache,
                "query_engine",
                "cache",
                crate::failure_domain::RecoveryClass::TransactionallyRecoverable,
            )
            .map_err(|_| QueryError::LockPoisoned {
                lock_name: "cache".to_string(),
            })?;
            if let Some(entry) = cache_guard.get(&key) {
                // Validate cache schema version.
                if entry.output.cache_key.schema_version == SUBSTRATE_CACHE_SCHEMA_VERSION {
                    return Ok(entry.output.clone());
                }
            }
        }

        // Cycle detection: check active query stack.
        {
            let mut stack = crate::failure_domain::govern_mutex(
                &self.active_stack,
                "query_engine",
                "active_stack",
                crate::failure_domain::RecoveryClass::TransactionallyRecoverable,
            )
            .map_err(|_| QueryError::LockPoisoned {
                lock_name: "active_stack".to_string(),
            })?;
            if stack.contains(&key) {
                return Err(QueryError::Cycle {
                    query: key.clone(),
                    path: stack.clone(),
                });
            }
            stack.push(key.clone());
        }

        // Run query computation with cleanup guard for cycle stack.
        struct StackGuard<'a> {
            stack: &'a Mutex<Vec<QueryKey>>,
        }
        impl Drop for StackGuard<'_> {
            fn drop(&mut self) {
                // Skipping the pop would strand this key on the stack and make
                // every later query for it report a false cycle, so the pop
                // happens whether or not a panic poisoned the lock.
                let mut stack = crate::failure_domain::reclaim_poisoned_mutex(
                    &self.stack,
                    "foundation substrate query engine",
                    "the query cycle stack",
                );
                stack.pop();
            }
        }
        let _guard = StackGuard {
            stack: &self.active_stack,
        };

        let result = query.compute(self, token)?;

        // Ensure cancellation check before committing output to cache.
        token.check_cancelled()?;

        let cache_key = VersionedCacheKey::new(
            key.compiler_level(),
            SUBSTRATE_CACHE_SCHEMA_VERSION,
            *blake3::hash(format!("{key:?}").as_bytes()).as_bytes(),
            0,
        );

        let output = query.to_query_output(&result, cache_key);
        let deps = query.declared_dependencies();

        // Memory accounting.
        let entry_bytes = output.output_bytes.len() + std::mem::size_of::<CachedQueryResult>();
        self.memory.check_budget(entry_bytes)?;
        self.memory.record_alloc(entry_bytes);

        // Commit to cache.
        {
            let mut cache_guard = crate::failure_domain::govern_rwlock_write(
                &self.cache,
                "query_engine",
                "cache",
                crate::failure_domain::RecoveryClass::TransactionallyRecoverable,
            )
            .map_err(|_| QueryError::LockPoisoned {
                lock_name: "cache".to_string(),
            })?;
            cache_guard.insert(
                key.clone(),
                CachedQueryResult {
                    output: output.clone(),
                    revision: self.current_revision(),
                    dependencies: deps.clone(),
                },
            );
        }

        // Update reverse dependencies for precise invalidation.
        {
            let mut rev_guard = crate::failure_domain::govern_rwlock_write(
                &self.reverse_deps,
                "query_engine",
                "reverse_deps",
                crate::failure_domain::RecoveryClass::TransactionallyRecoverable,
            )
            .map_err(|_| QueryError::LockPoisoned {
                lock_name: "reverse_deps".to_string(),
            })?;
            for dep in deps {
                rev_guard.entry(dep).or_default().insert(key.clone());
            }
        }
        Ok(output)
    }

    /// Parallel deterministic query evaluation.
    ///
    /// Evaluates queries concurrently across available threads, then commits results
    /// in canonical sorted key order. Produces identical bytes, diagnostics, receipts,
    /// and cache keys at every thread count.
    pub fn execute_parallel<Q: Query>(
        &self,
        queries: &[Q],
        token: &CancellationToken,
    ) -> Result<Vec<QueryOutput>, QueryError> {
        token.check_cancelled()?;

        if queries.is_empty() {
            return Ok(Vec::new());
        }

        // Sort inputs into canonical key order.
        let mut sorted_indices: Vec<usize> = (0..queries.len()).collect();
        sorted_indices.sort_by_key(|&idx| queries[idx].key());

        // Concurrent execution across chunks.
        let results = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(sorted_indices.len());
            for &idx in &sorted_indices {
                let q = &queries[idx];
                handles.push(s.spawn(move || self.execute(q, token)));
            }

            let mut outputs = Vec::with_capacity(handles.len());
            for handle in handles {
                outputs.push(handle.join().expect("Thread panicked in query execution")?);
            }
            Ok::<Vec<QueryOutput>, QueryError>(outputs)
        })?;

        // Order results to match original query slice indices.
        let mut ordered_outputs = vec![None; queries.len()];
        for (i, &original_idx) in sorted_indices.iter().enumerate() {
            ordered_outputs[original_idx] = Some(results[i].clone());
        }

        Ok(ordered_outputs
            .into_iter()
            .map(|opt| opt.expect("All outputs present"))
            .collect())
    }

    /// Invalidate queries matching `dirty_keys` and all their transitive dependents.
    ///
    /// Returns the total number of invalidated query cache slots.
    pub fn invalidate(&self, dirty_keys: &[QueryKey]) -> usize {
        let mut to_invalidate = BTreeSet::new();
        let mut worklist: Vec<QueryKey> = dirty_keys.to_vec();

        let rev_guard = match crate::failure_domain::govern_rwlock_read(
            &self.reverse_deps,
            "query_engine",
            "reverse_deps",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
        ) {
            Ok(guard) => guard,
            Err(_) => return 0,
        };
        while let Some(key) = worklist.pop() {
            if to_invalidate.insert(key.clone()) {
                if let Some(dependents) = rev_guard.get(&key) {
                    for dep in dependents {
                        worklist.push(dep.clone());
                    }
                }
            }
        }
        drop(rev_guard);

        let mut cache_guard = match crate::failure_domain::govern_rwlock_write_with_reset(
            &self.cache,
            "query_engine",
            "cache",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
            |c| c.clear(),
        ) {
            Ok(guard) => guard,
            Err(_) => return 0,
        };
        let count = to_invalidate.len();
        for key in &to_invalidate {
            if let Some(removed) = cache_guard.remove(key) {
                let bytes =
                    removed.output.output_bytes.len() + std::mem::size_of::<CachedQueryResult>();
                self.memory.record_dealloc(bytes);
            }
        }

        count
    }

    /// Total number of active cached query results.
    #[must_use]
    pub fn cached_count(&self) -> usize {
        crate::failure_domain::govern_rwlock_read(
            &self.cache,
            "query_engine",
            "cache",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
        )
        .map(|g| g.len())
        .unwrap_or(0)
    }

    /// Check whether a query key is cached.
    #[must_use]
    pub fn is_cached(&self, key: &QueryKey) -> bool {
        crate::failure_domain::govern_rwlock_read(
            &self.cache,
            "query_engine",
            "cache",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
        )
        .map(|g| g.contains_key(key))
        .unwrap_or(false)
    }
    /// Revision when a cached entry was computed.
    #[must_use]
    pub fn entry_revision(&self, key: &QueryKey) -> Option<Revision> {
        crate::failure_domain::govern_rwlock_read(
            &self.cache,
            "query_engine",
            "cache",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
        )
        .ok()
        .and_then(|g| g.get(key).map(|e| e.revision))
    }

    /// Declared dependencies of a cached entry.
    #[must_use]
    pub fn entry_dependencies(&self, key: &QueryKey) -> Option<Vec<QueryKey>> {
        crate::failure_domain::govern_rwlock_read(
            &self.cache,
            "query_engine",
            "cache",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
        )
        .ok()
        .and_then(|g| g.get(key).map(|e| e.dependencies.clone()))
    }

    /// Clear all cached query results.
    pub fn clear_cache(&self) {
        let _unused_cache = crate::failure_domain::govern_rwlock_write_with_reset(
            &self.cache,
            "query_engine",
            "cache",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
            |c| c.clear(),
        );
        let _unused_deps = crate::failure_domain::govern_rwlock_write_with_reset(
            &self.reverse_deps,
            "query_engine",
            "reverse_deps",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
            |r| r.clear(),
        );
        self.memory.query_cache_bytes.store(0, Ordering::Relaxed);
    }
}
