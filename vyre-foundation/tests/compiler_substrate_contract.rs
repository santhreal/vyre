//! Contract and regression tests for compiler data substrate, arenas, and query engine (Row 89).
//!
//! Tests verify:
//! 1. Hash-consed arenas and canonical interners for strings, types, constants, layouts, expressions, nodes, and regions.
//! 2. Stage-specific read-only views and level boundary smuggling enforcement across all five compiler levels.
//! 3. Deterministic query engine evaluation: parallel and serial evaluations produce byte-identical output, diagnostics, receipts, and cache keys.
//! 4. Versioned cache keys and fail-closed rejection of stale cache entries.
//! 5. Precise invalidation and dependency tracking over revision transitions.
//! 6. Cooperative cancellation leaving zero partial cache entries or corrupted state.
//! 7. Asymptotic memory accounting and budget ceiling enforcement.
//! 8. Pass immutability contracts derived from source at runtime.

use vyre_foundation::ir::{DataType, Expr, Node, ProgramGraph};
use vyre_foundation::substrate::{
    derive_registered_pass_descriptors, enforce_level_boundary, CancellationToken, CanonicalConst,
    CanonicalLayout, CompilerLevelStage, LevelAccessError, Query, QueryEngine, QueryError,
    QueryKey, QueryOutput, StaleCacheError, SubstrateArena, VersionedCacheEntry, VersionedCacheKey,
    WholeProgramGraphView, SUBSTRATE_CACHE_SCHEMA_VERSION,
};

/// Test query simulating deterministic parsing and validation.
#[derive(Debug, Clone)]
struct MockValidationQuery {
    key: QueryKey,
    payload: Vec<u8>,
    diagnostics: Vec<String>,
}

impl Query for MockValidationQuery {
    type Output = Vec<u8>;

    fn key(&self) -> QueryKey {
        self.key.clone()
    }

    fn compute(
        &self,
        _engine: &QueryEngine,
        token: &CancellationToken,
    ) -> Result<Self::Output, QueryError> {
        token.check_cancelled()?;
        // Deterministic compute step
        let mut out = self.payload.clone();
        out.reverse();
        Ok(out)
    }

    fn to_query_output(&self, output: &Self::Output, key: VersionedCacheKey) -> QueryOutput {
        QueryOutput::new(key, output.clone(), self.diagnostics.clone())
    }
}

/// Test query simulating dependent semantic facts inference.
#[derive(Debug, Clone)]
struct MockDependentQuery {
    key: QueryKey,
    parent_key: QueryKey,
    node_id: u32,
}

impl Query for MockDependentQuery {
    type Output = u32;

    fn key(&self) -> QueryKey {
        self.key.clone()
    }

    fn declared_dependencies(&self) -> Vec<QueryKey> {
        vec![self.parent_key.clone()]
    }

    fn compute(
        &self,
        _engine: &QueryEngine,
        token: &CancellationToken,
    ) -> Result<Self::Output, QueryError> {
        token.check_cancelled()?;
        Ok(self.node_id * 42)
    }

    fn to_query_output(&self, output: &Self::Output, key: VersionedCacheKey) -> QueryOutput {
        QueryOutput::new(
            key,
            output.to_le_bytes().to_vec(),
            vec![format!("node_fact:{}", self.node_id)],
        )
    }
}

/// Test query simulating cyclic dependencies.
#[derive(Debug, Clone)]
struct MockCyclicQuery {
    key: QueryKey,
    partner_key: QueryKey,
}

impl Query for MockCyclicQuery {
    type Output = u32;

    fn key(&self) -> QueryKey {
        self.key.clone()
    }

    fn compute(
        &self,
        engine: &QueryEngine,
        token: &CancellationToken,
    ) -> Result<Self::Output, QueryError> {
        token.check_cancelled()?;
        // Mutual recursion to trigger cycle detection
        let sub_query = MockCyclicQuery {
            key: self.partner_key.clone(),
            partner_key: self.key.clone(),
        };
        let _ = engine.execute(&sub_query, token)?;
        Ok(100)
    }

    fn to_query_output(&self, output: &Self::Output, key: VersionedCacheKey) -> QueryOutput {
        QueryOutput::new(key, output.to_le_bytes().to_vec(), Vec::new())
    }
}

#[test]
fn hash_consed_arenas_and_interners_guarantee_structural_sharing() {
    let substrate = SubstrateArena::new();

    // 1. String Interning
    let id_a1 = substrate.strings.intern("buffer_alpha");
    let id_a2 = substrate.strings.intern("buffer_alpha");
    let id_b = substrate.strings.intern("buffer_beta");

    assert_eq!(id_a1, id_a2, "Identical strings must share stable ID");
    assert_ne!(id_a1, id_b, "Distinct strings must have distinct IDs");
    assert_eq!(
        substrate.strings.lookup(id_a1).unwrap().as_ref(),
        "buffer_alpha"
    );
    assert_eq!(substrate.strings.len(), 2);

    // 2. Type Interning
    let ty_f32_1 = substrate.types.intern_scalar(DataType::F32);
    let ty_f32_2 = substrate.types.intern_scalar(DataType::F32);
    let ty_u32 = substrate.types.intern_scalar(DataType::U32);

    assert_eq!(ty_f32_1, ty_f32_2);
    assert_ne!(ty_f32_1, ty_u32);
    assert_eq!(substrate.types.len(), 2);

    // 3. Constant Interning
    let c1 = substrate.constants.intern(CanonicalConst::U32(1024));
    let c2 = substrate.constants.intern(CanonicalConst::U32(1024));
    let c3 = substrate.constants.intern(CanonicalConst::U32(2048));
    assert_eq!(c1, c2);
    assert_ne!(c1, c3);

    // 4. Layout Interning
    let l1 = substrate.layouts.intern(CanonicalLayout {
        strides: vec![1024, 1],
        alignment_bytes: 128,
        pitch_bytes: Some(1024),
    });
    let l2 = substrate.layouts.intern(CanonicalLayout {
        strides: vec![1024, 1],
        alignment_bytes: 128,
        pitch_bytes: Some(1024),
    });
    assert_eq!(l1, l2);

    // 5. Expression Hash-Consing
    let expr1 = Expr::add(Expr::var("x"), Expr::LitU32(42));
    let expr2 = Expr::add(Expr::var("x"), Expr::LitU32(42));
    let expr3 = Expr::add(Expr::var("y"), Expr::LitU32(42));

    let eid1 = substrate.intern_expr(&expr1);
    let eid2 = substrate.intern_expr(&expr2);

    assert_eq!(
        eid1, eid2,
        "Equivalent expressions must hash-cons to same ExprId"
    );
    assert_eq!(
        substrate.intern_expr(&substrate.expr(eid1)),
        eid1,
        "A rebuilt expression must intern back to its own id"
    );

    let before_expr3 = substrate.expr_count();
    let eid3 = substrate.intern_expr(&expr3);
    assert_ne!(eid1, eid3, "Distinct expressions must have distinct ExprId");
    assert_eq!(
        substrate.expr_count() - before_expr3,
        2,
        "The `42` leaf is already interned, so only `y` and the new sum are added"
    );

    // 6. Node Hash-Consing
    let node1 = Node::let_bind("res", Expr::LitU32(100));
    let node2 = Node::let_bind("res", Expr::LitU32(100));
    let nid1 = substrate.nodes.intern(node1);
    let nid2 = substrate.nodes.intern(node2);
    assert_eq!(nid1, nid2);
    assert_eq!(substrate.nodes.len(), 1);

    assert!(substrate.total_allocated_bytes() > 0);
}

#[test]
fn stage_views_enforce_immutability_and_level_smuggling_boundary() {
    let graph = ProgramGraph::new();
    let view = WholeProgramGraphView::new(&graph);

    assert_eq!(view.node_count(), 0);
    assert_eq!(view.value_count(), 0);
    assert_eq!(view.level(), CompilerLevelStage::WholeProgramGraph);

    // Level boundary enforcement:
    // Upper-level queries (Level 1 or 2) CANNOT access Level 4 or 5 objects.
    assert!(enforce_level_boundary(
        CompilerLevelStage::WholeProgramGraph,
        CompilerLevelStage::WholeProgramGraph
    )
    .is_ok());

    assert!(enforce_level_boundary(
        CompilerLevelStage::TargetPayload,
        CompilerLevelStage::WholeProgramGraph
    )
    .is_ok());

    let smuggling_err = enforce_level_boundary(
        CompilerLevelStage::WholeProgramGraph,
        CompilerLevelStage::PhysicalKernel,
    );
    assert!(matches!(
        smuggling_err,
        Err(LevelAccessError::SmugglingViolation {
            caller: CompilerLevelStage::WholeProgramGraph,
            target: CompilerLevelStage::PhysicalKernel,
        })
    ));

    let smuggling_err_payload = enforce_level_boundary(
        CompilerLevelStage::LogicalRegion,
        CompilerLevelStage::TargetPayload,
    );
    assert!(matches!(
        smuggling_err_payload,
        Err(LevelAccessError::SmugglingViolation {
            caller: CompilerLevelStage::LogicalRegion,
            target: CompilerLevelStage::TargetPayload,
        })
    ));
}

#[test]
fn parallel_and_serial_query_evaluations_produce_byte_identical_results() {
    let engine_serial = QueryEngine::new();
    let engine_parallel = QueryEngine::new();
    let token = CancellationToken::new();

    let mut queries = Vec::new();
    for i in 0..64 {
        let mut digest = [0u8; 32];
        digest[0] = (i & 0xFF) as u8;
        digest[1] = ((i >> 8) & 0xFF) as u8;

        queries.push(MockValidationQuery {
            key: QueryKey::ParseValidate {
                program_digest: digest,
            },
            payload: format!("program_ir_payload_chunk_{i}").into_bytes(),
            diagnostics: vec![
                format!("diag_warning_b_{i}"),
                format!("diag_info_a_{i}"),
                format!("diag_note_c_{i}"),
            ],
        });
    }

    // 1. Serial evaluation
    let mut serial_outputs = Vec::new();
    for q in &queries {
        let out = engine_serial
            .execute(q, &token)
            .expect("Serial query execution must succeed");
        serial_outputs.push(out);
    }

    // 2. Parallel evaluation
    let parallel_outputs = engine_parallel
        .execute_parallel(&queries, &token)
        .expect("Parallel query execution must succeed");

    assert_eq!(serial_outputs.len(), parallel_outputs.len());

    for (idx, (s_out, p_out)) in serial_outputs
        .iter()
        .zip(parallel_outputs.iter())
        .enumerate()
    {
        assert_eq!(
            s_out.output_bytes, p_out.output_bytes,
            "Byte mismatch at index {idx} between serial and parallel run"
        );
        assert_eq!(
            s_out.diagnostics, p_out.diagnostics,
            "Diagnostics mismatch at index {idx}"
        );
        assert_eq!(
            s_out.receipt_hash, p_out.receipt_hash,
            "Receipt digest mismatch at index {idx}"
        );
        assert_eq!(
            s_out.cache_key, p_out.cache_key,
            "Cache key mismatch at index {idx}"
        );
    }
}

#[test]
fn stale_cached_entry_under_changed_key_or_version_is_rejected() {
    let current_version = SUBSTRATE_CACHE_SCHEMA_VERSION;
    let stale_version = current_version + 1;
    let target_fp = 0xDEAD_BEEF_0000_1234;
    let wrong_target_fp = 0xCAFE_BABE_0000_5678;

    let key_valid = VersionedCacheKey::new(
        CompilerLevelStage::SelectedSchedule,
        current_version,
        [7u8; 32],
        target_fp,
    );

    let key_stale_version = VersionedCacheKey::new(
        CompilerLevelStage::SelectedSchedule,
        stale_version,
        [7u8; 32],
        target_fp,
    );

    let key_wrong_target = VersionedCacheKey::new(
        CompilerLevelStage::SelectedSchedule,
        current_version,
        [7u8; 32],
        wrong_target_fp,
    );

    let valid_entry = VersionedCacheEntry::new(key_valid, vec![1, 2, 3, 4], 1);
    let stale_entry = VersionedCacheEntry::new(key_stale_version, vec![1, 2, 3, 4], 1);
    let wrong_target_entry = VersionedCacheEntry::new(key_wrong_target, vec![1, 2, 3, 4], 1);

    // Valid entry passes
    assert!(valid_entry.validate(current_version, target_fp).is_ok());

    // Stale version fails closed with VersionMismatch
    let version_err = stale_entry.validate(current_version, target_fp);
    assert!(matches!(
        version_err,
        Err(StaleCacheError::VersionMismatch {
            expected: 1,
            found: 2,
        })
    ));

    // Wrong target fails closed with TargetMismatch
    let target_err = wrong_target_entry.validate(current_version, target_fp);
    assert!(matches!(
        target_err,
        Err(StaleCacheError::TargetMismatch {
            expected: 0xDEAD_BEEF_0000_1234,
            found: 0xCAFE_BABE_0000_5678,
        })
    ));
}

#[test]
fn precise_invalidation_clears_only_transitive_dependents() {
    let engine = QueryEngine::new();
    let token = CancellationToken::new();

    let parent_key_1 = QueryKey::ParseValidate {
        program_digest: [1u8; 32],
    };
    let parent_key_2 = QueryKey::ParseValidate {
        program_digest: [2u8; 32],
    };

    let child_key_1 = QueryKey::SemanticFacts {
        node_id: 10,
        program_digest: [1u8; 32],
    };
    let child_key_2 = QueryKey::SemanticFacts {
        node_id: 20,
        program_digest: [2u8; 32],
    };

    let q_p1 = MockValidationQuery {
        key: parent_key_1.clone(),
        payload: vec![10, 20],
        diagnostics: Vec::new(),
    };
    let q_p2 = MockValidationQuery {
        key: parent_key_2.clone(),
        payload: vec![30, 40],
        diagnostics: Vec::new(),
    };

    let q_c1 = MockDependentQuery {
        key: child_key_1.clone(),
        parent_key: parent_key_1.clone(),
        node_id: 10,
    };
    let q_c2 = MockDependentQuery {
        key: child_key_2.clone(),
        parent_key: parent_key_2.clone(),
        node_id: 20,
    };

    engine.execute(&q_p1, &token).unwrap();
    engine.execute(&q_p2, &token).unwrap();
    engine.execute(&q_c1, &token).unwrap();
    engine.execute(&q_c2, &token).unwrap();

    assert_eq!(engine.cached_count(), 4);
    assert!(engine.is_cached(&parent_key_1));
    assert!(engine.is_cached(&child_key_1));
    assert!(engine.is_cached(&parent_key_2));
    assert!(engine.is_cached(&child_key_2));

    // Invalidate parent 1: only parent 1 and child 1 must be removed.
    // Parent 2 and child 2 MUST remain intact in cache!
    let invalidated_count = engine.invalidate(&[parent_key_1.clone()]);
    assert_eq!(
        invalidated_count, 2,
        "Parent 1 + Child 1 must be invalidated"
    );

    assert!(!engine.is_cached(&parent_key_1), "Parent 1 must be purged");
    assert!(!engine.is_cached(&child_key_1), "Child 1 must be purged");
    assert!(
        engine.is_cached(&parent_key_2),
        "Parent 2 must be preserved"
    );
    assert!(engine.is_cached(&child_key_2), "Child 2 must be preserved");
    assert_eq!(engine.cached_count(), 2);
}

#[test]
fn cyclic_query_dependencies_are_detected_and_rejected() {
    let engine = QueryEngine::new();
    let token = CancellationToken::new();

    let key_a = QueryKey::EquivalenceFacts {
        op_name: "op_alpha".into(),
    };
    let key_b = QueryKey::EquivalenceFacts {
        op_name: "op_beta".into(),
    };

    let query_a = MockCyclicQuery {
        key: key_a.clone(),
        partner_key: key_b.clone(),
    };

    let result = engine.execute(&query_a, &token);
    assert!(
        matches!(result, Err(QueryError::Cycle { .. })),
        "Cyclic query dependency must return QueryError::Cycle"
    );
}

#[test]
fn query_cancellation_leaves_zero_partial_cache_entries() {
    let engine = QueryEngine::new();
    let token = CancellationToken::new();

    let key = QueryKey::ParseValidate {
        program_digest: [9u8; 32],
    };
    let query = MockValidationQuery {
        key: key.clone(),
        payload: vec![1, 2, 3],
        diagnostics: Vec::new(),
    };

    // Cancel token before / during execution
    token.cancel();

    let result = engine.execute(&query, &token);
    assert!(matches!(result, Err(QueryError::Cancelled)));

    // Cache MUST remain clean with zero entries
    assert_eq!(engine.cached_count(), 0);
    assert!(!engine.is_cached(&key));
}

#[test]
fn memory_budget_ceiling_is_enforced() {
    // Engine with tiny 64-byte memory budget
    let engine = QueryEngine::with_memory_budget(64);
    let token = CancellationToken::new();

    let key = QueryKey::ParseValidate {
        program_digest: [5u8; 32],
    };
    let query = MockValidationQuery {
        key: key.clone(),
        payload: vec![0u8; 256], // 256 bytes exceeds 64 byte budget
        diagnostics: Vec::new(),
    };

    let result = engine.execute(&query, &token);
    assert!(matches!(result, Err(QueryError::MemoryExceeded { .. })));
    assert_eq!(engine.cached_count(), 0);
}

#[test]
fn registered_passes_satisfy_immutability_and_no_unversioned_caching_contracts() {
    let passes = derive_registered_pass_descriptors();
    assert!(!passes.is_empty(), "Pass registry must not be empty");

    for pass in passes {
        assert!(
            pass.preserves_ir_immutability,
            "Pass {} must preserve IR immutability",
            pass.name
        );
        assert!(
            pass.forbids_unversioned_pointer_caching,
            "Pass {} must forbid unversioned pointer caching",
            pass.name
        );
        assert!(
            pass.zero_alloc_on_no_change,
            "Pass {} must guarantee zero allocations on no-op",
            pass.name
        );
    }
}
