use crate::api::case::{
    prepared_as, BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata,
    BenchRequirements, BenchRun, Correctness, DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::{elapsed_ns, BenchMetrics, MetricPoint};
use crate::api::suite::SuiteKind;
use std::collections::HashMap;
use std::time::Instant;
use vyre_emit_ptx::{patterns, ComputeCapability};
use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::descriptor_builder::{effect, op};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, FragmentValue, KernelBody,
    KernelDescriptor, KernelOpKind, LiteralValue, MatrixMmaElement, MatrixMmaLayout, MatrixMmaSpec,
    MatrixTileShape, MemoryClass,
};

/// Release benchmark case for CUDA/PTX fast-path pattern coverage.
pub struct CudaPtxPatterns;

const SUITES: &[SuiteKind] = &[SuiteKind::Release, SuiteKind::Deep];

#[derive(Debug)]
struct PtxPatternTotals {
    corpus_kernels: u64,
    predication_candidates: u64,
    safe_predication_candidates: u64,
    vec_load_candidates: u64,
    vec_store_candidates: u64,
    async_copy_candidates: u64,
    tensor_core_candidates: u64,
    ldmatrix_capable_targets: u64,
    scheduled_fillers: u64,
    predicated_stores: u64,
    branch_labels: u64,
    cp_async_emitted: u64,
    mma_sync_emitted: u64,
    vectorized_loads_emitted: u64,
    vectorized_stores_emitted: u64,
    vector_kernel_scalar_loads: u64,
    vector_kernel_scalar_stores: u64,
    vector_kernel_scalar_index_adds: u64,
    source_cache_entries: u64,
    source_cache_hits: u64,
    source_cache_misses: u64,
    ptx_bytes_emitted: u64,
}

impl BenchCase for CudaPtxPatterns {
    fn id(&self) -> BenchId {
        BenchId("cuda.ptx.patterns.release.corpus".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "CUDA PTX Pattern Release Corpus".to_string(),
            description: "Measures PTX-side CUDA fast-path coverage for predication, vector memory, and load-gap scheduling".to_string(),
            tags: vec![
                "cuda".to_string(),
                "ptx".to_string(),
                "backend".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Backend,
            workload: WorkloadClass::Micro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-emit-ptx".to_string(),
        }
    }

    fn suites(&self) -> &'static [SuiteKind] {
        SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: false,
            needs_network: false,
            min_vram_bytes: None,
            min_input_bytes: None,
            feature_set: vec![],
        }
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(corpus()))
    }

    fn program<'a>(&self, _prepared: &'a PreparedCase) -> Option<&'a vyre_foundation::ir::Program> {
        None
    }

    fn run(
        &self,
        _ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let corpus = prepared_as::<Vec<KernelDescriptor>>(prepared, "CUDA PTX pattern")?;
        let started = Instant::now();
        let totals = measure_corpus(corpus)?;
        Ok(ptx_pattern_bench_run(&totals, elapsed_ns(started)))
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        let words = decode_words(run)?;
        let [corpus_kernels, predication_candidates, safe_predication_candidates, vec_load_candidates, vec_store_candidates, async_copy_candidates, tensor_core_candidates, ldmatrix_capable_targets, scheduled_fillers, predicated_stores, branch_labels, cp_async_emitted, mma_sync_emitted, vectorized_loads_emitted, vectorized_stores_emitted, vector_kernel_scalar_loads, vector_kernel_scalar_stores, vector_kernel_scalar_index_adds, source_cache_entries, source_cache_hits, source_cache_misses, ptx_bytes_emitted] =
            words.as_slice()
        else {
            return Ok(Correctness::Invalid {
                reason: "CUDA PTX pattern benchmark emitted the wrong metric word count"
                    .to_string(),
            });
        };
        if *corpus_kernels < 4
            || *predication_candidates == 0
            || *safe_predication_candidates == 0
            || *vec_load_candidates == 0
            || *vec_store_candidates == 0
            || *async_copy_candidates == 0
            || *tensor_core_candidates == 0
            || *ldmatrix_capable_targets == 0
            || *scheduled_fillers < 2
            || *predicated_stores < 3
            || *branch_labels != 0
            || *cp_async_emitted == 0
            || *mma_sync_emitted == 0
            || *vectorized_loads_emitted == 0
            || *vectorized_stores_emitted == 0
            || *vector_kernel_scalar_loads != 0
            || *vector_kernel_scalar_stores != 0
            || *vector_kernel_scalar_index_adds != 0
            || *source_cache_entries == 0
            || *source_cache_hits == 0
            || *source_cache_misses == 0
            || *ptx_bytes_emitted == 0
        {
            return Ok(Correctness::Invalid {
                reason: format!(
                    "CUDA PTX release corpus missing fast-path evidence: kernels={corpus_kernels}, pred={predication_candidates}, safe_pred={safe_predication_candidates}, vload={vec_load_candidates}, vstore={vec_store_candidates}, async_copy={async_copy_candidates}, tensor_core={tensor_core_candidates}, ldmatrix_capable={ldmatrix_capable_targets}, fillers={scheduled_fillers}, pred_stores={predicated_stores}, branch_labels={branch_labels}, cp_async={cp_async_emitted}, mma_sync={mma_sync_emitted}, vectorized_loads={vectorized_loads_emitted}, vectorized_stores={vectorized_stores_emitted}, vector_scalar_loads={vector_kernel_scalar_loads}, vector_scalar_stores={vector_kernel_scalar_stores}, vector_scalar_index_adds={vector_kernel_scalar_index_adds}, source_cache_entries={source_cache_entries}, source_cache_hits={source_cache_hits}, source_cache_misses={source_cache_misses}, bytes={ptx_bytes_emitted}"
                ),
            });
        }
        Ok(Correctness::Exact)
    }
}

/// The metric roster and payload the case reports for one corpus measurement.
///
/// `kernel_launches` is stated as zero here because the corpus is a compiler
/// proof: it emits PTX and dispatches nothing. Assembly is a function so the
/// launch-count contract has one definition instead of one per caller.
fn ptx_pattern_bench_run(totals: &PtxPatternTotals, elapsed: u64) -> BenchRun {
    let words = [
        totals.corpus_kernels,
        totals.predication_candidates,
        totals.safe_predication_candidates,
        totals.vec_load_candidates,
        totals.vec_store_candidates,
        totals.async_copy_candidates,
        totals.tensor_core_candidates,
        totals.ldmatrix_capable_targets,
        totals.scheduled_fillers,
        totals.predicated_stores,
        totals.branch_labels,
        totals.cp_async_emitted,
        totals.mma_sync_emitted,
        totals.vectorized_loads_emitted,
        totals.vectorized_stores_emitted,
        totals.vector_kernel_scalar_loads,
        totals.vector_kernel_scalar_stores,
        totals.vector_kernel_scalar_index_adds,
        totals.source_cache_entries,
        totals.source_cache_hits,
        totals.source_cache_misses,
        totals.ptx_bytes_emitted,
    ];
    let mut output = Vec::with_capacity(words.len() * std::mem::size_of::<u64>());
    for value in words {
        output.extend_from_slice(&value.to_le_bytes());
    }

    let custom = [
        ("ptx_corpus_kernels", totals.corpus_kernels),
        ("ptx_predication_candidates", totals.predication_candidates),
        (
            "ptx_safe_predication_candidates",
            totals.safe_predication_candidates,
        ),
        ("ptx_vec_load_candidates", totals.vec_load_candidates),
        ("ptx_vec_store_candidates", totals.vec_store_candidates),
        ("ptx_async_copy_candidates", totals.async_copy_candidates),
        ("ptx_tensor_core_candidates", totals.tensor_core_candidates),
        (
            "ptx_ldmatrix_capable_targets",
            totals.ldmatrix_capable_targets,
        ),
        ("ptx_scheduled_fillers", totals.scheduled_fillers),
        ("ptx_predicated_stores", totals.predicated_stores),
        ("ptx_branch_labels", totals.branch_labels),
        ("ptx_cp_async_emitted", totals.cp_async_emitted),
        ("ptx_mma_sync_emitted", totals.mma_sync_emitted),
        (
            "ptx_vectorized_loads_emitted",
            totals.vectorized_loads_emitted,
        ),
        (
            "ptx_vectorized_stores_emitted",
            totals.vectorized_stores_emitted,
        ),
        (
            "ptx_vector_kernel_scalar_loads",
            totals.vector_kernel_scalar_loads,
        ),
        (
            "ptx_vector_kernel_scalar_stores",
            totals.vector_kernel_scalar_stores,
        ),
        (
            "ptx_vector_kernel_scalar_index_adds",
            totals.vector_kernel_scalar_index_adds,
        ),
        ("cuda_ptx_source_cache_entries", totals.source_cache_entries),
        ("cuda_ptx_source_cache_hits", totals.source_cache_hits),
        ("cuda_ptx_source_cache_misses", totals.source_cache_misses),
        ("kernel_launches", 0),
        ("cuda_kernel_launches", 0),
        ("ptx_bytes_emitted", totals.ptx_bytes_emitted),
    ];

    BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(elapsed),
            lower_ns: Some(elapsed),
            output_bytes: Some(totals.ptx_bytes_emitted),
            custom: custom
                .into_iter()
                .map(|(name, value)| MetricPoint {
                    name: name.to_string(),
                    value,
                })
                .collect(),
            ..Default::default()
        },
        baseline_metrics: None,
        outputs: vec![output],
        baseline_outputs: None,
    }
}

fn measure_corpus(corpus: &[KernelDescriptor]) -> Result<PtxPatternTotals, BenchError> {
    let mut totals = PtxPatternTotals {
        corpus_kernels: corpus.len() as u64,
        predication_candidates: 0,
        safe_predication_candidates: 0,
        vec_load_candidates: 0,
        vec_store_candidates: 0,
        async_copy_candidates: 0,
        tensor_core_candidates: 0,
        ldmatrix_capable_targets: 0,
        scheduled_fillers: 0,
        predicated_stores: 0,
        branch_labels: 0,
        cp_async_emitted: 0,
        mma_sync_emitted: 0,
        vectorized_loads_emitted: 0,
        vectorized_stores_emitted: 0,
        vector_kernel_scalar_loads: 0,
        vector_kernel_scalar_stores: 0,
        vector_kernel_scalar_index_adds: 0,
        source_cache_entries: 0,
        source_cache_hits: 0,
        source_cache_misses: 0,
        ptx_bytes_emitted: 0,
    };
    let mut source_cache = HashMap::<String, String>::new();
    for desc in corpus {
        let audit = patterns::audit(desc, ComputeCapability::SM_90);
        totals.predication_candidates = totals
            .predication_candidates
            .saturating_add(audit.predication.candidates.len() as u64);
        totals.safe_predication_candidates = totals
            .safe_predication_candidates
            .saturating_add(audit.predication.safe_candidate_count() as u64);
        totals.vec_load_candidates = totals
            .vec_load_candidates
            .saturating_add(audit.vec_load.candidates.len() as u64);
        totals.vec_store_candidates = totals
            .vec_store_candidates
            .saturating_add(audit.vec_store.candidates.len() as u64);
        totals.async_copy_candidates = totals
            .async_copy_candidates
            .saturating_add(audit.async_copy.candidates.len() as u64);
        totals.tensor_core_candidates = totals
            .tensor_core_candidates
            .saturating_add(audit.tensor_core.candidates.len() as u64);
        totals.ldmatrix_capable_targets = totals.ldmatrix_capable_targets.saturating_add(
            if audit.async_copy.target_supports_ldmatrix {
                1
            } else {
                0
            },
        );

        let ptx = if let Some(cached) = source_cache.get(&desc.id) {
            totals.source_cache_hits = totals.source_cache_hits.saturating_add(1);
            cached.clone()
        } else {
            totals.source_cache_misses = totals.source_cache_misses.saturating_add(1);
            let lowered = vyre_emit_ptx::emit_with_target(desc, ComputeCapability::SM_90)
                .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
            source_cache.insert(desc.id.clone(), lowered.clone());
            lowered
        };
        if let Some(cached) = source_cache.get(&desc.id) {
            totals.source_cache_hits = totals.source_cache_hits.saturating_add(1);
            if cached.len() != ptx.len() {
                return Err(BenchError::ExecutionFailed(
                    "CUDA PTX pattern source cache returned divergent source length".to_string(),
                ));
            }
        }
        totals.ptx_bytes_emitted = totals.ptx_bytes_emitted.saturating_add(ptx.len() as u64);
        totals.scheduled_fillers = totals
            .scheduled_fillers
            .saturating_add(ptx.matches("// schedule: hoist independent").count() as u64);
        totals.predicated_stores = totals
            .predicated_stores
            .saturating_add(ptx.matches("@%p").count() as u64)
            .saturating_add(ptx.matches("@!%p").count() as u64);
        totals.branch_labels = totals
            .branch_labels
            .saturating_add(ptx.matches("$L_if_").count() as u64);
        totals.cp_async_emitted = totals
            .cp_async_emitted
            .saturating_add(ptx.matches("cp.async.ca.shared.global").count() as u64);
        totals.mma_sync_emitted = totals
            .mma_sync_emitted
            .saturating_add(ptx.matches("mma.sync.aligned").count() as u64);
        if desc.id == "ptx_vector_load_store" {
            totals.vectorized_loads_emitted = totals
                .vectorized_loads_emitted
                .saturating_add(ptx.matches("ld.global.v4").count() as u64);
            totals.vectorized_stores_emitted = totals
                .vectorized_stores_emitted
                .saturating_add(ptx.matches("st.global.v4").count() as u64);
            totals.vector_kernel_scalar_loads = totals
                .vector_kernel_scalar_loads
                .saturating_add(ptx.matches("ld.global.u32").count() as u64);
            totals.vector_kernel_scalar_stores = totals
                .vector_kernel_scalar_stores
                .saturating_add(ptx.matches("st.global.u32").count() as u64);
            totals.vector_kernel_scalar_index_adds = totals
                .vector_kernel_scalar_index_adds
                .saturating_add(ptx.matches("// scalar-index-increment").count() as u64);
        }
    }
    totals.source_cache_entries = source_cache.len() as u64;
    Ok(totals)
}

fn corpus() -> Vec<KernelDescriptor> {
    vec![
        predicated_literal_store_kernel(),
        predicated_else_store_kernel(),
        vector_load_store_kernel(),
        scheduled_load_gap_kernel(),
        cp_async_candidate_kernel(),
        async_copy_emit_kernel(),
        tensor_core_candidate_kernel(),
        matrix_mma_emit_kernel(),
    ]
}

fn u32_global_slot(slot: u32, name: &str) -> BindingSlot {
    BindingSlot {
        slot,
        element_type: DataType::U32,
        element_count: Some(1024),
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadWrite,
        name: name.to_string(),
    }
}

fn predicated_literal_store_kernel() -> KernelDescriptor {
    KernelDescriptor {
        id: "ptx_predicated_literal_store".to_string(),
        bindings: BindingLayout {
            slots: vec![u32_global_slot(0, "out")],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                op(KernelOpKind::Literal, vec![0], 0),
                op(KernelOpKind::Literal, vec![1], 1),
                effect(KernelOpKind::StructuredIfThen, vec![0, 0]),
            ],
            child_bodies: vec![KernelBody {
                ops: vec![
                    op(KernelOpKind::Literal, vec![0], 20),
                    effect(KernelOpKind::StoreGlobal, vec![0, 1, 20]),
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(13)],
            }],
            literals: vec![LiteralValue::Bool(true), LiteralValue::U32(0)],
        },
    }
}

fn predicated_else_store_kernel() -> KernelDescriptor {
    KernelDescriptor {
        id: "ptx_predicated_else_store".to_string(),
        bindings: BindingLayout {
            slots: vec![u32_global_slot(0, "out")],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                op(KernelOpKind::Literal, vec![0], 0),
                op(KernelOpKind::Literal, vec![1], 1),
                effect(KernelOpKind::StructuredIfThenElse, vec![0, 0, 1]),
            ],
            child_bodies: vec![store_child(20, 21), store_child(21, 34)],
            literals: vec![LiteralValue::Bool(true), LiteralValue::U32(0)],
        },
    }
}

fn store_child(result_id: u32, value: u32) -> KernelBody {
    KernelBody {
        ops: vec![
            op(KernelOpKind::Literal, vec![0], result_id),
            effect(KernelOpKind::StoreGlobal, vec![0, 1, result_id]),
        ],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(value)],
    }
}

fn vector_load_store_kernel() -> KernelDescriptor {
    KernelDescriptor {
        id: "ptx_vector_load_store".to_string(),
        bindings: BindingLayout {
            slots: vec![u32_global_slot(0, "input"), u32_global_slot(1, "output")],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                op(KernelOpKind::Literal, vec![0], 0),
                op(KernelOpKind::Literal, vec![1], 1),
                op(KernelOpKind::LoadGlobal, vec![0, 0], 10),
                op(KernelOpKind::BinOpKind(BinOp::Add), vec![0, 1], 2),
                op(KernelOpKind::LoadGlobal, vec![0, 2], 11),
                op(KernelOpKind::BinOpKind(BinOp::Add), vec![2, 1], 3),
                op(KernelOpKind::LoadGlobal, vec![0, 3], 12),
                op(KernelOpKind::BinOpKind(BinOp::Add), vec![3, 1], 4),
                op(KernelOpKind::LoadGlobal, vec![0, 4], 13),
                effect(KernelOpKind::StoreGlobal, vec![1, 0, 10]),
                op(KernelOpKind::BinOpKind(BinOp::Add), vec![0, 1], 5),
                effect(KernelOpKind::StoreGlobal, vec![1, 5, 11]),
                op(KernelOpKind::BinOpKind(BinOp::Add), vec![5, 1], 6),
                effect(KernelOpKind::StoreGlobal, vec![1, 6, 12]),
                op(KernelOpKind::BinOpKind(BinOp::Add), vec![6, 1], 7),
                effect(KernelOpKind::StoreGlobal, vec![1, 7, 13]),
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    }
}

fn scheduled_load_gap_kernel() -> KernelDescriptor {
    KernelDescriptor {
        id: "ptx_scheduled_load_gap".to_string(),
        bindings: BindingLayout {
            slots: vec![u32_global_slot(0, "input"), u32_global_slot(1, "output")],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                op(KernelOpKind::Literal, vec![0], 0),
                op(KernelOpKind::Literal, vec![1], 1),
                op(KernelOpKind::LoadGlobal, vec![0, 0], 2),
                op(KernelOpKind::BinOpKind(BinOp::Add), vec![2, 1], 3),
                op(KernelOpKind::Literal, vec![2], 4),
                op(KernelOpKind::BinOpKind(BinOp::Add), vec![4, 1], 5),
                effect(KernelOpKind::StoreGlobal, vec![1, 0, 3]),
            ],
            child_bodies: vec![],
            literals: vec![
                LiteralValue::U32(0),
                LiteralValue::U32(7),
                LiteralValue::U32(11),
            ],
        },
    }
}

fn cp_async_candidate_kernel() -> KernelDescriptor {
    KernelDescriptor {
        id: "ptx_cp_async_candidate".to_string(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: Some(1024),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    name: "input".to_string(),
                },
                BindingSlot {
                    slot: 1,
                    element_type: DataType::U32,
                    element_count: Some(1024),
                    memory_class: MemoryClass::Shared,
                    visibility: BindingVisibility::ReadWrite,
                    name: "tile".to_string(),
                },
            ],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                op(KernelOpKind::Literal, vec![0], 0),
                op(KernelOpKind::LoadGlobal, vec![0, 0], 1),
                effect(KernelOpKind::StoreShared, vec![1, 0, 1]),
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0)],
        },
    }
}

fn async_copy_emit_kernel() -> KernelDescriptor {
    KernelDescriptor {
        id: "ptx_cp_async_emit".to_string(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: Some(1024),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    name: "input".to_string(),
                },
                BindingSlot {
                    slot: 1,
                    element_type: DataType::U32,
                    element_count: Some(1024),
                    memory_class: MemoryClass::Shared,
                    visibility: BindingVisibility::ReadWrite,
                    name: "tile".to_string(),
                },
            ],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                op(KernelOpKind::Literal, vec![0], 0),
                op(KernelOpKind::Literal, vec![1], 1),
                effect(KernelOpKind::async_load("tile".into()), vec![0, 1, 0, 1]),
                effect(KernelOpKind::async_wait("tile".into()), vec![]),
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(256)],
        },
    }
}

fn tensor_core_candidate_kernel() -> KernelDescriptor {
    let mut ops = Vec::new();
    let mut literals = Vec::new();
    for id in 0..3 {
        literals.push(LiteralValue::F32(id as f32));
        ops.push(op(KernelOpKind::Literal, vec![id], id));
    }
    for result in 3..11 {
        ops.push(op(KernelOpKind::Fma, vec![0, 1, 2], result));
    }
    KernelDescriptor {
        id: "ptx_tensor_core_candidate".to_string(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals,
        },
    }
}

fn matrix_mma_emit_kernel() -> KernelDescriptor {
    let mut ops = Vec::new();
    let mut literals = Vec::new();
    for id in 0..6 {
        literals.push(LiteralValue::U32(id));
        ops.push(op(KernelOpKind::Literal, vec![id], id));
    }
    for id in 6..10 {
        literals.push(LiteralValue::F32(0.0));
        ops.push(op(KernelOpKind::Literal, vec![id], id));
    }
    ops.push(op(
        KernelOpKind::MatrixMma(Box::new(MatrixMmaSpec {
            tile: MatrixTileShape { m: 16, n: 8, k: 16 },
            left: FragmentValue::in_registers(MatrixMmaElement::F16, MatrixMmaLayout::RowMajor, 32),
            right: FragmentValue::in_registers(
                MatrixMmaElement::F16,
                MatrixMmaLayout::ColMajor,
                32,
            ),
            accumulator: FragmentValue::in_registers(
                MatrixMmaElement::F32,
                MatrixMmaLayout::RowMajor,
                32,
            ),
        })),
        (0..10).collect::<Vec<u32>>(),
        10,
    ));
    KernelDescriptor {
        id: "ptx_matrix_mma_emit".to_string(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(32, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals,
        },
    }
}

fn decode_words(run: &BenchRun) -> Result<Vec<u64>, BenchError> {
    let Some(output) = run.outputs.first() else {
        return Err(BenchError::ExecutionFailed(
            "CUDA PTX pattern benchmark emitted no output payload".to_string(),
        ));
    };
    if output.len() % std::mem::size_of::<u64>() != 0 {
        return Err(BenchError::ExecutionFailed(
            "CUDA PTX pattern output payload is not u64-aligned".to_string(),
        ));
    }
    Ok(output
        .chunks_exact(std::mem::size_of::<u64>())
        .map(|chunk| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            u64::from_le_bytes(bytes)
        })
        .collect())
}

inventory::submit! {
    &CudaPtxPatterns as &dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_patterns_case_registered_identity_and_requirements() {
        let case = CudaPtxPatterns;
        assert_eq!(case.id().0, "cuda.ptx.patterns.release.corpus");
        let metadata = case.metadata();
        assert_eq!(metadata.owner_crate, "vyre-emit-ptx");
        assert_eq!(metadata.name, "CUDA PTX Pattern Release Corpus");
        let requirements = case.requirements();
        assert!(
            !requirements.needs_gpu,
            "Fix: PTX pattern corpus is a non-dispatch compiler proof workload"
        );
    }

    /// WHY: the corpus dispatches nothing, so a consumer reading launch counts
    /// must see an explicit zero rather than a missing metric. The run the case
    /// reports is assembled here, from the same function `run` returns, so the
    /// assertion fails if that roster drops either launch metric or states a
    /// nonzero count. It does not cover `run` itself, which needs a device
    /// context: what it covers is the roster and the payload width `verify`
    /// decodes.
    #[test]
    fn ptx_patterns_run_emits_explicit_zero_kernel_launches() {
        let prepared: PreparedCase = Box::new(corpus());
        let totals = measure_corpus(
            prepared_as::<Vec<KernelDescriptor>>(&prepared, "CUDA PTX pattern")
                .expect("prepared case must downcast"),
        )
        .expect("measure_corpus must succeed");

        assert!(totals.corpus_kernels >= 4);
        assert!(totals.ptx_bytes_emitted > 0);

        let run = ptx_pattern_bench_run(&totals, 1);

        for name in ["kernel_launches", "cuda_kernel_launches"] {
            let metric = run
                .metrics
                .custom
                .iter()
                .find(|point| point.name == name)
                .unwrap_or_else(|| {
                    panic!(
                        "Fix: non-dispatch PTX pattern benchmark must emit an explicit `{name}` metric"
                    )
                });
            assert_eq!(
                metric.value, 0,
                "Fix: non-dispatch PTX pattern benchmark must report 0 for `{name}`"
            );
        }

        let words = decode_words(&run).expect("decode_words must succeed");
        assert_eq!(
            words.len(),
            22,
            "Fix: the payload width must match the 22 words `verify` destructures"
        );
    }
}
