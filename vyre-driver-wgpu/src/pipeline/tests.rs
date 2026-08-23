#[cfg(feature = "device-tests")]
use std::hash::BuildHasherDefault;
#[cfg(feature = "device-tests")]
use std::sync::Arc;

use vyre_driver::tuner::Mode;
use vyre_driver::validation::LaunchGeometryLimits;
#[cfg(feature = "device-tests")]
use vyre_driver::BackendError;
#[cfg(feature = "device-tests")]
use vyre_driver::DEFAULT_PIPELINE_CACHE_ENTRIES;
use vyre_foundation::execution_plan::{self, ReadbackStrategy};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryKind, Node, Program};

use super::tuning::wgpu_effective_dispatch_config_for_limits;
use super::{enforce_actual_output_budget, DispatchConfig};
#[cfg(feature = "device-tests")]
use super::{BindGroupLayoutCache, WgpuPipeline};
#[cfg(feature = "device-tests")]
use crate::buffer::BufferPool;
#[cfg(feature = "device-tests")]
use crate::engine::record_and_readback::{record_and_readback, DispatchLabels, RecordAndReadback};
#[cfg(feature = "device-tests")]
use crate::runtime::cache::pipeline::LruPipelineCache;
#[cfg(feature = "device-tests")]
use crate::runtime::device::EnabledFeatures;
#[cfg(feature = "device-tests")]
use crate::DispatchArena;

/// Device, queue, dispatch config and the two compile caches every pipeline
/// contract test needs. Each test used to spell this block out again.
#[cfg(feature = "device-tests")]
struct PipelineHarness {
    device_queue: Arc<(wgpu::Device, wgpu::Queue)>,
    adapter_info: wgpu::AdapterInfo,
    enabled_features: EnabledFeatures,
    config: DispatchConfig,
    pipeline_cache: Arc<LruPipelineCache>,
    layout_cache: Arc<BindGroupLayoutCache>,
}

#[cfg(feature = "device-tests")]
impl PipelineHarness {
    /// `purpose` completes "Fix: GPU required for {purpose}" when no device opens.
    fn new(purpose: &str) -> Self {
        let ((device, queue), adapter_info, enabled_features) = crate::runtime::init_device()
            .unwrap_or_else(|err| panic!("Fix: GPU required for {purpose}: {err:?}"));
        Self {
            device_queue: Arc::new((device, queue)),
            adapter_info,
            enabled_features,
            config: DispatchConfig::default(),
            pipeline_cache: Arc::new(LruPipelineCache::new(DEFAULT_PIPELINE_CACHE_ENTRIES as u32)),
            layout_cache: Arc::new(BindGroupLayoutCache::with_hasher(BuildHasherDefault::<
                rustc_hash::FxHasher,
            >::default())),
        }
    }

    /// A dispatch arena over this harness's device and queue.
    fn arena(&self) -> Arc<DispatchArena> {
        Arc::new(DispatchArena::new(
            self.device_queue.0.clone(),
            self.device_queue.1.clone(),
            &self.config,
        ))
    }

    /// Compile against the shared caches, binding `pool` as the persistent pool.
    fn compile(
        &self,
        program: &Program,
        pool: BufferPool,
    ) -> Result<Arc<WgpuPipeline>, BackendError> {
        WgpuPipeline::compile_with_device_queue(
            program,
            &self.config,
            self.adapter_info.clone(),
            self.enabled_features,
            self.device_queue.clone(),
            pool,
            self.pipeline_cache.clone(),
            self.layout_cache.clone(),
            None,
        )
    }

    /// Compile against the pool `arena` owns, so buffer Arc identities match
    /// between compile-time bindings and run-time recording. A separate
    /// `BufferPool::new()` would make every dispatch a bind-group-cache miss.
    fn compile_on_arena(
        &self,
        program: &Program,
        arena: &Arc<DispatchArena>,
    ) -> Result<Arc<WgpuPipeline>, BackendError> {
        self.compile(program, arena.pool().clone())
    }
}

/// A one-node program storing `value` at index 0 of a `count`-element `u32`
/// output buffer named `name`.
///
/// The minimum program that produces an observable output. Six contract tests
/// spelled it out, so a change to the fixture shape had to be applied six
/// times or the tests stopped exercising the same program.
#[cfg(feature = "device-tests")]
fn stores_u32(name: &str, count: u32, value: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output(name, 0, DataType::U32).with_count(count)],
        [1, 1, 1],
        vec![Node::store(name, Expr::u32(0), Expr::u32(value))],
    )
}

/// One direct dispatch through the shared record path. Every pipeline contract
/// test issues a single unprofiled 1x1x1 dispatch with no inputs over the
/// arena's own pool; only the debug labels and whether readback rings are in
/// play differ.
#[cfg(feature = "device-tests")]
fn record_once(
    pipeline: &WgpuPipeline,
    arena: &DispatchArena,
    readback_rings: bool,
    labels: DispatchLabels,
) -> Result<vyre_driver::OutputBuffers, BackendError> {
    let empty_inputs: [&[u8]; 0] = [];
    record_and_readback(RecordAndReadback {
        device_queue: &pipeline.device_queue,
        pool: arena.pool(),
        readback_rings: readback_rings.then(|| arena.readback_rings()),
        pipeline: &pipeline.pipeline,
        bind_group_layouts: &pipeline.bind_group_layouts,
        bind_group_cache: Some(pipeline.bind_group_cache.as_ref()),
        buffer_bindings: &pipeline.buffer_bindings,
        inputs: &empty_inputs,
        output_bindings: Arc::clone(&pipeline.output_bindings),
        trap_tags: &pipeline.trap_tags,
        workgroup_count: [1, 1, 1],
        indirect: pipeline.indirect.as_ref(),
        labels,
        iterations: 1,
        timestamp_profile: false,
        inferred_grid_shape: None,
    })
}

#[cfg(feature = "device-tests")]
mod bind_group_cache_contracts {
    use super::*;

    /// PERF-HOT-01: two WgpuPipeline instances for the same compiled shader
    /// must share one BindGroupCache (Arc identity). Different compiled
    /// shaders must have independent caches.
    #[test]
    fn bind_group_cache_shared_per_compiled_shader() {
        let harness = PipelineHarness::new("cache-sharing test");
        let pool = BufferPool::new(
            harness.device_queue.0.clone(),
            harness.device_queue.1.clone(),
            &harness.config,
        );
        let layout_cache = Arc::clone(&harness.layout_cache);

        let program1 = stores_u32("out", 4, 7);

        let p1 = harness
            .compile(&program1, pool.clone())
            .expect("Fix: first compile must succeed; restore this invariant before continuing.");
        assert_eq!(
            layout_cache.len(),
            1,
            "Fix: first compile should insert one shared bind-group layout fingerprint"
        );

        let p2 = harness
            .compile(&program1, pool.clone())
            .expect("Fix: second compile of same program must succeed; restore this invariant before continuing.");
        assert_eq!(
            layout_cache.len(),
            1,
            "Fix: recompiling the same layout must hit the shared layout cache"
        );

        assert!(
            Arc::ptr_eq(&p1.bind_group_cache, &p2.bind_group_cache),
            "Fix: same compiled shader must share BindGroupCache (HOT-01)"
        );

        let (input_handles, mut output_handles) = p1.legacy_handles_from_inputs(&[]).expect(
            "Fix: legacy handle creation must succeed; restore this invariant before continuing.",
        );
        p1.dispatch_persistent(&input_handles, &mut output_handles, None, [1, 1, 1])
            .expect("Fix: first dispatch must succeed; restore this invariant before continuing.");
        let stats_after_miss = p1.bind_group_cache_stats();
        assert_eq!(
            stats_after_miss.misses, 1,
            "Fix: first dispatch of a new signature must be a cache miss"
        );
        assert_eq!(stats_after_miss.hits, 0);

        p1.dispatch_persistent(&input_handles, &mut output_handles, None, [1, 1, 1])
            .expect("Fix: second dispatch must succeed; restore this invariant before continuing.");
        let stats_after_hit = p1.bind_group_cache_stats();
        assert_eq!(
            stats_after_hit.hits, 1,
            "Fix: second dispatch with identical handles must be a cache hit"
        );
        assert_eq!(stats_after_hit.misses, 1);

        let program2 = stores_u32("out2", 8, 42);

        let p3 = harness.compile(&program2, pool).expect(
            "Fix: compile of different program must succeed; restore this invariant before continuing.",
        );
        assert_eq!(
            layout_cache.len(),
            1,
            "Fix: compatible output-only programs must share the same wgpu bind-group layout cache entry"
        );

        assert!(
            !Arc::ptr_eq(&p1.bind_group_cache, &p3.bind_group_cache),
            "Fix: different compiled shaders must have independent BindGroupCaches"
        );
    }

    #[test]
    fn compiled_borrowed_timed_dispatch_reports_device_ns() {
        use vyre_driver::CompiledPipeline;

        let harness = PipelineHarness::new("compiled timing test");
        let device = &harness.device_queue.0;
        assert!(
            device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
                && device
                    .features()
                    .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            "Fix: WGPU compiled timing test requires timestamp query features to be negotiated."
        );
        let arena = harness.arena();

        let program = stores_u32("out", 1, 7);
        let pipeline = harness
            .compile_on_arena(&program, &arena)
            .expect("Fix: compiled timed dispatch test pipeline must compile.");

        let timed = pipeline
            .dispatch_borrowed_timed(&[], &harness.config)
            .expect("Fix: compiled borrowed timed dispatch must succeed.");
        assert_eq!(
            u32::from_le_bytes(timed.outputs[0][0..4].try_into().unwrap()),
            7
        );
        assert!(
            timed.device_ns.is_some_and(|ns| ns > 0),
            "Fix: WGPU compiled borrowed timed dispatch must report GPU device nanoseconds."
        );
        assert!(timed.enqueue_ns.is_some_and(|ns| ns > 0));
        assert!(timed.wait_ns.is_some_and(|ns| ns > 0));
    }
}

mod layout_config_contracts {
    use super::*;
    use vyre_driver::LaunchGeometry;

    #[test]
    fn hex_short_truncates_to_eight_bytes() {
        let hash = *blake3::hash(b"vyre-pipeline").as_bytes();
        let expected = vyre_driver::hex_encode(&hash[..8]);
        assert_eq!(vyre_driver::hex_short(&hash).len(), 16);
        assert_eq!(vyre_driver::hex_short(&hash), expected);
    }

    #[test]
    fn actual_output_budget_rejects_combined_outputs() {
        let mut config = DispatchConfig::default();
        config.max_output_bytes = Some(3);
        let err = enforce_actual_output_budget(&config, &[vec![0; 2], vec![0; 2]])
            .expect_err("combined readback over budget must fail");
        assert!(
            err.to_string().contains("max_output_bytes"),
            "Fix: budget rejection must name the violated policy, got {err}"
        );
    }

    #[test]
    fn output_layout_matches_trimmed_execution_plan() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)
                .with_count(1024)
                .with_output_byte_range(4..12)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
        );
        let plan = execution_plan::plan(&program).expect(
            "Fix: trimmed output program must plan; restore this invariant before continuing.",
        );
        assert_eq!(
            plan.strategy.readback,
            ReadbackStrategy::Trimmed {
                visible_bytes: 8,
                avoided_bytes: 4088,
            }
        );
        let layouts = vyre_driver::output_binding_layouts(&program)
            .expect("Fix: layout must derive; restore this invariant before continuing.");
        assert_eq!(layouts[0].layout.read_size, 8);
        assert_eq!(layouts[0].layout.copy_size, 8);
    }

    #[test]
    fn wgpu_compile_config_receives_natural_gradient_workgroup_before_lowering() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(4096)],
            [32, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
        );
        let limits = LaunchGeometryLimits {
            backend: "wgpu-test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 0,
        };

        let effective = wgpu_effective_dispatch_config_for_limits(
            &program,
            &DispatchConfig::default(),
            limits,
            Mode::NaturalGradient,
            LaunchGeometry::Untracked,
        )
        .expect("Fix: WGPU natural-gradient config derivation must be pure and valid");

        assert_eq!(
            effective.workgroup_override,
            Some([1024, 1, 1]),
            "Fix: WGPU lowering config must include the natural-gradient workgroup so WGSL @workgroup_size and dispatch metadata agree. WebGPU reports no per-compute-unit thread budget (max_threads_per_sm 0), so residency-aware cold start is inert here and this width is unchanged by it."
        );
    }

    #[test]
    fn wgpu_natural_gradient_compile_config_preserves_semantic_safety_gates() {
        let program = Program::wrapped(
            vec![
                BufferDecl::output("out", 0, DataType::U32).with_count(4096),
                BufferDecl::workgroup("scratch", 64, DataType::U32).with_kind(MemoryKind::Shared),
            ],
            [64, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
        );
        let limits = LaunchGeometryLimits {
            backend: "wgpu-test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 0,
        };
        let mut explicit = DispatchConfig::default();
        explicit.workgroup_override = Some([256, 1, 1]);

        let explicit_effective = wgpu_effective_dispatch_config_for_limits(
            &program,
            &explicit,
            limits,
            Mode::NaturalGradient,
            LaunchGeometry::Untracked,
        )
        .expect("Fix: explicit WGPU workgroup override must stay valid");
        assert_eq!(explicit_effective.workgroup_override, Some([256, 1, 1]));

        let shared_effective = wgpu_effective_dispatch_config_for_limits(
            &program,
            &DispatchConfig::default(),
            limits,
            Mode::NaturalGradient,
            LaunchGeometry::Untracked,
        )
        .expect("Fix: shared-memory WGPU config should remain valid without autotuning");
        assert_eq!(
            shared_effective.workgroup_override, None,
            "Fix: workgroup-local scratch kernels must keep the Program-declared WGPU workgroup."
        );
    }

    /// WHY: 150.15. The compiler searches the workgroup dimension and records the
    /// winning geometry in the artifact, and the authenticated module declares that
    /// shape. A launch tuner that picked another width would dispatch a kernel nobody
    /// compiled. Before this, the wgpu path pinned the descriptor width through a
    /// dispatch override, so any caller override applied first won instead.
    #[test]
    fn recorded_artifact_geometry_outranks_the_launch_tuner_and_caller_overrides() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(4096)],
            [32, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
        );
        let limits = LaunchGeometryLimits {
            backend: "wgpu-test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 0,
        };
        let mut caller_pinned = DispatchConfig::default();
        caller_pinned.workgroup_override = Some([256, 1, 1]);

        for config in [DispatchConfig::default(), caller_pinned] {
            let effective = wgpu_effective_dispatch_config_for_limits(
                &program,
                &config,
                limits,
                Mode::NaturalGradient,
                LaunchGeometry::Compiled([64, 1, 1]),
            )
            .expect("Fix: a recorded compiled geometry must resolve without error");
            assert_eq!(
                effective.workgroup_override,
                Some([64, 1, 1]),
                "Fix: the recorded artifact geometry must win over both the launch tuner and a caller override."
            );
        }
    }

    /// WHY: 150.15 boundary. A descriptor that records no geometry is an invalid
    /// artifact, not an invitation to choose one. A silent fall back to the declared
    /// or tuned width would launch a shape the artifact never authenticated.
    #[test]
    fn a_descriptor_without_recorded_geometry_is_an_error() {
        for absent in [[0, 1, 1], [64, 0, 1], [64, 1, 0], [0, 0, 0]] {
            let error = LaunchGeometry::from_recorded(absent, "wgpu")
                .expect_err("Fix: an absent geometry record must fail the launch");
            assert!(
                error.message().contains("records no workgroup geometry"),
                "Fix: the error must name the missing record, got {error}"
            );
        }
        assert_eq!(
            LaunchGeometry::from_recorded([64, 1, 1], "wgpu").expect("a full record is valid"),
            LaunchGeometry::Compiled([64, 1, 1])
        );
    }
}

#[cfg(feature = "device-tests")]
mod prerecorded_contracts {
    use super::*;

    /// Pre-recording a persistent dispatch builds bind groups and records the
    /// compute pass through the same code the direct persistent path uses, only
    /// under its own wgpu labels. Replaying the recorded command buffer must
    /// therefore land the same bytes in the output buffer that a direct dispatch
    /// of the same program lands.
    #[test]
    fn prerecorded_replay_writes_the_same_output_as_direct_dispatch() {
        let harness = PipelineHarness::new("pre-recorded dispatch replay test");
        let arena = harness.arena();

        let program = stores_u32("out", 4, 7);

        let pipeline = harness
            .compile_on_arena(&program, &arena)
            .expect("Fix: pre-recorded dispatch test pipeline must compile.");

        let direct = record_once(
            &pipeline,
            &arena,
            false,
            DispatchLabels {
                bind_group: "vyre prerecord parity direct bind group",
                encoder: "vyre prerecord parity direct",
                compute: "vyre prerecord parity direct compute",
            },
        )
        .expect("Fix: direct persistent dispatch must succeed before comparing against replay.");

        let prerecorded = pipeline
            .prerecord_borrowed_dispatch(&[], [1, 1, 1])
            .expect("Fix: pre-recording a persistent dispatch must succeed.");
        prerecorded
            .replay(&harness.device_queue.1)
            .expect("Fix: first replay of a pre-recorded command buffer must succeed.");
        let replayed = prerecorded
            .read_output(0)
            .expect("Fix: reading a replayed output buffer must succeed.");

        assert_eq!(
            u32::from_le_bytes(replayed[0..4].try_into().unwrap()),
            7,
            "Fix: replayed pre-recorded dispatch must write the program's stored value."
        );
        assert_eq!(
            replayed[0..16],
            direct[0][0..16],
            "Fix: pre-recorded replay and direct persistent dispatch must produce identical output bytes."
        );
    }

    /// A wgpu command buffer is single-submit. The second replay must be a
    /// structured error rather than a raw wgpu panic.
    #[test]
    fn prerecorded_second_replay_is_a_structured_error() {
        let harness = PipelineHarness::new("pre-recorded dispatch resubmit test");
        let arena = harness.arena();

        let program = stores_u32("out", 1, 3);

        let pipeline = harness
            .compile_on_arena(&program, &arena)
            .expect("Fix: pre-recorded resubmit test pipeline must compile.");
        let prerecorded = pipeline
            .prerecord_borrowed_dispatch(&[], [1, 1, 1])
            .expect("Fix: pre-recording a persistent dispatch must succeed.");

        prerecorded
            .replay(&harness.device_queue.1)
            .expect("Fix: first replay of a pre-recorded command buffer must succeed.");
        let error = prerecorded
            .replay(&harness.device_queue.1)
            .expect_err("Fix: a pre-recorded command buffer must refuse a second submission.");
        assert!(
            error.to_string().contains("already submitted"),
            "Fix: expected the single-submit diagnostic, got: {error}"
        );
    }
}

#[cfg(feature = "device-tests")]
mod readback_ring_contracts {
    use super::*;

    #[test]
    fn direct_record_and_readback_reuses_bind_groups() {
        let harness = PipelineHarness::new("direct cache test");
        let arena = harness.arena();

        let program = stores_u32("out", 4, 7);

        let pipeline = harness
            .compile_on_arena(&program, &arena)
            .expect("Fix: compile must succeed; restore this invariant before continuing.");

        for _ in 0..2 {
            let outputs = record_once(
                &pipeline,
                &arena,
                false,
                DispatchLabels {
                    bind_group: "vyre direct cache test bind group",
                    encoder: "vyre direct cache test",
                    compute: "vyre direct cache test compute",
                },
            )
            .expect(
                "Fix: direct record/readback must succeed; restore this invariant before continuing.",
            );
            assert_eq!(u32::from_le_bytes(outputs[0][0..4].try_into().unwrap()), 7);
        }

        let stats = pipeline.bind_group_cache_stats();
        // The pool may or may not return the same buffer Arc across two
        // back-to-back readbacks (the prior readback's pinning, plus
        // size-class bucketing, decides). What we DO require: the cache
        // is exercised on every dispatch (misses + hits >= 2) and never
        // reports a negative-cost path (no double-build for a given Arc).
        let total = stats.misses + stats.hits;
        assert!(
            total >= 2,
            "two dispatches should each consult the bind-group cache (got misses={}, hits={})",
            stats.misses,
            stats.hits,
        );
        assert!(
            stats.misses <= 2,
            "no more than one bind-group build per distinct buffer identity (got misses={})",
            stats.misses,
        );
    }

    #[test]
    fn direct_record_and_readback_trap_uses_readback_rings_only() {
        let harness = PipelineHarness::new("trap-sidecar allocation test");
        let with_rings_arena = harness.arena();
        let with_rings_pool = with_rings_arena.pool().clone();

        let program = Program::wrapped(
            vec![],
            [1, 1, 1],
            vec![Node::trap(Expr::u32(3), "direct-readback-ring-trap")],
        );

        let pipeline = harness
            .compile_on_arena(&program, &with_rings_arena)
            .expect(
            "Fix: trapped program compile must succeed; restore this invariant before continuing.",
        );

        let before_allocations = with_rings_pool.stats().allocations;
        let error = record_once(
            &pipeline,
            &with_rings_arena,
            true,
            DispatchLabels {
                bind_group: "vyre direct trap readback ring test bind group",
                encoder: "vyre direct trap readback ring test",
                compute: "vyre direct trap readback ring test compute",
            },
        )
        .expect_err(
            "Fix: trapped dispatch with readback rings must return the underlying trap sidecar error and not succeed",
        );
        let after_allocations = with_rings_pool.stats().allocations;

        assert!(
            error.to_string().contains("wgpu dispatch trapped"),
            "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
        );
        assert!(
            error.to_string().contains("direct-readback-ring-trap"),
            "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
        );
        assert_eq!(
            after_allocations,
            before_allocations + 1,
            "Fix: readback-ring trap path must use the dedicated trap sidecar buffer only (no pooled full-sidecar readback buffer allocation).",
        );
    }

    #[test]

    fn direct_record_and_readback_trap_without_readback_rings_allocates_full_sidecar_copy() {
        let harness = PipelineHarness::new("trap-sidecar allocation delta test");
        let arena = harness.arena();
        let pool = arena.pool().clone();

        let program = Program::wrapped(
            vec![],
            [1, 1, 1],
            vec![Node::trap(Expr::u32(5), "direct-readback-no-ring-trap")],
        );

        let pipeline = harness.compile_on_arena(&program, &arena).expect(
            "Fix: trapped program compile must succeed; restore this invariant before continuing.",
        );

        let before_allocations = pool.stats().allocations;
        let error = record_once(
            &pipeline,
            &arena,
            false,
            DispatchLabels {
                bind_group: "vyre direct trap readback no-ring test bind group",
                encoder: "vyre direct trap readback no-ring test",
                compute: "vyre direct trap readback no-ring test compute",
            },
        )
        .expect_err(
            "Fix: trapped dispatch without rings must still return the underlying trap sidecar error and not succeed",
        );
        let after_allocations = pool.stats().allocations;

        assert!(
            error.to_string().contains("wgpu dispatch trapped"),
            "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
        );
        assert!(
            error.to_string().contains("direct-readback-no-ring-trap"),
            "Fix: expected the trap tag to be preserved across fallback sidecar decode, got: {error}"
        );
        assert_eq!(
            after_allocations,
            before_allocations + 2,
            "Fix: non-ring trap path must allocate exactly the full-sidecar pooled readback buffer plus trap sidecar allocation (before={before_allocations}, after={after_allocations})."
        );
    }
}

#[cfg(feature = "device-tests")]
mod trap_output_contracts {
    use super::*;

    #[test]
    fn direct_record_and_readback_trap_with_output_preserves_ring_fast_path() {
        let harness = PipelineHarness::new("trap+output readback allocation contract test");
        let with_rings_arena = harness.arena();
        let without_rings_arena = harness.arena();
        let with_rings_pool = with_rings_arena.pool().clone();
        let without_rings_pool = without_rings_arena.pool().clone();

        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::store("out", Expr::u32(0), Expr::u32(99)),
                Node::trap(Expr::u32(9), "mixed-output-ring-trap"),
            ],
        );

        let pipeline = harness
            .compile_on_arena(&program, &with_rings_arena)
            .expect("Fix: trapped program with output compile must succeed; restore this invariant before continuing.");

        let with_rings_before = with_rings_pool.stats().allocations;
        let with_rings_error = record_once(
            &pipeline,
            &with_rings_arena,
            true,
            DispatchLabels {
                bind_group: "vyre mixed output ring test bind group",
                encoder: "vyre mixed output ring test",
                compute: "vyre mixed output ring test compute",
            },
        )
        .expect_err(
            "Fix: trapped dispatch with output and rings must still surface trap errors and not succeed",
        );
        let with_rings_after = with_rings_pool.stats().allocations;

        assert!(
            with_rings_error
                .to_string()
                .contains("wgpu dispatch trapped"),
            "Fix: expected trap dispatch to surface a backend trap error, got: {with_rings_error}"
        );
        assert!(
            with_rings_error.to_string().contains("mixed-output-ring-trap"),
            "Fix: expected trap tag to be preserved through mixed-output ring path, got: {with_rings_error}"
        );
        assert_eq!(
            with_rings_after,
            with_rings_before + 2,
            "Fix: ring-backed mixed output+trap path should add only output + trap buffer allocations from pool before first successful mapping.",
        );

        let without_rings_before = without_rings_pool.stats().allocations;
        let without_rings_error = record_once(
            &pipeline,
            &without_rings_arena,
            false,
            DispatchLabels {
                bind_group: "vyre mixed output no-ring test bind group",
                encoder: "vyre mixed output no-ring test",
                compute: "vyre mixed output no-ring test compute",
            },
        )
        .expect_err(
            "Fix: trapped dispatch without rings should surface the trap error and not succeed",
        );
        let without_rings_after = without_rings_pool.stats().allocations;

        assert!(
            without_rings_error
                .to_string()
                .contains("wgpu dispatch trapped"),
            "Fix: expected trap dispatch to surface a backend trap error, got: {without_rings_error}"
        );
        assert!(
            without_rings_error.to_string().contains("mixed-output-ring-trap"),
            "Fix: expected trap tag to be preserved through mixed-output fallback path, got: {without_rings_error}"
        );
        assert_eq!(
            without_rings_after,
            without_rings_before + 4,
            "Fix: no-ring mixed output+trap path should allocate output storage, trap storage, output readback, and trap readback buffers; ring-backed dispatch must be the path that avoids the two pooled readback allocations.",
        );
    }
}
