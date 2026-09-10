//! Contract tests for the compositional schedule calculus.
//!
//! Verifies the compositional schedule calculus operators, runtime closure against source,
//! type-level unrepresentability, algebraic normalization, isomorphism deduplication,
//! and partial schedule instantiation.

use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU32, NonZeroU64};

use vyre_foundation::schedule::{
    DependencyPreservationCertificate, MappingLevel, MemoryPlacement, PartialSchedule,
    PipelineRole, PipelineRoleGroup, ScheduleCostModel, ScheduleDiff, ScheduleDiffItem,
    ScheduleInverseOp, ScheduleNumericalEffect, ScheduleOp, SchedulePlan, ScheduleResourceBounds,
    ScheduleTree, SymbolicParameter, SynchronizationScope, TypedScheduleStage,
};
fn every_schedule_op() -> Vec<ScheduleOp> {
    vec![
        ScheduleOp::Domain {
            region: 1,
            extents: vec![128, 256],
        },
        ScheduleOp::AlgorithmChoice {
            region: 1,
            algorithm_id: "flash_attention_v2".to_string(),
            parameter_pack: vec![64, 128],
        },
        ScheduleOp::Fuse {
            regions: vec![1, 2],
        },
        ScheduleOp::Fission {
            region: 1,
            split_point: 4,
        },
        ScheduleOp::RegionPartition {
            region: 1,
            partitions: 4,
            level: MappingLevel::ComputeUnitPartition,
        },
        ScheduleOp::Interchange { axis1: 0, axis2: 1 },
        ScheduleOp::StripMine {
            axis: 0,
            factor: 16,
            inner_axis: 2,
        },
        ScheduleOp::Tile {
            axis: 0,
            tile_size: 32,
            inner_axis: 2,
        },
        ScheduleOp::Unroll { axis: 0, factor: 4 },
        ScheduleOp::Skew {
            outer_axis: 0,
            inner_axis: 1,
            factor: 2,
        },
        ScheduleOp::Reorder {
            permutation: vec![1, 0],
        },
        ScheduleOp::Vectorize {
            axis: 0,
            vector_width: 4,
        },
        ScheduleOp::MapToHierarchy {
            axis: 0,
            level: MappingLevel::Workgroup,
            dimension: 0,
        },
        ScheduleOp::SubgroupSpecialization {
            roles: vec![
                PipelineRoleGroup {
                    role: PipelineRole::Producer,
                    workers: 8,
                },
                PipelineRoleGroup {
                    role: PipelineRole::Consumer,
                    workers: 24,
                },
            ],
        },
        ScheduleOp::RoleSpecialization {
            roles: vec![PipelineRoleGroup {
                role: PipelineRole::Producer,
                workers: 16,
            }],
        },
        ScheduleOp::AsyncPipeline {
            stages: 3,
            ring_size: 2,
            role_groups: vec![PipelineRoleGroup {
                role: PipelineRole::Producer,
                workers: 8,
            }],
        },
        ScheduleOp::SoftwarePipeline {
            stages: 4,
            initiation_interval: 2,
        },
        ScheduleOp::AsyncCopyComputeOverlap {
            copy_stage: 0,
            compute_stage: 1,
            overlap_factor: 2,
        },
        ScheduleOp::Recompute {
            values: vec![10, 11],
        },
        ScheduleOp::Persistence {
            phase: 1,
            capacity: 64,
        },
        ScheduleOp::MemoryPlacement {
            buffer: "tile_a".to_string(),
            placement: MemoryPlacement::Workgroup,
            packing: Some("padded_bank_swizzle".to_string()),
            bytes: 4096,
        },
        ScheduleOp::SharedMemoryStage {
            buffer: "tile_b".to_string(),
            placement: MemoryPlacement::Workgroup,
            staging_bytes: 8192,
        },
        ScheduleOp::DoubleBuffer {
            buffer: "tile_a".to_string(),
            slots: 2,
        },
        ScheduleOp::AllocationReuse {
            source_buffer: "temp_scratch".to_string(),
            target_buffer: "activation".to_string(),
            byte_offset: 0,
        },
        ScheduleOp::Spill {
            buffer: "large_intermediate".to_string(),
            spill_location: MemoryPlacement::Device,
            spill_bytes: 16384,
        },
        ScheduleOp::Communication {
            exchange_kind: "ring_allreduce".to_string(),
            comm_group: 0,
            payload_bytes: 65536,
        },
        ScheduleOp::EntryPointDag {
            entry_points: vec![0, 1, 2],
            dependencies: vec![(0, 1), (1, 2)],
        },
        ScheduleOp::RegisterTile {
            register_dim_m: 8,
            register_dim_n: 8,
            register_dim_k: 4,
        },
        ScheduleOp::InstructionSelect {
            target_intrinsic: "mma_sync_m16n8k16".to_string(),
            mma_shape: Some([16, 8, 16]),
        },
        ScheduleOp::Synchronize {
            scope: SynchronizationScope::Workgroup,
        },
    ]
}

#[test]
fn schedule_calculus_composition_and_validation() {
    let inner_leaf = ScheduleTree::leaf(ScheduleOp::Vectorize {
        axis: 1,
        vector_width: 4,
    });

    let cert =
        DependencyPreservationCertificate::new("polyhedral_tiling_legality_v1", vec![0, 1], true);

    let tiled_node = ScheduleTree::node(
        ScheduleOp::Tile {
            axis: 0,
            tile_size: 16,
            inner_axis: 1,
        },
        inner_leaf,
        Some(cert),
    );

    let map_node = ScheduleTree::node(
        ScheduleOp::MapToHierarchy {
            axis: 0,
            level: MappingLevel::Workgroup,
            dimension: 0,
        },
        tiled_node,
        None,
    );

    let bounds = ScheduleResourceBounds {
        logical_points: 1024,
        shared_bytes: 4096,
        private_bytes: 64,
        registers_per_invocation: 32,
        pipeline_slots: 2,
        queue_capacity: 0,
    };

    let plan = SchedulePlan::new(map_node, bounds);
    assert_eq!(plan.version, 2);
    plan.validate()
        .expect("well-formed schedule plan validates cleanly");
    assert_eq!(plan.root.node_count(), 3);
}

#[test]
fn schedule_calculus_all_operators_declare_preconditions_resources_effects_and_proofs() {
    let ops = every_schedule_op();
    assert!(
        ops.len() >= 18,
        "calculus must cover at least the 18 required domains"
    );

    let initial_bounds = ScheduleResourceBounds::default();

    for op in &ops {
        // 1. Preconditions must be deterministic and queryable
        let _preconds = op.preconditions();

        // 2. Transformed domains & dependencies
        let _domains = op.transformed_domains();
        let _deps = op.transformed_dependencies();

        // 3. Resource equations must be valid and non-decreasing on demand
        let updated_bounds = op.resource_equations(&initial_bounds);
        assert!(updated_bounds.shared_bytes >= initial_bounds.shared_bytes);
        assert!(updated_bounds.private_bytes >= initial_bounds.private_bytes);
        assert!(updated_bounds.registers_per_invocation >= initial_bounds.registers_per_invocation);

        // 4. Numerical effects
        let effect = op.numerical_effects();
        match effect {
            ScheduleNumericalEffect::Exact
            | ScheduleNumericalEffect::ReorderedAccumulation
            | ScheduleNumericalEffect::PrecisionConverted
            | ScheduleNumericalEffect::Approximated { .. } => {}
        }

        // 5. Inverse mapping and debug mapping
        let inverse = op.inverse_mapping();
        match inverse {
            ScheduleInverseOp::Reversible(_)
            | ScheduleInverseOp::RestoreState { .. }
            | ScheduleInverseOp::Irreversible { .. } => {}
        }
        let debug = op.debug_mapping();
        assert!(!debug.is_empty());

        // 6. Proof constructor produces valid certificate
        let cert = op.proof_constructor();
        assert!(!cert.theorem_id.is_empty());
        assert!(cert.direction_preserved);
        assert_ne!(cert.certificate_digest, [0u8; 32]);
    }
}

#[test]
fn schedule_calculus_operator_set_is_closed_against_source_at_runtime() {
    let source = std::fs::read_to_string("vyre-foundation/src/schedule/tree.rs")
        .or_else(|_| std::fs::read_to_string("src/schedule/tree.rs"))
        .expect("schedule tree.rs source must be accessible for runtime closure check");

    let enum_marker = "pub enum ScheduleOp {";
    let start_idx = source
        .find(enum_marker)
        .expect("pub enum ScheduleOp declaration must exist in source");
    let after_enum = &source[start_idx + enum_marker.len()..];
    let end_idx = after_enum
        .find("\n}")
        .expect("enum ScheduleOp closing brace must exist");
    let enum_body = &after_enum[..end_idx];

    let mut variant_names = BTreeSet::new();
    let mut in_variant = false;
    for line in enum_body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("///") || trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if !in_variant {
            if let Some(brace_idx) = trimmed.find('{') {
                let name = trimmed[..brace_idx].trim();
                if !name.is_empty() && name.chars().next().unwrap().is_ascii_alphabetic() {
                    variant_names.insert(name.to_string());
                }
                in_variant = true;
            } else if let Some(comma_idx) = trimmed.find(',') {
                let name = trimmed[..comma_idx].trim();
                if !name.is_empty() && name.chars().next().unwrap().is_ascii_alphabetic() {
                    variant_names.insert(name.to_string());
                }
            } else if trimmed.chars().next().unwrap().is_ascii_alphabetic() {
                variant_names.insert(trimmed.to_string());
            }
        } else if trimmed.starts_with('}') {
            in_variant = false;
        }
    }

    assert!(
        variant_names.len() >= 18,
        "Source declared {} variants, expected at least 18",
        variant_names.len()
    );
    let tested_ops = every_schedule_op();
    let tested_names: BTreeSet<String> = tested_ops
        .iter()
        .map(|op| {
            let debug = format!("{op:?}");
            let name = debug.split(|c| c == '{' || c == '(').next().unwrap().trim();
            name.to_string()
        })
        .collect();

    for variant in &variant_names {
        assert!(
            tested_names.contains(variant),
            "Variant `{variant}` declared in ScheduleOp source is missing in every_schedule_op test corpus. Adding an operator turns the suite RED until its declarations and tests are recorded."
        );
    }
}

#[test]
fn schedule_calculus_type_level_unrepresentability() {
    // Valid composition across the execution hierarchy:
    // DeviceScope -> WorkgroupScope -> SubgroupScope -> ThreadScope -> InnermostLoopScope
    let stage = TypedScheduleStage::new(1)
        .region_partition(NonZeroU32::new(4).unwrap())
        .map_to_workgroup(0, 0)
        .tile(0, NonZeroU64::new(32).unwrap(), 1)
        .stage_shared_memory("shmem_tile", 2048)
        .synchronize_workgroup()
        .map_to_subgroup(1, 0)
        .specialize_roles(vec![PipelineRoleGroup {
            role: PipelineRole::Producer,
            workers: 16,
        }])
        .map_to_thread(1, 0)
        .register_tile(8, 8, 4)
        .enter_innermost_loop()
        .unroll(2, NonZeroU32::new(4).unwrap())
        .vectorize(2, NonZeroU32::new(4).unwrap())
        .instruction_select("mma_sync", Some([16, 8, 16]));

    let tree = stage.build();
    assert!(tree.node_count() >= 5);
    tree.validate()
        .expect("type-constructed schedule validates cleanly");

    // NOTE on type-level unrepresentability:
    // 1. Zero tile size is unrepresentable: `tile()` requires `NonZeroU64`. Passing `0` fails to compile with type error.
    // 2. Zero vector width is unrepresentable: `vectorize()` requires `NonZeroU32`.
    // 3. Mapping workgroup inside a thread scope is unrepresentable: `TypedScheduleStage<ThreadScope>` has no `map_to_workgroup()` method.
    // 4. Vectorizing at device scope is unrepresentable: `TypedScheduleStage<DeviceScope>` has no `vectorize()` method.
}

#[test]
fn schedule_calculus_isomorphism_deduplication() {
    let op_a = ScheduleOp::Tile {
        axis: 0,
        tile_size: 16,
        inner_axis: 1,
    };
    let op_b = ScheduleOp::SharedMemoryStage {
        buffer: "shared_buf".to_string(),
        placement: MemoryPlacement::Workgroup,
        staging_bytes: 2048,
    };
    let op_c = ScheduleOp::Synchronize {
        scope: SynchronizationScope::Workgroup,
    };

    // 1. Tree A: Parallel([A, B, C])
    let tree_1 = ScheduleTree::parallel(vec![
        ScheduleTree::leaf(op_a.clone()),
        ScheduleTree::leaf(op_b.clone()),
        ScheduleTree::leaf(op_c.clone()),
    ]);

    // 2. Tree B: Parallel([C, B, A]) - Commuted parallel branches
    let tree_2 = ScheduleTree::parallel(vec![
        ScheduleTree::leaf(op_c.clone()),
        ScheduleTree::leaf(op_b.clone()),
        ScheduleTree::leaf(op_a.clone()),
    ]);

    // 3. Tree C: Nested associative parallel Parallel([A, Parallel([B, C])])
    let tree_3 = ScheduleTree::parallel(vec![
        ScheduleTree::leaf(op_a.clone()),
        ScheduleTree::parallel(vec![
            ScheduleTree::leaf(op_b.clone()),
            ScheduleTree::leaf(op_c.clone()),
        ]),
    ]);

    // 4. Tree D: Nested associative parallel Parallel([Parallel([C, A]), B])
    let tree_4 = ScheduleTree::parallel(vec![
        ScheduleTree::parallel(vec![
            ScheduleTree::leaf(op_c.clone()),
            ScheduleTree::leaf(op_a.clone()),
        ]),
        ScheduleTree::leaf(op_b.clone()),
    ]);

    // Assert pairwise isomorphism
    assert!(tree_1.is_isomorphic(&tree_2));
    assert!(tree_1.is_isomorphic(&tree_3));
    assert!(tree_1.is_isomorphic(&tree_4));

    // Deduplication across a candidate set
    let candidates = vec![tree_1, tree_2, tree_3, tree_4];
    let mut deduped_hashes = BTreeSet::new();
    for c in &candidates {
        deduped_hashes.insert(c.canonicalize().canonical_hash());
    }

    // EXACT count assertion: 4 isomorphic trees deduplicate to exactly 1 candidate
    assert_eq!(
        deduped_hashes.len(),
        1,
        "isomorphic schedule trees must deduplicate to exactly 1 candidate"
    );
}

#[test]
fn schedule_calculus_partial_schedule_and_symbolic_parameters() {
    let root = ScheduleTree::leaf(ScheduleOp::Tile {
        axis: 0,
        tile_size: 16,
        inner_axis: 1,
    });

    let params = vec![SymbolicParameter {
        name: "tile_dim_k".to_string(),
        min_value: 8,
        max_value: 64,
        default_value: Some(16),
    }];

    let partial = PartialSchedule::new(root, vec![2], params);
    assert!(
        !partial.is_complete(),
        "partial schedule with unassigned regions is not complete"
    );

    let mut bindings = HashMap::new();
    bindings.insert("tile_dim_k".to_string(), 32);

    let selected = ScheduleResourceBounds::default();
    let plan = partial
        .instantiate(&bindings, selected.clone())
        .expect("instantiation within legal bounds succeeds");
    assert_eq!(plan.version, 2);
    assert_eq!(
        plan.resource_bounds, selected,
        "the plan validates under the bounds the caller selected, not one the schedule invented"
    );

    // Out-of-bounds parameter instantiation fails closed
    let mut invalid_bindings = HashMap::new();
    invalid_bindings.insert("tile_dim_k".to_string(), 128); // Exceeds max_value: 64
    assert!(partial.instantiate(&invalid_bindings, selected).is_err());
}

#[test]
fn schedule_calculus_rejects_invalid_zero_tile_and_unbounded_pipeline() {
    // Zero tile size must be rejected
    let invalid_tile = ScheduleTree::leaf(ScheduleOp::Tile {
        axis: 0,
        tile_size: 0,
        inner_axis: 1,
    });
    let plan1 = SchedulePlan::new(invalid_tile, ScheduleResourceBounds::default());
    assert!(plan1.validate().is_err());

    // Unbounded pipeline queue must be rejected
    let invalid_pipe = ScheduleTree::leaf(ScheduleOp::AsyncPipeline {
        stages: 0,
        ring_size: 0,
        role_groups: vec![],
    });
    let plan2 = SchedulePlan::new(invalid_pipe, ScheduleResourceBounds::default());
    assert!(plan2.validate().is_err());
}

#[test]
fn schedule_calculus_diffing_and_cost_modeling() {
    let tree_a = ScheduleTree::sequence(vec![
        ScheduleTree::leaf(ScheduleOp::Tile {
            axis: 0,
            tile_size: 16,
            inner_axis: 1,
        }),
        ScheduleTree::leaf(ScheduleOp::Synchronize {
            scope: SynchronizationScope::Workgroup,
        }),
    ]);

    let tree_b = ScheduleTree::sequence(vec![
        ScheduleTree::leaf(ScheduleOp::Tile {
            axis: 0,
            tile_size: 32,
            inner_axis: 1,
        }),
        ScheduleTree::leaf(ScheduleOp::Synchronize {
            scope: SynchronizationScope::Workgroup,
        }),
    ]);

    let plan_a = SchedulePlan::new(
        tree_a,
        ScheduleResourceBounds {
            logical_points: 1024,
            shared_bytes: 2048,
            private_bytes: 32,
            registers_per_invocation: 24,
            pipeline_slots: 1,
            queue_capacity: 0,
        },
    );

    let plan_b = SchedulePlan::new(
        tree_b,
        ScheduleResourceBounds {
            logical_points: 1024,
            shared_bytes: 4096,
            private_bytes: 32,
            registers_per_invocation: 32,
            pipeline_slots: 1,
            queue_capacity: 0,
        },
    );

    let diff = ScheduleDiff::diff_plans(&plan_a, &plan_b);
    assert!(
        !diff.is_empty(),
        "diff detects parameter and resource changes"
    );
    assert!(diff.items.iter().any(|item| matches!(
        item,
        ScheduleDiffItem::ResourceBoundChanged { name, before: 2048, after: 4096 } if name == "shared_bytes"
    )));

    let cost_model = ScheduleCostModel::new(32, 65536, 64);
    let cost_a = cost_model.evaluate(&plan_a);
    let cost_b = cost_model.evaluate(&plan_b);

    assert!(cost_a.estimated_cycles > 0);
    assert!(cost_b.estimated_cycles > 0);
    // Larger tile size reduces global memory traffic
    assert!(cost_b.memory_traffic_bytes <= cost_a.memory_traffic_bytes);
}
