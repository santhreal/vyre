use super::*;
use crate::ir::{BufferDecl, DataType, Expr, Node, Program};

#[test]
fn derive_use_counts_simple() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("x", Expr::u32(1)),
            Node::let_bind("y", Expr::add(Expr::var("x"), Expr::var("x"))),
            Node::store("out", Expr::u32(0), Expr::var("y")),
        ],
    );
    let cache = FactCache::derive(&program);
    assert_eq!(cache.use_count_of(&Ident::from("x")), 2);
    assert_eq!(cache.use_count_of(&Ident::from("y")), 1);
    assert_eq!(cache.use_count_of(&Ident::from("z")), 0);
}

#[test]
fn derive_use_counts_async_operands() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(4),
            BufferDecl::read_write("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("offset", Expr::u32(1)),
            Node::let_bind("size", Expr::u32(2)),
            Node::async_load_gpu_driven(
                Ident::from("input"),
                Ident::from("out"),
                Expr::var("offset"),
                Expr::var("size"),
                Ident::from("copy"),
            ),
        ],
    );
    let cache = FactCache::derive(&program);
    assert_eq!(cache.use_count_of(&Ident::from("offset")), 1);
    assert_eq!(cache.use_count_of(&Ident::from("size")), 1);
}

#[test]
fn derive_use_facts_records_buffer_accesses_and_index_axes() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(64),
            BufferDecl::read_write("out", 1, DataType::U32).with_count(64),
        ],
        [8, 8, 1],
        vec![Node::store(
            "out",
            Expr::gid_y(),
            Expr::load("input", Expr::gid_x()),
        )],
    );

    let cache = FactCache::derive_use_only(&program);
    assert!(cache.has_fresh_use_facts_for(&program));
    assert!(!cache.is_fresh_for(&program));
    let facts = cache.use_facts().unwrap();
    assert_eq!(facts.buffer_reads.get(&Ident::from("input")), Some(&1));
    assert_eq!(facts.buffer_writes.get(&Ident::from("out")), Some(&1));
    assert_eq!(facts.dominant_index_axis(&Ident::from("input")), Some(0));
    assert_eq!(facts.dominant_index_axis(&Ident::from("out")), Some(1));
}

#[test]
fn derive_use_facts_records_scalar_mediated_buffer_dependencies() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(1),
            BufferDecl::read_write("scratch", 1, DataType::U32).with_count(1),
            BufferDecl::output("out", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("x", Expr::load("input", Expr::u32(0))),
            Node::store("scratch", Expr::u32(0), Expr::var("x")),
            Node::store("out", Expr::u32(0), Expr::load("scratch", Expr::u32(0))),
        ],
    );

    let cache = FactCache::derive_use_only(&program);
    let facts = cache.use_facts().unwrap();
    assert!(facts
        .var_buffer_deps
        .get(&Ident::from("x"))
        .is_some_and(|deps| deps.contains(&Ident::from("input"))));
    assert!(facts
        .buffer_write_deps
        .get(&Ident::from("scratch"))
        .is_some_and(|deps| deps.contains(&Ident::from("input"))));
    assert!(facts
        .buffer_write_deps
        .get(&Ident::from("out"))
        .is_some_and(|deps| deps.contains(&Ident::from("scratch"))));
}

#[test]
fn derive_use_facts_records_indirect_dispatch_count_buffers() {
    let program = Program::wrapped(
        vec![BufferDecl::read("counts", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::indirect_dispatch("counts", 0)],
    );

    let cache = FactCache::derive_use_only(&program);
    let facts = cache.use_facts().unwrap();
    assert!(facts
        .indirect_dispatch_buffers
        .contains(&Ident::from("counts")));
    assert_eq!(facts.buffer_reads.get(&Ident::from("counts")), Some(&1));
}

#[test]
fn derive_type_facts_float_propagation() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("a", Expr::f32(1.0)),
            Node::let_bind("b", Expr::add(Expr::var("a"), Expr::f32(2.0))),
        ],
    );
    let cache = FactCache::derive(&program);
    let types = cache.type_map.as_ref().unwrap();
    assert_eq!(types.var_types.get(&Ident::from("a")), Some(&DataType::F32));
    assert_eq!(types.var_types.get(&Ident::from("b")), Some(&DataType::F32));
}

#[test]
fn derive_type_facts_records_loads_and_expression_types() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::F32).with_count(1),
            BufferDecl::read_write("out", 1, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("x", Expr::load("input", Expr::u32(0))),
            Node::store("out", Expr::u32(0), Expr::var("x")),
        ],
    );

    let cache = FactCache::derive(&program);
    let types = cache.type_map.as_ref().unwrap();
    assert_eq!(types.var_types.get(&Ident::from("x")), Some(&DataType::F32));
    assert!(
        !types.expr_types.is_empty(),
        "FactCache::TypeFacts promises expression type facts; derive() must populate them"
    );
}

#[test]
fn derive_type_facts_loop_induction_binding_and_restoration() {
    // Tests:
    // 1. Induction variable is typed as U32 inside loop body.
    // 2. Expressions depending on induction variable are resolved as U32.
    // 3. Shadowed outer variable type is restored after loop completes.
    // 4. Unshadowed induction variable is removed after loop completes.
    // 5. Nested loops bind both induction variables in inner body.
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("shadowed_var", Expr::f32(1.0)),
            Node::Loop {
                var: Ident::from("shadowed_var"),
                from: Expr::u32(0),
                to: Expr::u32(4),
                body: vec![
                    Node::let_bind(
                        "inner_dep",
                        Expr::add(Expr::var("shadowed_var"), Expr::u32(1)),
                    ),
                    Node::Loop {
                        var: Ident::from("nested_var"),
                        from: Expr::u32(0),
                        to: Expr::u32(2),
                        body: vec![Node::let_bind(
                            "nested_dep",
                            Expr::add(Expr::var("shadowed_var"), Expr::var("nested_var")),
                        )],
                    },
                ],
            },
            Node::let_bind(
                "post_loop_dep",
                Expr::add(Expr::var("shadowed_var"), Expr::f32(2.0)),
            ),
            Node::Loop {
                var: Ident::from("unshadowed_var"),
                from: Expr::u32(0),
                to: Expr::u32(3),
                body: vec![Node::let_bind(
                    "unshadowed_dep",
                    Expr::mul(Expr::var("unshadowed_var"), Expr::u32(2)),
                )],
            },
        ],
    );

    let cache = FactCache::derive(&program);
    let types = cache.type_map.as_ref().unwrap();

    // Body bindings depending on loop variables were successfully typed as U32
    assert_eq!(
        types.var_types.get(&Ident::from("inner_dep")),
        Some(&DataType::U32),
        "expression depending on induction variable must be inferred as U32"
    );
    assert_eq!(
        types.var_types.get(&Ident::from("nested_dep")),
        Some(&DataType::U32),
        "expression depending on nested induction variables must be inferred as U32"
    );
    assert_eq!(
        types.var_types.get(&Ident::from("unshadowed_dep")),
        Some(&DataType::U32),
        "expression depending on unshadowed loop variable must be inferred as U32"
    );

    // Outer shadowed variable was restored to F32, so post-loop expression is typed as F32
    assert_eq!(
        types.var_types.get(&Ident::from("shadowed_var")),
        Some(&DataType::F32),
        "outer variable type must be restored after loop exit"
    );
    assert_eq!(
        types.var_types.get(&Ident::from("post_loop_dep")),
        Some(&DataType::F32),
        "post-loop expression using restored variable must be inferred as F32"
    );

    // Unshadowed loop variable is not present in final var_types
    assert_eq!(
        types.var_types.get(&Ident::from("unshadowed_var")),
        None,
        "unshadowed loop variable must be removed after loop exit"
    );
    assert_eq!(
        types.var_types.get(&Ident::from("nested_var")),
        None,
        "nested loop variable must be removed after loop exit"
    );
}

#[test]
fn derive_type_facts_assign_unknown_evicts_stale_facts() {
    // Tests:
    // 1. Assign with unknown type removes stale var_types entry.
    // 2. Assign with valid type updates var_types.
    // 3. Assign with different type updates var_types to new type.
    // 4. Subsequent uses of evicted variable cannot resolve stale type.
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("a", Expr::f32(1.0)),
            Node::Assign {
                name: Ident::from("a"),
                value: Expr::var("unbound_unknown_identifier"),
            },
            Node::let_bind("b", Expr::f32(1.0)),
            Node::Assign {
                name: Ident::from("b"),
                value: Expr::f32(2.5),
            },
            Node::let_bind("c", Expr::f32(1.0)),
            Node::Assign {
                name: Ident::from("c"),
                value: Expr::u32(42),
            },
        ],
    );

    let cache = FactCache::derive(&program);
    let types = cache.type_map.as_ref().unwrap();

    assert_eq!(
        types.var_types.get(&Ident::from("a")),
        None,
        "reassignment to unknown expression must evict stale type fact"
    );
    assert_eq!(
        types.var_types.get(&Ident::from("b")),
        Some(&DataType::F32),
        "reassignment to matching type must update type fact"
    );
    assert_eq!(
        types.var_types.get(&Ident::from("c")),
        Some(&DataType::U32),
        "reassignment to different valid type must update type fact to new type"
    );
}

#[test]
fn invalidate_clears_all() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let mut cache = FactCache::derive(&program);
    assert!(cache.is_fresh_for(&program));
    cache.invalidate();
    assert!(!cache.is_fresh_for(&program));
    assert!(cache.shape.is_none());
}

#[test]
fn typed_partitions_report_freshness_by_backing_fact_family() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );

    let full = FactCache::derive(&program);
    let full_partitions = full.fresh_partitions_for(&program);
    for partition in FactPartition::ALL {
        assert!(
            full_partitions.contains(&partition),
            "derive() must populate typed fact partition {partition:?}"
        );
    }

    let shape_use = FactCache::derive_shape_and_use(&program);
    assert!(shape_use.has_fresh_partition_for(&program, FactPartition::Graph));
    assert!(shape_use.has_fresh_partition_for(&program, FactPartition::Shape));
    assert!(shape_use.has_fresh_partition_for(&program, FactPartition::Use));
    assert!(shape_use.has_fresh_partition_for(&program, FactPartition::Effects));
    assert!(shape_use.has_fresh_partition_for(&program, FactPartition::DataflowFrontier));
    assert!(!shape_use.has_fresh_partition_for(&program, FactPartition::Type));

    let use_only = FactCache::derive_use_only(&program);
    assert!(use_only.has_fresh_partition_for(&program, FactPartition::Use));
    assert!(use_only.has_fresh_partition_for(&program, FactPartition::Effects));
    assert!(use_only.has_fresh_partition_for(&program, FactPartition::DataflowFrontier));
    assert!(!use_only.has_fresh_partition_for(&program, FactPartition::Graph));
    assert!(!use_only.has_fresh_partition_for(&program, FactPartition::Shape));
    assert!(!use_only.has_fresh_partition_for(&program, FactPartition::Type));
}

#[test]
fn typed_partition_invalidation_preserves_unrelated_fact_families() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let mut cache = FactCache::derive(&program);

    cache.invalidate_partition(FactPartition::Shape);
    assert!(!cache.has_partition(FactPartition::Shape));
    assert!(!cache.has_partition(FactPartition::Graph));
    assert!(cache.has_partition(FactPartition::Use));
    assert!(cache.has_partition(FactPartition::Type));

    cache.invalidate_partition(FactPartition::DataflowFrontier);
    assert!(!cache.has_partition(FactPartition::Use));
    assert!(!cache.has_partition(FactPartition::Effects));
    assert!(!cache.has_partition(FactPartition::DataflowFrontier));
    assert!(cache.has_partition(FactPartition::Type));
}

#[test]
fn derive_use_counts_handles_large_blocks_in_one_pass() {
    let block = Node::block(
        (0..4096)
            .map(|index| Node::let_bind(format!("sink_{index}"), Expr::var("x")))
            .collect(),
    );
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::let_bind("x", Expr::u32(1)), block],
    );
    let cache = FactCache::derive(&program);
    assert_eq!(cache.use_count_of(&Ident::from("x")), 4096);
}

#[test]
fn derive_cached_returns_equivalent_facts() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("x", Expr::u32(1)),
            Node::store("out", Expr::u32(0), Expr::var("x")),
        ],
    );
    let direct = FactCache::derive(&program);
    let cached = FactCache::derive_cached(&program);
    let cached_again = FactCache::derive_cached(&program);
    assert_eq!(direct.use_count_of(&Ident::from("x")), 1);
    assert_eq!(cached.use_count_of(&Ident::from("x")), 1);
    assert_eq!(cached_again.use_count_of(&Ident::from("x")), 1);
    let direct_use_facts = direct
        .use_facts()
        .expect("Fix: derive must populate use_facts");
    let cached_use_facts = cached
        .use_facts()
        .expect("Fix: derive_cached must populate use_facts");
    assert_eq!(
        direct_use_facts.buffer_writes,
        cached_use_facts.buffer_writes
    );
}

#[test]
fn derive_use_only_cached_returns_equivalent_facts() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("a", Expr::u32(7)),
            Node::let_bind("b", Expr::add(Expr::var("a"), Expr::var("a"))),
            Node::store("out", Expr::u32(0), Expr::var("b")),
        ],
    );
    let direct = FactCache::derive_use_only(&program);
    let cached = FactCache::derive_use_only_cached(&program);
    let cached_again = FactCache::derive_use_only_cached(&program);
    for (s, label) in [
        (&direct, "direct"),
        (&cached, "cached"),
        (&cached_again, "cached_again"),
    ] {
        assert_eq!(s.use_count_of(&Ident::from("a")), 2, "{label}");
        assert_eq!(s.use_count_of(&Ident::from("b")), 1, "{label}");
    }
}

#[test]
fn derive_shape_and_use_cached_keys_on_program_fingerprint() {
    let program_a = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let program_b = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(8)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let cached_a = FactCache::derive_shape_and_use_cached(&program_a);
    let cached_b = FactCache::derive_shape_and_use_cached(&program_b);
    let cached_a_again = FactCache::derive_shape_and_use_cached(&program_a);
    assert_eq!(cached_a.fingerprint, program_a.fingerprint());
    assert_eq!(cached_b.fingerprint, program_b.fingerprint());
    assert_eq!(cached_a_again.fingerprint, program_a.fingerprint());
    assert_ne!(cached_a.fingerprint, cached_b.fingerprint);
}
