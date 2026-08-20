//! The columnar fact view every hot optimizer pass queries instead of walking
//! the tree again.
//!
//! A pass asks where a name is bound, which sites touch a buffer, and whether a
//! kind occurs at all. Each answer is a column scan or an index lookup, and each
//! is only worth having if it agrees with the tree the walk read. These cases
//! hold the columns, the lazily built indices, the parent chain and the region
//! metadata to the program they were built from.
//!
//! Every item read here is public, so the suite sits outside the crate and
//! exercises the same surface a pass does.

use vyre_foundation::ir::{
    AtomicOp, BufferAccess, BufferDecl, DataType, Expr, Ident, MemoryOrdering, Node, Program,
};
use vyre_foundation::optimizer::program_soa::{
    kind_mask, BufferRefKind, NodeIndex, NodeKind, ProgramFacts,
};

fn buf(name: &str) -> BufferDecl {
    BufferDecl::storage(name, 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
}

fn program(entry: Vec<Node>) -> Program {
    Program::wrapped(vec![buf("a"), buf("b")], [1, 1, 1], entry)
}

#[test]
fn program_facts_build_exposes_fallible_reservation_path() {
    let facts = ProgramFacts::try_build(&program(vec![Node::let_bind("x", Expr::u32(1))]))
        .expect("Fix: small ProgramFacts build should reserve successfully");
    assert_eq!(facts.let_sites_of("x").len(), 1);
}

/// `build` returns an empty fact table for an entry tree that
/// has no user nodes (the wrapping Region itself counts as one
/// node and is recorded).
#[test]
fn empty_program_has_only_region_node() {
    let facts = ProgramFacts::build(&program(Vec::default()));
    assert_eq!(facts.node_count(), 1);
    assert_eq!(facts.kind_at(NodeIndex(0)), NodeKind::Region);
    assert!(facts.lets().is_empty());
    assert!(facts.var_reads().is_empty());
    assert!(facts.buffer_refs().is_empty());
}

/// Empty entry tree has the wrapping Region in `kinds_present`
/// and nothing else  -  no Lets, no Loops, no Stores.
#[test]
fn kinds_present_bitset_starts_empty_then_records_each_kind() {
    let facts = ProgramFacts::build(&program(Vec::default()));
    // Wrapping Region IS recorded by `build`.
    assert!(facts.has_kind(NodeKind::Region));
    // But nothing else is.
    assert!(!facts.has_kind(NodeKind::Let));
    assert!(!facts.has_kind(NodeKind::Loop));
    assert!(!facts.has_kind(NodeKind::Store));
    assert!(!facts.has_kind(NodeKind::If));
    assert!(!facts.has_kind(NodeKind::Barrier));
}

/// Each observed Node sets exactly its bit in `kinds_present`.
#[test]
fn kinds_present_records_every_observed_kind() {
    let facts = ProgramFacts::build(&program(vec![
        Node::let_bind("x", Expr::u32(1)),
        Node::store("a", Expr::u32(0), Expr::u32(7)),
        Node::if_then(Expr::var("x"), vec![Node::Return]),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![Node::Block(Vec::default())],
        ),
        Node::Barrier {
            ordering: MemoryOrdering::SeqCst,
        },
    ]));
    assert!(facts.has_kind(NodeKind::Let));
    assert!(facts.has_kind(NodeKind::Store));
    assert!(facts.has_kind(NodeKind::If));
    assert!(facts.has_kind(NodeKind::Return));
    assert!(facts.has_kind(NodeKind::Loop));
    assert!(facts.has_kind(NodeKind::Block));
    assert!(facts.has_kind(NodeKind::Barrier));
    assert!(facts.has_kind(NodeKind::Region));
    // Kinds we never produced must remain false.
    assert!(!facts.has_kind(NodeKind::Assign));
    assert!(!facts.has_kind(NodeKind::AsyncLoad));
    assert!(!facts.has_kind(NodeKind::AsyncStore));
    assert!(!facts.has_kind(NodeKind::IndirectDispatch));
    assert!(!facts.has_kind(NodeKind::Trap));
}

/// `has_any_kind_in_mask` ORs across the kinds_present bitset:
/// a program with a Let alone matches a (Let | Loop) mask and
/// not a (Loop | Trap) mask.
#[test]
fn has_any_kind_in_mask_is_or_across_observed_kinds() {
    let facts = ProgramFacts::build(&program(vec![Node::let_bind("x", Expr::u32(1))]));
    assert!(facts.has_any_kind_in_mask(kind_mask(NodeKind::Let)));
    assert!(facts.has_any_kind_in_mask(kind_mask(NodeKind::Let) | kind_mask(NodeKind::Loop)));
    assert!(!facts.has_any_kind_in_mask(kind_mask(NodeKind::Loop) | kind_mask(NodeKind::Trap)));
    assert!(facts.has_kind(NodeKind::Let));
    assert!(!facts.has_kind(NodeKind::Loop));
}

/// `kinds_present()` mask exposes the raw bitset for callers that
/// want to short-circuit on multiple kinds with a single AND.
#[test]
fn kinds_present_mask_round_trips_through_kind_mask_helper() {
    let facts = ProgramFacts::build(&program(vec![
        Node::let_bind("x", Expr::u32(1)),
        Node::Return,
    ]));
    let mask = facts.kinds_present();
    // Exactly the bits we expect: Let, Return, Region (the
    // wrapping Region is always recorded).
    let expected =
        kind_mask(NodeKind::Let) | kind_mask(NodeKind::Return) | kind_mask(NodeKind::Region);
    assert_eq!(mask, expected);
}

/// Lets are recorded in preorder with the right name.
#[test]
fn let_sites_recorded_in_preorder() {
    let facts = ProgramFacts::build(&program(vec![
        Node::let_bind("x", Expr::u32(1)),
        Node::let_bind("y", Expr::u32(2)),
    ]));
    let lets = facts.lets();
    assert_eq!(lets.len(), 2);
    assert_eq!(lets[0].1.as_str(), "x");
    assert_eq!(lets[1].1.as_str(), "y");
}

/// Var reads and buffer touches are observed across nesting.
#[test]
fn nested_if_collects_var_reads_and_buffer_refs() {
    let facts = ProgramFacts::build(&program(vec![
        Node::let_bind("x", Expr::u32(7)),
        Node::If {
            cond: Expr::var("c"),
            then: vec![Node::store("a", Expr::var("x"), Expr::u32(1))],
            otherwise: vec![Node::store("b", Expr::var("x"), Expr::u32(2))],
        },
    ]));
    let var_reads: Vec<&str> = facts.var_reads().iter().map(|(_, n)| n.as_str()).collect();
    assert!(var_reads.contains(&"c"));
    let x_count = var_reads.iter().filter(|n| **n == "x").count();
    assert_eq!(x_count, 2, "x read in both arms");
    let a_writes: Vec<_> = facts
        .buffer_refs_of("a")
        .iter()
        .filter(|(_, k)| *k == BufferRefKind::Write)
        .collect();
    assert_eq!(a_writes.len(), 1);
    let b_writes: Vec<_> = facts
        .buffer_refs_of("b")
        .iter()
        .filter(|(_, k)| *k == BufferRefKind::Write)
        .collect();
    assert_eq!(b_writes.len(), 1);
}

/// `let_sites_of` returns every Let-site for a name; lookup
/// indices are built lazily and reused.
#[test]
fn let_sites_of_resolves_via_lookup_index() {
    let facts = ProgramFacts::build(&program(vec![
        Node::let_bind("x", Expr::u32(1)),
        Node::Block(vec![Node::let_bind("x", Expr::u32(2))]),
    ]));
    let sites = facts.let_sites_of("x");
    assert_eq!(sites.len(), 2, "both Let-sites of `x` are recorded");
    assert!(facts.let_sites_of("missing").is_empty());
}

#[test]
fn descendant_query_uses_parent_column() {
    let facts = ProgramFacts::build(&program(vec![Node::Block(vec![Node::let_bind(
        "x",
        Expr::u32(1),
    )])]));
    let root = facts.regions()[0].node;
    let let_idx = facts.lets()[0].0;
    assert!(facts.is_descendant_of(root, root));
    assert!(facts.is_descendant_of(let_idx, root));
    assert!(!facts.is_descendant_of(root, let_idx));
}

/// Atomic touches are recorded with the AtomicOp.
#[test]
fn atomic_buffer_refs_record_op() {
    let facts = ProgramFacts::build(&program(vec![Node::let_bind(
        "x",
        Expr::Atomic {
            op: AtomicOp::Add,
            buffer: Ident::from("a"),
            index: Box::new(Expr::u32(0)),
            expected: None,
            value: Box::new(Expr::u32(1)),
            ordering: MemoryOrdering::Relaxed,
        },
    )]));
    let touches = facts.buffer_refs_of("a");
    assert_eq!(touches.len(), 1);
    assert_eq!(touches[0].1, BufferRefKind::Atomic(AtomicOp::Add));
}

/// `is_name_rebound` distinguishes single Let, multi Let, Assign,
/// and Loop-var rebinding.
#[test]
fn is_name_rebound_detects_every_shape() {
    let facts_single = ProgramFacts::build(&program(vec![Node::let_bind("x", Expr::u32(1))]));
    assert!(!facts_single.is_name_rebound("x"));
    assert!(!facts_single.is_name_rebound("y"));

    let facts_assign = ProgramFacts::build(&program(vec![
        Node::let_bind("x", Expr::u32(1)),
        Node::Assign {
            name: Ident::from("x"),
            value: Expr::u32(2),
        },
    ]));
    assert!(facts_assign.is_name_rebound("x"));

    let facts_loop = ProgramFacts::build(&program(vec![Node::Loop {
        var: Ident::from("i"),
        from: Expr::u32(0),
        to: Expr::u32(4),
        body: vec![],
    }]));
    assert!(facts_loop.is_name_rebound("i"));

    let facts_double_let = ProgramFacts::build(&program(vec![
        Node::let_bind("x", Expr::u32(1)),
        Node::Block(vec![Node::let_bind("x", Expr::u32(2))]),
    ]));
    assert!(facts_double_let.is_name_rebound("x"));
}

/// Every Loop-var binding is recorded in `loop_vars`.
#[test]
fn loop_vars_recorded_for_every_loop() {
    let facts = ProgramFacts::build(&program(vec![Node::Loop {
        var: Ident::from("i"),
        from: Expr::u32(0),
        to: Expr::u32(4),
        body: vec![Node::Loop {
            var: Ident::from("j"),
            from: Expr::u32(0),
            to: Expr::u32(4),
            body: vec![],
        }],
    }]));
    let names: Vec<&str> = facts.loop_vars().iter().map(|(_, n)| n.as_str()).collect();
    assert_eq!(names, vec!["i", "j"]);
}

/// `parent_of` reports the enclosing container for nested
/// nodes; the root entry's wrapping Region has no parent.
#[test]
fn parent_of_reports_immediate_container() {
    let facts = ProgramFacts::build(&program(vec![Node::If {
        cond: Expr::var("c"),
        then: vec![Node::let_bind("x", Expr::u32(1))],
        otherwise: vec![Node::let_bind("y", Expr::u32(2))],
    }]));
    let region = NodeIndex(0);
    assert_eq!(facts.kind_at(region), NodeKind::Region);
    assert_eq!(facts.parent_of(region), None);
    let if_idx = facts
        .iter_nodes()
        .find(|(_, k)| *k == NodeKind::If)
        .map(|(i, _)| i)
        .expect("Fix: If node present");
    assert_eq!(facts.parent_of(if_idx), Some(region));
    let let_idxs: Vec<_> = facts.lets().iter().map(|(i, _)| *i).collect();
    for let_idx in let_idxs {
        assert_eq!(facts.parent_of(let_idx), Some(if_idx));
    }
}

/// `buffer_refs_of` reports the Write site of a Store, the
/// Read site of a load inside its value, and distinguishes the
/// two by `BufferRefKind`.
#[test]
fn buffer_refs_of_separates_read_and_write() {
    let facts = ProgramFacts::build(&program(vec![Node::store(
        "a",
        Expr::u32(0),
        Expr::Load {
            buffer: Ident::from("b"),
            index: Box::new(Expr::u32(0)),
        },
    )]));
    let a_touches = facts.buffer_refs_of("a");
    assert_eq!(a_touches.len(), 1);
    assert_eq!(a_touches[0].1, BufferRefKind::Write);
    let b_touches = facts.buffer_refs_of("b");
    assert_eq!(b_touches.len(), 1);
    assert_eq!(b_touches[0].1, BufferRefKind::Read);
}

/// `has_kind` short-circuits passes that have no candidate
/// nodes.
#[test]
fn has_kind_short_circuits_missing_variants() {
    let facts = ProgramFacts::build(&program(vec![Node::let_bind("x", Expr::u32(1))]));
    assert!(facts.has_kind(NodeKind::Let));
    assert!(!facts.has_kind(NodeKind::Loop));
    assert!(!facts.has_kind(NodeKind::Trap));
}

/// `iter_nodes` yields every node in preorder with its kind.
#[test]
fn iter_nodes_yields_preorder() {
    let facts = ProgramFacts::build(&program(vec![
        Node::let_bind("x", Expr::u32(1)),
        Node::let_bind("y", Expr::u32(2)),
    ]));
    let kinds: Vec<NodeKind> = facts.iter_nodes().map(|(_, k)| k).collect();
    assert_eq!(kinds, vec![NodeKind::Region, NodeKind::Let, NodeKind::Let]);
}

// ──── region/source metadata side-table ────

/// `regions()` records every Region in the entry tree, including
/// the wrapping Region that `Program::wrapped` injects when the
/// entry contains non-Region top-level nodes.
#[test]
fn regions_records_wrapping_and_nested() {
    // Mixing a non-Region top-level node forces `Program::wrapped`
    // to inject the root Region, so the fact table sees exactly
    // two regions: the wrapper plus the explicit inner one.
    let inner = Node::Region {
        generator: Ident::from("inner_pass"),
        source_region: None,
        body: std::sync::Arc::new(vec![Node::let_bind("x", Expr::u32(1))]),
    };
    let facts = ProgramFacts::build(&program(vec![Node::let_bind("z", Expr::u32(0)), inner]));
    let regions = facts.regions();
    assert_eq!(regions.len(), 2, "wrapping Region + inner Region");
    assert!(regions.iter().any(|r| r.generator.as_str() == "inner_pass"));
}

/// `region_at(idx)` looks up the Region metadata for a Region
/// node by its `NodeIndex`. Returns None for non-Region nodes.
#[test]
fn region_at_resolves_by_node_index() {
    let inner = Node::Region {
        generator: Ident::from("custom"),
        source_region: None,
        body: std::sync::Arc::new(vec![]),
    };
    let facts = ProgramFacts::build(&program(vec![inner]));
    let region_idx = facts
        .iter_nodes()
        .filter(|(_, k)| *k == NodeKind::Region)
        .map(|(i, _)| i)
        .find(|i| {
            facts
                .region_at(*i)
                .map(|m| m.generator.as_str() == "custom")
                .unwrap_or(false)
        })
        .expect("Fix: custom-generator Region present");
    let meta = facts.region_at(region_idx).expect("Fix: region recorded");
    assert_eq!(meta.generator.as_str(), "custom");
    assert_eq!(meta.source_region, None);
    let let_idx = facts.lets().first().map(|(i, _)| *i);
    if let Some(let_idx) = let_idx {
        assert!(facts.region_at(let_idx).is_none());
    }
}

/// `regions_by_generator(name)` returns every Region whose
/// generator matches.
#[test]
fn regions_by_generator_filters_by_ident() {
    let entry = vec![
        Node::Region {
            generator: Ident::from("vec"),
            source_region: None,
            body: std::sync::Arc::new(vec![Node::let_bind("x", Expr::u32(1))]),
        },
        Node::Region {
            generator: Ident::from("dce"),
            source_region: None,
            body: std::sync::Arc::new(vec![Node::let_bind("y", Expr::u32(2))]),
        },
        Node::Region {
            generator: Ident::from("vec"),
            source_region: None,
            body: std::sync::Arc::new(vec![Node::let_bind("z", Expr::u32(3))]),
        },
    ];
    let facts = ProgramFacts::build(&program(entry));
    let vec_count = facts.regions_by_generator("vec").count();
    assert_eq!(vec_count, 2);
    let dce_count = facts.regions_by_generator("dce").count();
    assert_eq!(dce_count, 1);
    let missing = facts.regions_by_generator("missing").count();
    assert_eq!(missing, 0);
}

/// Region wrappers are provenance rows, not semantic optimizer
/// nodes. The regionless view keeps preorder for real work while
/// skipping both the wrapper inserted by Program::wrapped and any
/// explicit nested Region.
#[test]
fn regionless_nodes_skip_provenance_wrappers() {
    let facts = ProgramFacts::build(&program(vec![
        Node::let_bind("root", Expr::u32(0)),
        Node::Region {
            generator: Ident::from("inner"),
            source_region: None,
            body: std::sync::Arc::new(vec![Node::let_bind("nested", Expr::u32(1))]),
        },
    ]));
    let kinds: Vec<NodeKind> = facts
        .iter_regionless_nodes()
        .map(|(_, kind)| kind)
        .collect();
    assert_eq!(kinds, vec![NodeKind::Let, NodeKind::Let]);
}

/// `regionless_parent_of` skips Region ancestors but preserves
/// real structural parents such as Block. Optimizer passes can
/// use this for scope queries without treating provenance
/// wrappers as part of the compute tree.
#[test]
fn regionless_parent_skips_only_region_ancestors() {
    let facts = ProgramFacts::build(&program(vec![Node::Block(vec![Node::Region {
        generator: Ident::from("inner"),
        source_region: None,
        body: std::sync::Arc::new(vec![Node::let_bind("x", Expr::u32(1))]),
    }])]));
    let block = facts
        .iter_nodes()
        .find(|(_, kind)| *kind == NodeKind::Block)
        .map(|(idx, _)| idx)
        .expect("Fix: Block node present");
    let let_idx = facts.lets()[0].0;
    assert_eq!(facts.regionless_parent_of(block), None);
    assert_eq!(facts.regionless_parent_of(let_idx), Some(block));
}

// ──── points-to facts (buffers_provably_distinct) ────

/// Two distinct named buffers in the program both touched at
/// least once → provably distinct.
#[test]
fn buffers_provably_distinct_for_distinct_names() {
    let facts = ProgramFacts::build(&program(vec![
        Node::store("a", Expr::u32(0), Expr::u32(1)),
        Node::store("b", Expr::u32(0), Expr::u32(2)),
    ]));
    assert!(facts.buffers_provably_distinct("a", "b"));
    assert!(facts.buffers_provably_distinct("b", "a"));
}

/// A buffer trivially aliases itself.
#[test]
fn buffers_provably_distinct_rejects_same_name() {
    let facts = ProgramFacts::build(&program(vec![Node::store("a", Expr::u32(0), Expr::u32(1))]));
    assert!(!facts.buffers_provably_distinct("a", "a"));
}

/// A name that doesn't appear in the buffer_refs column is not
/// a real buffer  -  the fact returns false to keep the contract
/// honest.
#[test]
fn buffers_provably_distinct_rejects_phantom_name() {
    let facts = ProgramFacts::build(&program(vec![Node::store("a", Expr::u32(0), Expr::u32(1))]));
    assert!(!facts.buffers_provably_distinct("a", "phantom"));
}

// ──── escape facts (buffer_escapes) ────

/// A buffer that's only read (Load) does NOT escape  -  its
/// contents are an input the host produced.
#[test]
fn buffer_does_not_escape_when_read_only() {
    let facts = ProgramFacts::build(&program(vec![Node::let_bind(
        "x",
        Expr::Load {
            buffer: Ident::from("a"),
            index: Box::new(Expr::u32(0)),
        },
    )]));
    assert!(!facts.buffer_escapes("a"));
}

/// A buffer that's stored to escapes (host reads back).
#[test]
fn buffer_escapes_when_stored_to() {
    let facts = ProgramFacts::build(&program(vec![Node::store("a", Expr::u32(0), Expr::u32(1))]));
    assert!(facts.buffer_escapes("a"));
}

/// A buffer touched atomically escapes (atomic results are
/// observable across workgroups + the host).
#[test]
fn buffer_escapes_when_atomically_touched() {
    let facts = ProgramFacts::build(&program(vec![Node::let_bind(
        "x",
        Expr::Atomic {
            op: AtomicOp::Add,
            buffer: Ident::from("a"),
            index: Box::new(Expr::u32(0)),
            expected: None,
            value: Box::new(Expr::u32(1)),
            ordering: MemoryOrdering::Relaxed,
        },
    )]));
    assert!(facts.buffer_escapes("a"));
}

/// `escaping_buffers()` enumerates the set in one go.
#[test]
fn escaping_buffers_enumerates_set() {
    let facts = ProgramFacts::build(&program(vec![
        Node::store("a", Expr::u32(0), Expr::u32(1)),
        Node::let_bind(
            "x",
            Expr::Load {
                buffer: Ident::from("b"),
                index: Box::new(Expr::u32(0)),
            },
        ),
    ]));
    let escaping = facts.escaping_buffers();
    assert_eq!(escaping.len(), 1);
    assert!(escaping.iter().any(|k| k.as_str() == "a"));
}
