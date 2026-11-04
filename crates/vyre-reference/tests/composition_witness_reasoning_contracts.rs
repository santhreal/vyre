//! Independent known-answer integration contracts for reasoning witnesses.

use vyre_reference::composition_witness::{
    adjoint_pair_witness, adjustment_set_ordering_is_safe_witness,
    adjustment_set_pass_descendants_witness, compile_dnnf_witness, dnnf_is_satisfiable_witness,
    dnnf_is_tautology_witness, dnnf_model_count_witness, kan_extension_at_witness,
    kan_extension_table_witness, natural_transformation_count_witness, yoneda_embedding_witness,
    zx_color_change_witness, zx_identity_removal_witness, zx_simplified_diagram_witness,
    zx_spider_fusion_witness, AdjointPair, DnnfGate, FiniteCategory, FiniteFunctor, KanDirection,
    ZxColor, ZxDiagram, ZxSpider,
};

#[test]
fn dnnf_compile_and_count_empty_formula() {
    let dag = compile_dnnf_witness(&[], 0, 4);
    assert_eq!(dag.gates.last(), Some(&DnnfGate::True));
    assert_eq!(dnnf_model_count_witness(&dag), 1);
    assert!(dnnf_is_satisfiable_witness(&dag));
    assert!(dnnf_is_tautology_witness(&dag, 0));
}

#[test]
fn dnnf_single_literal_and_contradiction() {
    let pos = compile_dnnf_witness(&[vec![(0, true)]], 1, 4);
    assert_eq!(dnnf_model_count_witness(&pos), 1);
    assert!(dnnf_is_satisfiable_witness(&pos));
    assert!(!dnnf_is_tautology_witness(&pos, 1));

    let contra = compile_dnnf_witness(&[vec![(0, true)], vec![(0, false)]], 1, 4);
    assert_eq!(dnnf_model_count_witness(&contra), 0);
    assert!(!dnnf_is_satisfiable_witness(&contra));
    assert!(!dnnf_is_tautology_witness(&contra, 1));
}

#[test]
fn dnnf_or_clause_and_tautology() {
    // (x0 ∨ x1) over 2 vars -> 3 models
    let dag = compile_dnnf_witness(&[vec![(0, true), (1, true)]], 2, 4);
    assert_eq!(dnnf_model_count_witness(&dag), 3);
    assert!(dnnf_is_satisfiable_witness(&dag));
    assert!(!dnnf_is_tautology_witness(&dag, 2));

    // Free variable smoothing: (x0) over 2 vars -> 2 models: (x0=1, x1=0) and (x0=1, x1=1)
    let smoothed = compile_dnnf_witness(&[vec![(0, true)]], 2, 4);
    assert_eq!(dnnf_model_count_witness(&smoothed), 2);
}

#[test]
fn dnnf_bounded_recursion_depth() {
    let clauses = vec![
        vec![(0, true), (1, true)],
        vec![(2, true), (3, true)],
        vec![(4, true), (5, true)],
    ];
    let dag = compile_dnnf_witness(&clauses, 6, 2);
    assert_eq!(dag.num_vars, 6);
    assert!(!dag.gates.is_empty());
}

#[test]
fn finite_category_discrete_and_adjoint() {
    let cat = FiniteCategory::discrete(3);
    assert_eq!(cat.hom(0, 0), 1);
    assert_eq!(cat.hom(0, 1), 0);
    assert_eq!(cat.hom(10, 0), 0);

    let id = FiniteFunctor::identity(3);
    let adj = adjoint_pair_witness(&cat, &cat, &id, &id);
    assert_eq!(
        adj,
        AdjointPair {
            is_adjoint: true,
            witness: None
        }
    );

    let cat2 = FiniteCategory::discrete(2);
    let f = FiniteFunctor {
        object_map: vec![0, 0],
    };
    let g = FiniteFunctor::identity(2);
    let adj_fail = adjoint_pair_witness(&cat2, &cat2, &f, &g);
    assert_eq!(
        adj_fail,
        AdjointPair {
            is_adjoint: false,
            witness: Some((1, 0))
        }
    );
}

#[test]
fn kan_extension_left_and_right() {
    let k = FiniteFunctor {
        object_map: vec![0, 0, 1],
    };
    let f = vec![3, 5, 7];

    // Left Kan (sum): at 0 -> 3 + 5 = 8; at 1 -> 7
    assert_eq!(kan_extension_at_witness(KanDirection::Left, &k, &f, 0), 8);
    assert_eq!(kan_extension_at_witness(KanDirection::Left, &k, &f, 1), 7);

    // Right Kan (product): at 0 -> 3 * 5 = 15; at 1 -> 7
    assert_eq!(kan_extension_at_witness(KanDirection::Right, &k, &f, 0), 15);
    assert_eq!(kan_extension_at_witness(KanDirection::Right, &k, &f, 1), 7);

    // Empty preimage
    assert_eq!(kan_extension_at_witness(KanDirection::Left, &k, &f, 2), 0);
    assert_eq!(kan_extension_at_witness(KanDirection::Right, &k, &f, 2), 1);

    // Table matches pointwise
    let table = kan_extension_table_witness(KanDirection::Left, &k, &f, 3);
    assert_eq!(table, vec![8, 7, 0]);
}

#[test]
fn yoneda_embedding_and_natural_transformation() {
    let cat = FiniteCategory::discrete(3);
    assert_eq!(yoneda_embedding_witness(&cat, 0), vec![1, 0, 0]);
    assert_eq!(yoneda_embedding_witness(&cat, 1), vec![0, 1, 0]);
    assert_eq!(yoneda_embedding_witness(&cat, 2), vec![0, 0, 1]);

    assert_eq!(natural_transformation_count_witness(&cat, 0, 42), 42);
    assert_eq!(natural_transformation_count_witness(&cat, 1, 0), 0);
}

#[test]
fn zx_diagram_rewrites() {
    let spiders = vec![
        ZxSpider {
            color: ZxColor::Z,
            phase_num: 1,
        },
        ZxSpider {
            color: ZxColor::Z,
            phase_num: 3,
        },
    ];
    let edges = vec![(0, 1)];
    let d = ZxDiagram {
        phase_denom: 8,
        spiders,
        edges,
    };

    let fused = zx_spider_fusion_witness(d);
    assert_eq!(fused.spiders.len(), 1);
    assert_eq!(fused.spiders[0].phase_num, 4);
    assert!(fused.edges.is_empty());

    // Identity removal
    let d2 = ZxDiagram {
        phase_denom: 8,
        spiders: vec![
            ZxSpider {
                color: ZxColor::Z,
                phase_num: 1,
            },
            ZxSpider {
                color: ZxColor::Z,
                phase_num: 0,
            },
            ZxSpider {
                color: ZxColor::Z,
                phase_num: 2,
            },
        ],
        edges: vec![(0, 1), (1, 2)],
    };
    let removed = zx_identity_removal_witness(d2);
    assert_eq!(removed.spiders.len(), 2);
    assert_eq!(removed.edges, vec![(0, 1)]);

    // Color change
    let mut d3 = ZxDiagram {
        phase_denom: 8,
        spiders: vec![ZxSpider {
            color: ZxColor::Z,
            phase_num: 3,
        }],
        edges: vec![],
    };
    zx_color_change_witness(&mut d3, 0);
    assert_eq!(d3.spiders[0].color, ZxColor::X);

    // Simplified diagram fixpoint
    let simp = zx_simplified_diagram_witness(removed);
    assert_eq!(simp.spiders.len(), 1);
    assert_eq!(simp.spiders[0].phase_num, 3);
}

#[test]
fn adjustment_set_safety_and_descendants() {
    let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
    assert!(adjustment_set_ordering_is_safe_witness(&adj, 0, 1, 3));
    assert!(!adjustment_set_ordering_is_safe_witness(&adj, 1, 0, 3));

    let desc = adjustment_set_pass_descendants_witness(&adj, 3);
    assert_eq!(desc[0], vec![1, 2]);
    assert_eq!(desc[1], vec![2]);
    assert!(desc[2].is_empty());

    // Empty and invalid dimensions
    assert!(!adjustment_set_ordering_is_safe_witness(
        &[0, 1, 0],
        0,
        1,
        2
    ));
    assert!(!adjustment_set_ordering_is_safe_witness(
        &[0, 1, 0, 0],
        2,
        1,
        2
    ));
    assert!(adjustment_set_pass_descendants_witness(&[], 0).is_empty());
}
