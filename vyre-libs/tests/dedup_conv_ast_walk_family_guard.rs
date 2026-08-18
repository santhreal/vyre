//! Permanent guard for the two clone families:
//! `math::conv` (direct conv vs im2col patch extraction vs decision router) and
//! `graph::ast_walk_*` / `graph::vast_tree_walk` (preorder vs postorder VAST traversal).
//!
//! Replaces stale serialized IR hash fingerprints with live family derivation,
//! compile-time/run-time closure over all declared public entry points,
//! canonical shared-owner structural invariants, and byte-exact reference
//! evaluation or algebraic equivalence against host oracles.
//!
//! GPU acquisition: none - reference interpreter only.

#![cfg(feature = "graph")]
#![forbid(unsafe_code)]

mod harness;

use std::collections::BTreeSet;

use vyre_foundation::ir::Program;
use vyre_foundation::vast::{
    pack_spine_vast, walk_postorder_indices, walk_preorder_indices, NODE_STRIDE_U32,
};
use vyre_libs::graph::vast_tree_walk::{
    self, try_ast_walk_plan, try_ast_walk_postorder, try_ast_walk_preorder, VastTreeWalkPlan,
    VastWalkOrder,
};
use vyre_libs::graph::{
    ast_walk, ast_walk_postorder, ast_walk_postorder_nodes, ast_walk_preorder,
    pack_branching_fixture,
};
use vyre_libs::math::conv::{conv2d_3x3_decision, conv2d_3x3_direct, im2col_3x3};
use vyre_primitives::wire::{
    decode_f32_le_bytes_all, decode_u32_le_bytes_all, pack_f32_slice, pack_u32_slice,
};
use vyre_reference::value::Value;

// ===========================================================================
// Section 1: Live Family Derivation & Classification Gates
// ===========================================================================

/// Classified members of the `math::conv` dialect family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ConvFamilyMember {
    Direct,
    Im2Col,
    Decision,
}

impl ConvFamilyMember {
    const ALL: [Self; 3] = [Self::Direct, Self::Im2Col, Self::Decision];
}

/// Classified members of the `graph::ast_walk` / `graph::vast_tree_walk` family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum AstWalkFamilyMember {
    AstWalkGeneric,
    AstWalkPreorder,
    AstWalkPostorderNodes,
    AstWalkPostorderSpine,
    VastBuildPlan,
    VastBuildCheckedPreorder,
    VastBuildCheckedPostorder,
    VastBuildTrustedPreorder,
    VastBuildTrustedPostorder,
    VastTryOrder,
    VastTryPreorder,
    VastTryPostorder,
    VastTryPlan,
    VastPreorder,
    VastPostorder,
}

impl AstWalkFamilyMember {
    #[allow(dead_code)]
    const ALL: [Self; 15] = [
        Self::AstWalkGeneric,
        Self::AstWalkPreorder,
        Self::AstWalkPostorderNodes,
        Self::AstWalkPostorderSpine,
        Self::VastBuildPlan,
        Self::VastBuildCheckedPreorder,
        Self::VastBuildCheckedPostorder,
        Self::VastBuildTrustedPreorder,
        Self::VastBuildTrustedPostorder,
        Self::VastTryOrder,
        Self::VastTryPreorder,
        Self::VastTryPostorder,
        Self::VastTryPlan,
        Self::VastPreorder,
        Self::VastPostorder,
    ];
}

/// Extract all `pub fn` identifiers and `pub use` re-exports from a Rust source file.
fn extract_declared_public_items(source: &str) -> BTreeSet<String> {
    let mut items = BTreeSet::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed
            .strip_prefix("pub const fn ")
            .or_else(|| trimmed.strip_prefix("pub fn "))
        {
            if let Some((name, _)) = rest.split_once('(') {
                let name = name.trim().trim_start_matches("r#");
                if !name.is_empty() && !name.starts_with('_') {
                    items.insert(name.to_string());
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("pub use ") {
            let item = rest.split("//").next().unwrap_or(rest).trim();
            let item = item.trim_end_matches(';').trim();
            if let Some((_, name)) = item.rsplit_once("::") {
                let name = name
                    .trim()
                    .trim_start_matches('{')
                    .trim_end_matches('}')
                    .trim();
                for sub in name.split(',') {
                    let sub = sub.trim();
                    if !sub.is_empty() {
                        items.insert(sub.to_string());
                    }
                }
            }
        }
    }
    items
}

fn classify_conv_entrypoint(name: &str) -> Option<ConvFamilyMember> {
    match name {
        "conv2d_3x3_direct" => Some(ConvFamilyMember::Direct),
        "im2col_3x3" => Some(ConvFamilyMember::Im2Col),
        "conv2d_3x3_decision" => Some(ConvFamilyMember::Decision),
        _ => None,
    }
}

#[test]
fn conv_family_roster_is_exhaustively_classified() {
    let source = harness::crate_file("src/math/conv/mod.rs");
    let declared = extract_declared_public_items(&source);
    assert!(
        declared.contains("conv2d_3x3_direct")
            && declared.contains("im2col_3x3")
            && declared.contains("conv2d_3x3_decision"),
        "Fix: src/math/conv/mod.rs must declare the core conv family entrypoints: {declared:?}"
    );

    let mut unclassified = Vec::new();
    let mut classified_members = BTreeSet::new();

    for name in &declared {
        match classify_conv_entrypoint(name) {
            Some(member) => {
                classified_members.insert(member);
            }
            None => {
                unclassified.push(name.clone());
            }
        }
    }

    assert!(
        unclassified.is_empty(),
        "Fix: newly added conv family member(s) must be classified and tested: {:?}",
        unclassified
    );

    for member in ConvFamilyMember::ALL {
        assert!(
            classified_members.contains(&member),
            "Fix: ConvFamilyMember::{member:?} has no corresponding declaration in src/math/conv/mod.rs"
        );
    }
}

#[test]
fn ast_walk_family_roster_is_exhaustively_classified() {
    let walk_source = harness::crate_file("src/graph/ast_walk.rs");
    let vast_source = harness::crate_file("src/graph/vast_tree_walk.rs");

    let declared_walk = extract_declared_public_items(&walk_source);
    let declared_vast = extract_declared_public_items(&vast_source);

    // Filter out fixture generators which are test-only helpers
    let core_walk_fns: BTreeSet<String> = declared_walk
        .into_iter()
        .filter(|name| !name.starts_with("pack_"))
        .collect();

    let mut unclassified_walk = Vec::new();
    let mut covered_walk = BTreeSet::new();

    for name in &core_walk_fns {
        match name.as_str() {
            "ast_walk" => {
                covered_walk.insert(AstWalkFamilyMember::AstWalkGeneric);
            }
            "ast_walk_preorder" => {
                covered_walk.insert(AstWalkFamilyMember::AstWalkPreorder);
            }
            "ast_walk_postorder_nodes" => {
                covered_walk.insert(AstWalkFamilyMember::AstWalkPostorderNodes);
            }
            "ast_walk_postorder" => {
                covered_walk.insert(AstWalkFamilyMember::AstWalkPostorderSpine);
            }
            other => unclassified_walk.push(other.to_string()),
        }
    }

    assert!(
        unclassified_walk.is_empty(),
        "Fix: newly added ast_walk family member(s) must be classified and tested: {:?}",
        unclassified_walk
    );

    let mut unclassified_vast = Vec::new();
    let mut covered_vast = BTreeSet::new();

    for name in &declared_vast {
        match name.as_str() {
            "build_vast_tree_walk_plan" => {
                covered_vast.insert(AstWalkFamilyMember::VastBuildPlan);
            }
            "build_checked_preorder_walk" => {
                covered_vast.insert(AstWalkFamilyMember::VastBuildCheckedPreorder);
            }
            "build_checked_postorder_walk" => {
                covered_vast.insert(AstWalkFamilyMember::VastBuildCheckedPostorder);
            }
            "build_trusted_preorder_walk" => {
                covered_vast.insert(AstWalkFamilyMember::VastBuildTrustedPreorder);
            }
            "build_trusted_postorder_walk" => {
                covered_vast.insert(AstWalkFamilyMember::VastBuildTrustedPostorder);
            }
            "try_ast_walk_order" => {
                covered_vast.insert(AstWalkFamilyMember::VastTryOrder);
            }
            "try_ast_walk_preorder" => {
                covered_vast.insert(AstWalkFamilyMember::VastTryPreorder);
            }
            "try_ast_walk_postorder" => {
                covered_vast.insert(AstWalkFamilyMember::VastTryPostorder);
            }
            "try_ast_walk_plan" => {
                covered_vast.insert(AstWalkFamilyMember::VastTryPlan);
            }
            "ast_walk_preorder" => {
                covered_vast.insert(AstWalkFamilyMember::VastPreorder);
            }
            "ast_walk_postorder" => {
                covered_vast.insert(AstWalkFamilyMember::VastPostorder);
            }
            "primitive_op_ids" => {} // metadata helper
            other => unclassified_vast.push(other.to_string()),
        }
    }
    assert!(
        unclassified_vast.is_empty(),
        "Fix: newly added vast_tree_walk member(s) must be classified and tested: {:?}",
        unclassified_vast
    );
}

// ===========================================================================
// Section 2: Canonical Shared-Owner Closure Invariants
// ===========================================================================

#[test]
fn conv_family_shares_canonical_stencil_owner() {
    let conv2d_src = harness::crate_file("src/math/conv/conv2d.rs");
    let im2col_src = harness::crate_file("src/math/conv/im2col.rs");
    let mod_src = harness::crate_file("src/math/conv/mod.rs");

    // Both conv2d and im2col must use the shared stencil helper rather than hand-rolling coordinate/tap logic.
    assert!(
        conv2d_src.contains("stencil_3x3_taps") || conv2d_src.contains("decompose_index"),
        "Fix: conv2d.rs must build upon crate::builder::stencil shared owner."
    );
    assert!(
        im2col_src.contains("stencil_3x3_taps") || im2col_src.contains("decompose_index"),
        "Fix: im2col.rs must build upon crate::builder::stencil shared owner."
    );

    // conv2d_3x3_decision must delegate to conv2d_3x3_direct under both arms.
    assert!(
        mod_src.contains("conv2d_3x3_direct("),
        "Fix: conv2d_3x3_decision must delegate to conv2d_3x3_direct."
    );

    // Structural IR invariant check: conv2d_3x3_direct and im2col_3x3 share identical invocation grid shape
    let direct_prog = conv2d_3x3_direct("in", "k", "out", 8, 8).expect("Fix: direct conv builds");
    let im2col_prog = im2col_3x3("in", "out", 8, 8).expect("Fix: im2col builds");
    let decision_prog =
        conv2d_3x3_decision("in", "k", "out", 8, 8).expect("Fix: decision conv builds");

    assert_eq!(
        direct_prog.workgroup_size(),
        im2col_prog.workgroup_size(),
        "Direct conv and im2col must share workgroup geometry."
    );
    assert_eq!(
        direct_prog.workgroup_size(),
        decision_prog.workgroup_size(),
        "Direct conv and decision conv must share workgroup geometry."
    );
}

#[test]
fn ast_walk_family_shares_canonical_tree_walk_owner() {
    let walk_src = harness::crate_file("src/graph/ast_walk.rs");

    // ast_walk.rs must delegate to vast_tree_walk rather than implementing its own loop.
    assert!(
        walk_src.contains("vast_tree_walk::try_ast_walk_order"),
        "Fix: ast_walk.rs must delegate to vast_tree_walk::try_ast_walk_order."
    );

    // ast_walk_preorder and ast_walk_postorder_nodes must delegate to ast_walk.
    assert!(
        walk_src.contains("ast_walk(VastWalkOrder::Preorder"),
        "Fix: ast_walk_preorder must delegate to ast_walk."
    );
    assert!(
        walk_src.contains("ast_walk(VastWalkOrder::Postorder"),
        "Fix: ast_walk_postorder_nodes must delegate to ast_walk."
    );

    // Structural AST equivalence: compare IR nodes between ast_walk and vast_tree_walk
    let node_count = 6u32;
    let out_cap = 8u32;

    let pre_direct = ast_walk_preorder("nodes", "out", node_count, out_cap);
    let pre_order = ast_walk(VastWalkOrder::Preorder, "nodes", "out", node_count, out_cap);
    let pre_vast = try_ast_walk_preorder("nodes", "out", node_count, out_cap)
        .expect("Fix: try_ast_walk_preorder builds");

    // Buffers and workgroup size must be identical
    assert_eq!(pre_direct.buffers(), pre_order.buffers());
    assert_eq!(pre_direct.workgroup_size(), pre_order.workgroup_size());
    assert_eq!(pre_direct.buffers(), pre_vast.buffers());

    let post_direct = ast_walk_postorder_nodes("nodes", "out", node_count, out_cap);
    let post_order = ast_walk(
        VastWalkOrder::Postorder,
        "nodes",
        "out",
        node_count,
        out_cap,
    );
    let post_vast = try_ast_walk_postorder("nodes", "out", node_count, out_cap)
        .expect("Fix: try_ast_walk_postorder builds");

    assert_eq!(post_direct.buffers(), post_order.buffers());
    assert_eq!(post_direct.workgroup_size(), post_order.workgroup_size());
    assert_eq!(post_direct.buffers(), post_vast.buffers());
}

// ===========================================================================
// Section 3: Convolution Family Semantic Parity & Algebraic Equivalence
// ===========================================================================

const CONV_SHAPES: &[(u32, u32)] = &[
    (1, 1),
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 3),
    (4, 4),
    (8, 8),
    (16, 16),
    (64, 63),
    (64, 64),
    (65, 65),
];

fn test_image(h: u32, w: u32) -> Vec<f32> {
    (0..h * w).map(|i| (i % 17) as f32 - 4.5).collect()
}

const KERNELS: &[[f32; 9]] = &[
    [1.0; 9],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
    [-0.5, 0.25, 0.125, 2.0, -3.0, 0.0625, 7.5, -0.75, 1.5],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // identity tap
];

fn run_conv(h: u32, w: u32, input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let program = conv2d_3x3_direct("input", "kernel", "output", h, w)
        .expect("Fix: conv2d_3x3_direct must build.");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32_slice(input)),
            Value::from(pack_f32_slice(kernel)),
        ],
    )
    .expect("Fix: conv2d_3x3_direct must execute in the reference interpreter.");
    decode_f32_le_bytes_all(&outputs[0].to_bytes())
}

fn run_conv_decision(h: u32, w: u32, input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let program = conv2d_3x3_decision("input", "kernel", "output", h, w)
        .expect("Fix: conv2d_3x3_decision must build.");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32_slice(input)),
            Value::from(pack_f32_slice(kernel)),
        ],
    )
    .expect("Fix: conv2d_3x3_decision must execute in the reference interpreter.");
    decode_f32_le_bytes_all(&outputs[0].to_bytes())
}

fn run_im2col(h: u32, w: u32, input: &[f32]) -> Vec<f32> {
    let program = im2col_3x3("input", "output", h, w).expect("Fix: im2col_3x3 must build.");
    let outputs = vyre_reference::reference_eval(&program, &[Value::from(pack_f32_slice(input))])
        .expect("Fix: im2col_3x3 must execute in the reference interpreter.");
    decode_f32_le_bytes_all(&outputs[0].to_bytes())
}

fn host_conv(h: u32, w: u32, input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let (h, w) = (h as i64, w as i64);
    let mut out = vec![0.0f32; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for ky in 0..3i64 {
                for kx in 0..3i64 {
                    let ny = y + ky - 1;
                    let nx = x + kx - 1;
                    if ny < 0 || ny >= h || nx < 0 || nx >= w {
                        continue;
                    }
                    acc += input[(ny * w + nx) as usize] * kernel[(ky * 3 + kx) as usize];
                }
            }
            out[(y * w + x) as usize] = acc;
        }
    }
    out
}

fn host_im2col(h: u32, w: u32, input: &[f32]) -> Vec<f32> {
    let (h, w) = (h as i64, w as i64);
    let mut out = Vec::with_capacity((h * w * 9) as usize);
    for y in 0..h {
        for x in 0..w {
            for ky in 0..3i64 {
                for kx in 0..3i64 {
                    let ny = y + ky - 1;
                    let nx = x + kx - 1;
                    if ny >= 0 && ny < h && nx >= 0 && nx < w {
                        out.push(input[(ny * w + nx) as usize]);
                    } else {
                        out.push(0.0f32);
                    }
                }
            }
        }
    }
    out
}

#[test]
fn direct_conv_equals_im2col_contracted_with_the_kernel() {
    for &(h, w) in CONV_SHAPES {
        let input = test_image(h, w);
        let patches = run_im2col(h, w, &input);
        assert_eq!(
            patches.len(),
            (h * w * 9) as usize,
            "{h}x{w}: im2col must emit 9 cells per pixel"
        );
        for kernel in KERNELS {
            let direct = run_conv(h, w, &input, kernel);
            let gemm: Vec<f32> = (0..(h * w) as usize)
                .map(|pixel| {
                    (0..9)
                        .map(|tap| patches[pixel * 9 + tap] * kernel[tap])
                        .fold(0.0f32, |acc, term| acc + term)
                })
                .collect();
            assert_eq!(
                direct, gemm,
                "{h}x{w} kernel {kernel:?}: direct conv must equal im2col x kernel bit-for-bit"
            );
        }
    }
}

#[test]
fn conv_matches_host_zero_padded_convolution() {
    for &(h, w) in CONV_SHAPES {
        let input = test_image(h, w);
        for kernel in KERNELS {
            assert_eq!(
                run_conv(h, w, &input, kernel),
                host_conv(h, w, &input, kernel),
                "{h}x{w} kernel {kernel:?}: conv must match the host zero-padded oracle"
            );
        }
    }
}

#[test]
fn im2col_matches_host_patch_extraction() {
    for &(h, w) in CONV_SHAPES {
        let input = test_image(h, w);
        assert_eq!(
            run_im2col(h, w, &input),
            host_im2col(h, w, &input),
            "{h}x{w}: im2col output must match the host patch extraction oracle bit-for-bit"
        );
    }
}

#[test]
fn conv2d_decision_matches_direct_and_host_oracle_across_threshold() {
    for &(h, w) in CONV_SHAPES {
        let input = test_image(h, w);
        for kernel in KERNELS {
            let decision_result = run_conv_decision(h, w, &input, kernel);
            let host_result = host_conv(h, w, &input, kernel);
            assert_eq!(
                decision_result, host_result,
                "{h}x{w} kernel {kernel:?}: decision conv must match the host oracle bit-for-bit"
            );
        }

        let prog = conv2d_3x3_decision("in", "k", "out", h, w).expect("decision program builds");
        let pixels = h * w;
        if pixels >= 4096 {
            // Must contain hint for im2col preferred
            let dump = format!("{prog:?}");
            assert!(
                dump.contains("conv2d_3x3_im2col_preferred"),
                "{h}x{w}: large image must carry im2col preference tag in region header"
            );
        }
    }
}

#[test]
fn conv_rejects_degenerate_and_overflowing_shapes() {
    for (h, w) in [(0u32, 0u32), (0, 4), (4, 0)] {
        let error = conv2d_3x3_direct("input", "kernel", "output", h, w)
            .expect_err("Fix: a zero extent must be rejected, not silently emptied.");
        assert!(
            error.contains("non-zero height and width"),
            "{h}x{w}: {error}"
        );
    }

    // Overflow check
    let err = conv2d_3x3_direct("in", "k", "out", u32::MAX, 2)
        .expect_err("Fix: overflowing dimensions must error.");
    assert!(err.contains("overflows u32"), "{err}");

    let err_im = im2col_3x3("in", "out", u32::MAX / 4, 2)
        .expect_err("Fix: overflowing im2col dimensions must error.");
    assert!(err_im.contains("overflows u32"), "{err_im}");
}

// ===========================================================================
// Section 4: AST-Walk Family Semantic Parity & Host Oracle Equivalence
// ===========================================================================

fn spine_nodes(node_count: u32) -> Vec<u8> {
    let full = pack_spine_vast(&vec![1u32; node_count as usize]);
    let start = vyre_foundation::vast::HEADER_LEN;
    let len = (node_count as usize) * NODE_STRIDE_U32 * 4;
    full[start..start + len].to_vec()
}

fn run_walk(program: &Program, nodes: &[u8], out_words: usize) -> Vec<u32> {
    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(nodes.to_vec()),
            Value::from(pack_u32_slice(&vec![0u32; out_words])),
        ],
    )
    .expect("Fix: an AST walk must execute in the reference interpreter.");
    decode_u32_le_bytes_all(&outputs[0].to_bytes())
}

#[test]
fn both_walks_match_the_host_traversal_oracles_on_spines() {
    for node_count in [1u32, 2, 4, 8, 16] {
        let nodes = spine_nodes(node_count);
        let cap = 32u32;
        let words = cap as usize;

        // 1. ast_walk_preorder
        let pre = run_walk(
            &ast_walk_preorder("nodes", "out", node_count, cap),
            &nodes,
            words,
        );
        let host_pre = walk_preorder_indices(&nodes, node_count, 128)
            .expect("Fix: host preorder oracle must accept the spine fixture.");
        assert_eq!(
            &pre[..host_pre.len()],
            host_pre.as_slice(),
            "spine {node_count}: preorder walk must match the host oracle"
        );

        // 2. Order-parameterized preorder
        let pre_order = run_walk(
            &ast_walk(VastWalkOrder::Preorder, "nodes", "out", node_count, cap),
            &nodes,
            words,
        );
        assert_eq!(
            &pre_order[..host_pre.len()],
            host_pre.as_slice(),
            "spine {node_count}: order-parameterized preorder walk must match host oracle"
        );

        // 3. vast_tree_walk::try_ast_walk_preorder
        let vast_pre = run_walk(
            &try_ast_walk_preorder("nodes", "out", node_count, cap).expect("vast preorder builds"),
            &nodes,
            words,
        );
        assert_eq!(
            &vast_pre[..host_pre.len()],
            host_pre.as_slice(),
            "spine {node_count}: vast_tree_walk preorder must match host oracle"
        );

        // 4. ast_walk_postorder_nodes
        let post = run_walk(
            &ast_walk_postorder_nodes("nodes", "out", node_count, cap),
            &nodes,
            words,
        );
        let host_post = walk_postorder_indices(&nodes, node_count, 128)
            .expect("Fix: host postorder oracle must accept the spine fixture.");
        assert_eq!(
            &post[..host_post.len()],
            host_post.as_slice(),
            "spine {node_count}: postorder walk must match the host oracle"
        );

        // 5. Order-parameterized postorder
        let post_order = run_walk(
            &ast_walk(VastWalkOrder::Postorder, "nodes", "out", node_count, cap),
            &nodes,
            words,
        );
        assert_eq!(
            &post_order[..host_post.len()],
            host_post.as_slice(),
            "spine {node_count}: order-parameterized postorder walk must match host oracle"
        );

        // 6. vast_tree_walk::try_ast_walk_postorder
        let vast_post = run_walk(
            &try_ast_walk_postorder("nodes", "out", node_count, cap)
                .expect("vast postorder builds"),
            &nodes,
            words,
        );
        assert_eq!(
            &vast_post[..host_post.len()],
            host_post.as_slice(),
            "spine {node_count}: vast_tree_walk postorder must match host oracle"
        );

        // 7. Spine closed-form postorder helper agreement (single output buffer)
        let spine_post_prog = ast_walk_postorder("out", node_count);
        let spine_post = {
            let outputs = vyre_reference::reference_eval(
                &spine_post_prog,
                &[Value::from(pack_u32_slice(&vec![
                    0u32;
                    node_count as usize
                ]))],
            )
            .expect("Fix: spine postorder walk must execute in reference interpreter.");
            decode_u32_le_bytes_all(&outputs[0].to_bytes())
        };
        assert_eq!(
            spine_post.as_slice(),
            host_post.as_slice(),
            "spine {node_count}: ast_walk_postorder closed-form must match host postorder oracle"
        );

        // 7b. Checked and trusted vast_tree_walk builder helpers
        let checked_pre = run_walk(
            &vast_tree_walk::build_checked_preorder_walk("nodes", "out", node_count, cap)
                .expect("checked preorder builds"),
            &nodes,
            words,
        );
        assert_eq!(&checked_pre[..host_pre.len()], host_pre.as_slice());

        let checked_post = run_walk(
            &vast_tree_walk::build_checked_postorder_walk("nodes", "out", node_count, cap)
                .expect("checked postorder builds"),
            &nodes,
            words,
        );
        assert_eq!(&checked_post[..host_post.len()], host_post.as_slice());

        let trusted_pre = run_walk(
            &vast_tree_walk::build_trusted_preorder_walk("nodes", "out", node_count, cap),
            &nodes,
            words,
        );
        assert_eq!(&trusted_pre[..host_pre.len()], host_pre.as_slice());

        let trusted_post = run_walk(
            &vast_tree_walk::build_trusted_postorder_walk("nodes", "out", node_count, cap),
            &nodes,
            words,
        );
        assert_eq!(&trusted_post[..host_post.len()], host_post.as_slice());

        // 8. Preorder vs Postorder duality on a spine: postorder is exact reverse of preorder
        assert_eq!(
            host_post,
            host_pre.iter().rev().copied().collect::<Vec<_>>(),
            "spine {node_count}: postorder is the reverse of preorder"
        );

        // 9. Combined plan verification
        let plan: VastTreeWalkPlan = try_ast_walk_plan("nodes", "pre", "post", node_count, cap)
            .expect("Fix: try_ast_walk_plan must build successfully.");
        let plan_pre = run_walk(&plan.preorder, &nodes, words);
        let plan_post = run_walk(&plan.postorder, &nodes, words);
        assert_eq!(&plan_pre[..host_pre.len()], host_pre.as_slice());
        assert_eq!(&plan_post[..host_post.len()], host_post.as_slice());
    }
}

#[test]
fn both_walks_match_the_host_traversal_oracles_on_branching_trees() {
    let branching_nodes = pack_branching_fixture();
    let node_count = 6u32;
    let cap = 8u32;
    let words = cap as usize;

    let pre = run_walk(
        &ast_walk_preorder("nodes", "out", node_count, cap),
        &branching_nodes,
        words,
    );
    let host_pre = walk_preorder_indices(&branching_nodes, node_count, 128)
        .expect("Fix: host preorder oracle accepts branching fixture.");

    assert_eq!(
        &pre[..host_pre.len()],
        host_pre.as_slice(),
        "branching fixture: preorder walk must match host oracle [0, 1, 4, 2, 3, 5]"
    );

    let post = run_walk(
        &ast_walk_postorder_nodes("nodes", "out", node_count, cap),
        &branching_nodes,
        words,
    );
    let host_post = walk_postorder_indices(&branching_nodes, node_count, 128)
        .expect("Fix: host postorder oracle accepts branching fixture.");

    assert_eq!(
        &post[..host_post.len()],
        host_post.as_slice(),
        "branching fixture: postorder walk must match host oracle [4, 1, 2, 5, 3, 0]"
    );
}

#[test]
fn ast_walk_capacity_and_degenerate_invariants() {
    // Capacity truncation: walk must not overflow capacity
    let branching_nodes = pack_branching_fixture();
    let node_count = 6u32;
    let small_cap = 3u32;

    let program = ast_walk_preorder("nodes", "out", node_count, small_cap);
    let (outputs, oob_report) = vyre_reference::reference_eval_oob_report(
        &program,
        &[
            Value::from(branching_nodes),
            Value::from(pack_u32_slice(&vec![0u32; small_cap as usize])),
        ],
    )
    .expect("Fix: capacity-limited walk must execute without reference error.");

    assert_eq!(
        oob_report.total(),
        0,
        "Capacity-limited walk must produce zero out-of-bounds writes."
    );

    let decoded = decode_u32_le_bytes_all(&outputs[0].to_bytes());
    assert_eq!(
        decoded.len(),
        small_cap as usize,
        "Must emit exactly output capacity words."
    );

    // Empty tree (node_count = 0)
    let empty_prog = ast_walk_preorder("nodes", "out", 0, 4);
    let (empty_out, empty_oob) = vyre_reference::reference_eval_oob_report(
        &empty_prog,
        &[
            Value::from(vec![0u8; 32]),
            Value::from(pack_u32_slice(&[0u32; 4])),
        ],
    )
    .expect("Fix: 0-node walk must evaluate safely.");
    assert_eq!(empty_oob.total(), 0);
    assert_eq!(decode_u32_le_bytes_all(&empty_out[0].to_bytes()).len(), 4);
}

// ===========================================================================
// Section 5: Deliberate Semantic Divergence Defense & Mutation Gates
// ===========================================================================

#[test]
fn deliberate_semantic_divergence_fails_parity_assertions() {
    // Negative test 1: Mutated kernel tap must fail direct conv == im2col contraction
    let (h, w) = (4u32, 4u32);
    let input = test_image(h, w);
    let patches = run_im2col(h, w, &input);
    let kernel = [1.0f32; 9];
    let direct = run_conv(h, w, &input, &kernel);

    let mut corrupted_patches = patches.clone();
    corrupted_patches[0] += 0.1; // corrupt one cell
    let corrupted_gemm: Vec<f32> = (0..(h * w) as usize)
        .map(|pixel| {
            (0..9)
                .map(|tap| corrupted_patches[pixel * 9 + tap] * kernel[tap])
                .fold(0.0f32, |acc, term| acc + term)
        })
        .collect();

    assert_ne!(
        direct, corrupted_gemm,
        "Deliberate divergence in patch extraction must fail equality assertion."
    );

    // Negative test 2: Mutated walk index must fail host oracle assertion
    let nodes = spine_nodes(4);
    let host_pre = walk_preorder_indices(&nodes, 4, 128).unwrap();
    let mut corrupted_pre = host_pre.clone();
    corrupted_pre[0] ^= 1; // corrupt index
    assert_ne!(
        host_pre, corrupted_pre,
        "Deliberate divergence in walk indices must fail host oracle comparison."
    );

    // Negative test 3: Unclassified synthetic member name must fail classification
    assert!(
        classify_conv_entrypoint("conv2d_3x3_unclassified_variant").is_none(),
        "Unclassified entry point must return None and fail the exhaustiveness gate."
    );
}
