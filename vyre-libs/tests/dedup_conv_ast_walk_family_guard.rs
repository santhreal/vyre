//! Permanent guard for the two clone families collapsed in PR-13:
//! `math::conv` (direct conv vs im2col patch extraction) and
//! `graph::ast_walk_*` (preorder vs postorder VAST traversal).
//!
//! Section 1 pins `Program::fingerprint()` for every public entry point of
//! both families across a shape/size spread including boundary cases. A
//! fingerprint move means the emitted IR changed, which for a de-duplication
//! change is a semantic question, not a golden to re-pin.
//!
//! Section 2 is the numeric contract the fingerprints stand in for: direct
//! convolution must equal the im2col patch matrix contracted against the
//! kernel, both evaluated through the reference interpreter, and both walks
//! must equal the host traversal oracles. These survive an intentional IR
//! change; the fingerprints do not.
//!
//! GPU acquisition: none - reference interpreter only.

use vyre_foundation::ir::Program;
use vyre_libs::graph::{ast_walk_postorder, ast_walk_postorder_nodes, ast_walk_preorder};
use vyre_libs::math::conv::{conv2d_3x3_decision, conv2d_3x3_direct, im2col_3x3};
use vyre_primitives::wire::{decode_f32_le_bytes_all, pack_f32_slice, pack_u32_slice};
use vyre_reference::value::Value;

fn fingerprint_hex(program: &Program) -> String {
    program
        .fingerprint()
        .iter()
        .fold(String::with_capacity(64), |mut acc, byte| {
            acc.push_str(&format!("{byte:02x}"));
            acc
        })
}

/// Shape spread: degenerate rows/columns, the smallest square that has an
/// interior pixel, the registered fixture shape, and both sides of the
/// `conv2d_3x3_decision` im2col threshold (4096 pixels).
const CONV_SHAPES: &[(u32, u32)] = &[
    (1, 1),
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 3),
    (4, 4),
    (8, 8),
    (64, 63),
    (64, 64),
    (65, 65),
];

fn conv_family_fingerprints() -> Vec<(String, String)> {
    let mut rows = Vec::new();
    for &(h, w) in CONV_SHAPES {
        let direct = conv2d_3x3_direct("input", "kernel", "output", h, w)
            .expect("Fix: conv2d_3x3_direct must build for a non-degenerate shape.");
        rows.push((
            format!("conv2d_3x3_direct:{h}x{w}"),
            fingerprint_hex(&direct),
        ));

        let patches = im2col_3x3("input", "output", h, w)
            .expect("Fix: im2col_3x3 must build for a non-degenerate shape.");
        rows.push((format!("im2col_3x3:{h}x{w}"), fingerprint_hex(&patches)));

        let decision = conv2d_3x3_decision("input", "kernel", "output", h, w)
            .expect("Fix: conv2d_3x3_decision must build for a non-degenerate shape.");
        rows.push((
            format!("conv2d_3x3_decision:{h}x{w}"),
            fingerprint_hex(&decision),
        ));
    }
    rows
}

/// Walk spread: empty tree, single node, spine, the branching fixture, and
/// caps below / equal to / above the node count.
const WALK_SHAPES: &[(u32, u32)] = &[
    (0, 8),
    (1, 8),
    (1, 1),
    (4, 8),
    (6, 8),
    (8, 4),
    (8, 16),
    (8, 1),
];

fn walk_family_fingerprints() -> Vec<(String, String)> {
    let mut rows = Vec::new();
    for &(node_count, out_cap) in WALK_SHAPES {
        rows.push((
            format!("ast_walk_preorder:{node_count}/{out_cap}"),
            fingerprint_hex(&ast_walk_preorder("nodes", "out", node_count, out_cap)),
        ));
        rows.push((
            format!("ast_walk_postorder_nodes:{node_count}/{out_cap}"),
            fingerprint_hex(&ast_walk_postorder_nodes(
                "nodes", "out", node_count, out_cap,
            )),
        ));
    }
    for node_count in [0u32, 1, 4, 8] {
        rows.push((
            format!("ast_walk_postorder:{node_count}"),
            fingerprint_hex(&ast_walk_postorder("out", node_count)),
        ));
    }
    rows
}

fn assert_pinned(family: &str, actual: &[(String, String)], pinned: &[(&str, &str)]) {
    let mut divergences = Vec::new();
    if actual.len() != pinned.len() {
        divergences.push(format!(
            "entry count {} != pinned {}",
            actual.len(),
            pinned.len()
        ));
    }
    for (index, (label, digest)) in actual.iter().enumerate() {
        match pinned.get(index) {
            Some((pinned_label, pinned_digest)) => {
                if pinned_label != label {
                    divergences.push(format!("[{index}] label {pinned_label} != {label}"));
                } else if pinned_digest != digest {
                    divergences.push(format!("[{index}] {label}: {pinned_digest} != {digest}"));
                }
            }
            None => divergences.push(format!("[{index}] {label}: unpinned")),
        }
    }
    if divergences.is_empty() {
        return;
    }
    let table = actual
        .iter()
        .map(|(label, digest)| format!("    (\"{label}\", \"{digest}\"),"))
        .collect::<Vec<_>>()
        .join("\n");
    panic!(
        "{family} fingerprints diverged from the pinned pre-merge tree.\n\
         Divergences:\n  {}\n\
         A fingerprint move is a semantic question. Prove numeric equivalence \
         through the reference interpreter and record which behavior won before \
         touching this table.\nObserved:\n{table}",
        divergences.join("\n  ")
    );
}

/// Pinned on the pre-merge tree at b72b96dbc8, before either family was
/// collapsed onto a single owner.
const CONV_PINS: &[(&str, &str)] = &[
    (
        "conv2d_3x3_direct:1x1",
        "4c3ae41462bd7c0ede42aa23d3107f556c2f903d550a5156808135f912e79dfb",
    ),
    (
        "im2col_3x3:1x1",
        "09cd8133dfc56300f244c284235c03bfc9ba11427dbedc02fa44c65e8bcb96d7",
    ),
    (
        "conv2d_3x3_decision:1x1",
        "4c3ae41462bd7c0ede42aa23d3107f556c2f903d550a5156808135f912e79dfb",
    ),
    (
        "conv2d_3x3_direct:1x5",
        "60744680c2cd595acb224de3d02f5975284ae93f3a2170a9611ac1616fca9098",
    ),
    (
        "im2col_3x3:1x5",
        "ce5c180d5f242b74bbe0ee95fd375348411a6a69bd5348c02f58c1cd5a6b9ed6",
    ),
    (
        "conv2d_3x3_decision:1x5",
        "60744680c2cd595acb224de3d02f5975284ae93f3a2170a9611ac1616fca9098",
    ),
    (
        "conv2d_3x3_direct:5x1",
        "4ed6860ac6b6a384a1edbf22f126f7b1eec6ffac0a04785d03a06bde22917dbc",
    ),
    (
        "im2col_3x3:5x1",
        "7b0cd8b3d9cdc7b4f1aac98b93c9e4eddc019b3181eb9e5ab55b31a4a59084c4",
    ),
    (
        "conv2d_3x3_decision:5x1",
        "4ed6860ac6b6a384a1edbf22f126f7b1eec6ffac0a04785d03a06bde22917dbc",
    ),
    (
        "conv2d_3x3_direct:2x3",
        "d116c48341d46aece6f5b485efdb724a3f73b5d00b714bf3c867f2c908a9dec3",
    ),
    (
        "im2col_3x3:2x3",
        "e55f8e6d3b3c13812a0f6113cf37a8e12fdab1e1a8a95848c8758c1f56a2a95d",
    ),
    (
        "conv2d_3x3_decision:2x3",
        "d116c48341d46aece6f5b485efdb724a3f73b5d00b714bf3c867f2c908a9dec3",
    ),
    (
        "conv2d_3x3_direct:3x3",
        "22a4b5597652b20113c8299b607e418e043fd53a51e24c52e0c81a9a4638007b",
    ),
    (
        "im2col_3x3:3x3",
        "4b4c35959b639b4e3f5022f43cb18808dbcaf0a664fd3e39777dcb0b930122a1",
    ),
    (
        "conv2d_3x3_decision:3x3",
        "22a4b5597652b20113c8299b607e418e043fd53a51e24c52e0c81a9a4638007b",
    ),
    (
        "conv2d_3x3_direct:4x4",
        "8bb5a403a6f9b80c421afdbb9157c36cb1f286e28a35b641f02c5ba94e27fde8",
    ),
    (
        "im2col_3x3:4x4",
        "10f3acdebc88dd249887fd650e1807cd302048841ddc1fe65ff222c9635ad7fd",
    ),
    (
        "conv2d_3x3_decision:4x4",
        "8bb5a403a6f9b80c421afdbb9157c36cb1f286e28a35b641f02c5ba94e27fde8",
    ),
    (
        "conv2d_3x3_direct:8x8",
        "68f10294db503217f359bc7eee7b6c48e1f4b794718ed291d20d86fa2767334c",
    ),
    (
        "im2col_3x3:8x8",
        "032bb55e239de7b29228d1963603c55d7ae5790c738fa2df7ff14fd361f6562f",
    ),
    (
        "conv2d_3x3_decision:8x8",
        "68f10294db503217f359bc7eee7b6c48e1f4b794718ed291d20d86fa2767334c",
    ),
    (
        "conv2d_3x3_direct:64x63",
        "f63baaa5b6b7911ae7611b94c99bd885ca94c6d310f48d51ea8dca8fee453a38",
    ),
    (
        "im2col_3x3:64x63",
        "070cd727c4f472392b9ff60dcebaff22a44e45c93d06beb442ec6a78be81e05e",
    ),
    (
        "conv2d_3x3_decision:64x63",
        "f63baaa5b6b7911ae7611b94c99bd885ca94c6d310f48d51ea8dca8fee453a38",
    ),
    (
        "conv2d_3x3_direct:64x64",
        "fb45f42bc6664dedc8a8bbd21e9e56dfa2cdf3ecf5e5efb2d8a5398d9df2c1ea",
    ),
    (
        "im2col_3x3:64x64",
        "d062cb0a1024b50a85f515540ddef0055201adf9bc681511a4b52e60127a6b13",
    ),
    (
        "conv2d_3x3_decision:64x64",
        "5732ebfa1597679bed723d90f846933ff8f53bdca1fb6d94c435aade601d4995",
    ),
    (
        "conv2d_3x3_direct:65x65",
        "6d4463b09b4fc2c975d4cdcadcec95f2a64ff461511ccddc1a0763ebef027bac",
    ),
    (
        "im2col_3x3:65x65",
        "9b621bbe26261d0e8ce45104d6e06a631cc70370ae8c969d84d2621d65ce0e3c",
    ),
    (
        "conv2d_3x3_decision:65x65",
        "b796bf011ad434b9463efb9c86d99cff575a99426ce22696e67ad05b97c8d830",
    ),
];

/// Pinned on the same pre-merge tree.
const WALK_PINS: &[(&str, &str)] = &[
    (
        "ast_walk_preorder:0/8",
        "1a4ddf103c94b1da51a9e5c331709f69ab929597c16c99facdffad36612eaaf0",
    ),
    (
        "ast_walk_postorder_nodes:0/8",
        "1b73f269f582337d7e1bf15812f59bd924b76f3327762de91b4f4a4e6f847bc4",
    ),
    (
        "ast_walk_preorder:1/8",
        "3b3382780be9befa3f9e25030685e42c228ebb18eeb683d6b678f817f6ccf48b",
    ),
    (
        "ast_walk_postorder_nodes:1/8",
        "3e073d7fb96f9d8395d5bca82cff49e6858b2ade8b8d1718b2bef3d0f90cf789",
    ),
    (
        "ast_walk_preorder:1/1",
        "9d0d0b4716e4bfa53aafcd991ccb6b4f237a8dd9e1f8111990761e70e9a5119e",
    ),
    (
        "ast_walk_postorder_nodes:1/1",
        "a3e56147b00fa8c215067deaf49a2955db5e71dfe6743983c985efe29c9df84f",
    ),
    (
        "ast_walk_preorder:4/8",
        "e68761083cd25982fcf874300f7e88b036e299c20de07eced0e70c8be7f64f9c",
    ),
    (
        "ast_walk_postorder_nodes:4/8",
        "369d8b80bad1964d72f7813d2550d694d5a72c5b145a13fe578216ae2043f3d5",
    ),
    (
        "ast_walk_preorder:6/8",
        "dd5a5ec69059562fea7d9b3d8cb173a5a0a6c8ba98e06091ad0235e97f345c5b",
    ),
    (
        "ast_walk_postorder_nodes:6/8",
        "2e85a3e25ecce679723b46d1c1721019ae00d59a89481f8096d51cb99c5a80f9",
    ),
    (
        "ast_walk_preorder:8/4",
        "b4045fd328bd7564bdbbde6d8e7a5dc334f4e23f5d26783362b1b10420372cb8",
    ),
    (
        "ast_walk_postorder_nodes:8/4",
        "9940b91a745b61e96a399f2f8105fe4573ebda4b14702b2acf89209e31a174e1",
    ),
    (
        "ast_walk_preorder:8/16",
        "20928354ffbe04677f3f7db371f0ce3b3d6da4a990eebddf22ac3c5f7eaa57dd",
    ),
    (
        "ast_walk_postorder_nodes:8/16",
        "69d8539aef15f30f5a458dd55829bddd44d395df154de464675d3e79cef4a7b6",
    ),
    (
        "ast_walk_preorder:8/1",
        "6069843dba4a35d31be7c3da1589d798ab8368f029d99365256f57d003d726e4",
    ),
    (
        "ast_walk_postorder_nodes:8/1",
        "a0d960710cf14e44b4d862e5d4e3954c27e7d65f004373d2b5edcda29149ce0c",
    ),
    (
        "ast_walk_postorder:0",
        "5629e403cd40dd4a11b2b48f359cbab5868e24dc058f3579e3504295230464cd",
    ),
    (
        "ast_walk_postorder:1",
        "bf077cdd00d0c30fe461001aeb96e072ff24457af62cd9675c6f434169c81d01",
    ),
    (
        "ast_walk_postorder:4",
        "8db64ba191e7705338d3a74613e4777605ed4b4967b9e13e5e86e7985f86f58e",
    ),
    (
        "ast_walk_postorder:8",
        "ff8acc9176f0a5ef2b8b0d440fdafa5ac20cd1b86cd504a91cfbcc12eaca782b",
    ),
];

#[test]
fn conv_family_ir_is_pinned() {
    assert_pinned("math::conv", &conv_family_fingerprints(), CONV_PINS);
}

#[test]
fn ast_walk_family_ir_is_pinned() {
    assert_pinned("graph::ast_walk", &walk_family_fingerprints(), WALK_PINS);
}
// ---------------------------------------------------------------------------
// Numeric contracts: what the fingerprints stand in for.
// ---------------------------------------------------------------------------

/// Deterministic non-symmetric image; the ramp makes a transposed or
/// off-by-one patch index visible in the output.
fn image(h: u32, w: u32) -> Vec<f32> {
    (0..h * w).map(|i| (i % 17) as f32 - 4.5).collect()
}

/// Kernels with distinct taps, negative taps, and an asymmetric layout, so a
/// swapped `ky`/`kx` or a dropped tap cannot pass.
const KERNELS: &[[f32; 9]] = &[
    [1.0; 9],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
    [-0.5, 0.25, 0.125, 2.0, -3.0, 0.0625, 7.5, -0.75, 1.5],
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

/// The family's cross-entry-point equality: `conv2d_3x3_direct` must equal the
/// `im2col_3x3` patch matrix contracted against the kernel. This is the exact
/// property a "direct conv re-walks the convolution itself" implementation and
/// an "im2col plus gemm" implementation must agree on, so it holds across the
/// merge regardless of which one emits the IR.
#[test]
fn direct_conv_equals_im2col_contracted_with_the_kernel() {
    for &(h, w) in &[(1u32, 1u32), (1, 5), (5, 1), (2, 3), (3, 3), (4, 4), (8, 8)] {
        let input = image(h, w);
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
    for &(h, w) in &[(1u32, 1u32), (1, 5), (5, 1), (2, 3), (3, 3), (4, 4), (8, 8)] {
        let input = image(h, w);
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
fn conv_rejects_degenerate_shapes() {
    for (h, w) in [(0u32, 0u32), (0, 4), (4, 0)] {
        let error = conv2d_3x3_direct("input", "kernel", "output", h, w)
            .expect_err("Fix: a zero extent must be rejected, not silently emptied.");
        assert!(
            error.contains("non-zero height and width"),
            "{h}x{w}: {error}"
        );
    }
}

fn spine_nodes(node_count: u32) -> Vec<u8> {
    let full = vyre_foundation::vast::pack_spine_vast(&vec![1u32; node_count as usize]);
    let start = vyre_foundation::vast::HEADER_LEN;
    let len = (node_count as usize) * vyre_foundation::vast::NODE_STRIDE_U32 * 4;
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
    vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())
}

/// Both walk entry points must agree with the host traversal oracle on the same
/// tree, which is the contract a direction-parameterized single walk has to keep.
#[test]
fn both_walks_match_the_host_traversal_oracles() {
    for node_count in [1u32, 2, 4, 8] {
        let nodes = spine_nodes(node_count);
        let cap = 16u32;
        let words = cap as usize;

        let pre = run_walk(
            &ast_walk_preorder("nodes", "out", node_count, cap),
            &nodes,
            words,
        );
        let host_pre = vyre_foundation::vast::walk_preorder_indices(&nodes, node_count, 128)
            .expect("Fix: host preorder oracle must accept the spine fixture.");
        assert_eq!(
            &pre[..host_pre.len()],
            host_pre.as_slice(),
            "spine {node_count}: preorder walk must match the host oracle"
        );

        let post = run_walk(
            &ast_walk_postorder_nodes("nodes", "out", node_count, cap),
            &nodes,
            words,
        );
        let host_post = vyre_foundation::vast::walk_postorder_indices(&nodes, node_count, 128)
            .expect("Fix: host postorder oracle must accept the spine fixture.");
        assert_eq!(
            &post[..host_post.len()],
            host_post.as_slice(),
            "spine {node_count}: postorder walk must match the host oracle"
        );

        assert_eq!(
            host_post,
            host_pre.iter().rev().copied().collect::<Vec<_>>(),
            "spine {node_count}: postorder is the reverse of preorder"
        );
    }
}
