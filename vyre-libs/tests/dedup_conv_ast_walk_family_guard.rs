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
        "57ad59fc51c5fdf9231281415f254b85266422d52117d5b4f9edf74d473817c1",
    ),
    (
        "im2col_3x3:1x1",
        "3bb89380f338405ab83cdaf9773d469758087309d1c022a25234217bdc19d359",
    ),
    (
        "conv2d_3x3_decision:1x1",
        "57ad59fc51c5fdf9231281415f254b85266422d52117d5b4f9edf74d473817c1",
    ),
    (
        "conv2d_3x3_direct:1x5",
        "56814bc08017d0a879faf770206b272c3c83e05c828029585d3323cfae823104",
    ),
    (
        "im2col_3x3:1x5",
        "5cf8daef58d7ec019a9624aba1ae8b7574e60045d6c1a6e2f1ac496c97aa47d8",
    ),
    (
        "conv2d_3x3_decision:1x5",
        "56814bc08017d0a879faf770206b272c3c83e05c828029585d3323cfae823104",
    ),
    (
        "conv2d_3x3_direct:5x1",
        "613ce86590b5d7835bba2192bcca2b251f4e7468fee611db9ba06b51fa7ec196",
    ),
    (
        "im2col_3x3:5x1",
        "ad86f2a385a38c90033114c2bb6f19020e9378f8902775c370a62b1ae08de437",
    ),
    (
        "conv2d_3x3_decision:5x1",
        "613ce86590b5d7835bba2192bcca2b251f4e7468fee611db9ba06b51fa7ec196",
    ),
    (
        "conv2d_3x3_direct:2x3",
        "620957948987f1f1884b05819cc567df0a3a2b796d61461eca9513f1ebca1302",
    ),
    (
        "im2col_3x3:2x3",
        "923cb887e843796c73f4cceb13f91db1104c8b1606bfbbddecc3959fa105c0a3",
    ),
    (
        "conv2d_3x3_decision:2x3",
        "620957948987f1f1884b05819cc567df0a3a2b796d61461eca9513f1ebca1302",
    ),
    (
        "conv2d_3x3_direct:3x3",
        "dff1981905bc92ae158ff30d6d40429e047af8b3247d8afc5f04481b03f9662a",
    ),
    (
        "im2col_3x3:3x3",
        "b2463482d90d0cc0a44f8f49c8a85e38f08d3796be16a15c0c4e1d5ae849a5a3",
    ),
    (
        "conv2d_3x3_decision:3x3",
        "dff1981905bc92ae158ff30d6d40429e047af8b3247d8afc5f04481b03f9662a",
    ),
    (
        "conv2d_3x3_direct:4x4",
        "1eb1c3b9be19790eeff8c95437daddddf30dee7c639c6e7bf4b4e88f9815f0e5",
    ),
    (
        "im2col_3x3:4x4",
        "e11d70f66984287c73a68a77fed3d95737f0828ac887c6509024f0879ac46940",
    ),
    (
        "conv2d_3x3_decision:4x4",
        "1eb1c3b9be19790eeff8c95437daddddf30dee7c639c6e7bf4b4e88f9815f0e5",
    ),
    (
        "conv2d_3x3_direct:8x8",
        "9376b07e65bd33f647230c2a14c790d09a95abb4f5b62997fa52c1ac992c5de6",
    ),
    (
        "im2col_3x3:8x8",
        "6523b1251278b5eb733dc54b32bfd68e0e839a233e4bfec8d3fcb5ba8faa4e3b",
    ),
    (
        "conv2d_3x3_decision:8x8",
        "9376b07e65bd33f647230c2a14c790d09a95abb4f5b62997fa52c1ac992c5de6",
    ),
    (
        "conv2d_3x3_direct:64x63",
        "80a0e394c01b7f3855c1602a99e24ed548942488547e9368a06dab72d8979bbc",
    ),
    (
        "im2col_3x3:64x63",
        "7016be2fde917821feddb1139ace0b5189e0f548ec4746d66cf11a634400ff79",
    ),
    (
        "conv2d_3x3_decision:64x63",
        "80a0e394c01b7f3855c1602a99e24ed548942488547e9368a06dab72d8979bbc",
    ),
    (
        "conv2d_3x3_direct:64x64",
        "94ab044b3bc64fe726b07cdb97dd1a0aafe7224f9bd08c6ebe6e65e709604a9b",
    ),
    (
        "im2col_3x3:64x64",
        "22d35dde4f60e18f010117f1765e845d9397b2fd99e762fd53409446f64f335c",
    ),
    (
        "conv2d_3x3_decision:64x64",
        "ad1d28b769e4f14abc2c57b5e069205b3bb04da92505dd1c1115ad0d2758644a",
    ),
    (
        "conv2d_3x3_direct:65x65",
        "1df3ff08be5673910cc8c582960c24872c20bb499f7379708f7cefc4309b32e6",
    ),
    (
        "im2col_3x3:65x65",
        "b79b16cf7c386d7d95ba1caffcbe28c0462039ad9fe14d5aa140882d19d4a712",
    ),
    (
        "conv2d_3x3_decision:65x65",
        "5c31159747fd2b391426d8a65a4cc1db563f8b32b29c65b76a2bb566d056c690",
    ),
];

/// Pinned on the same pre-merge tree.
const WALK_PINS: &[(&str, &str)] = &[
    (
        "ast_walk_preorder:0/8",
        "11295d9687e535d7698850881d76428fa6c3be1a7439d773503ea08cdfa62f88",
    ),
    (
        "ast_walk_postorder_nodes:0/8",
        "2f8604e9b72ae78bb75d3ff13e05e06a5ba81d70f3efb87801abff9e701902e2",
    ),
    (
        "ast_walk_preorder:1/8",
        "e96b7368e16c43972a8e8fb11a28976bc0dcb095c7286ebee3f8ec71c974ffa7",
    ),
    (
        "ast_walk_postorder_nodes:1/8",
        "e4295589e720717345f9c752a29d37a2d228304128dfa04b7041fc0327adbd96",
    ),
    (
        "ast_walk_preorder:1/1",
        "0ff959771f9edeb9bca907d0ffaac7ff3d6872e5e96c03886f0c74dc97ae1ee0",
    ),
    (
        "ast_walk_postorder_nodes:1/1",
        "5fbdd20cdebf42d3c2497a8457415ee43387f83b8450a2f712d4666464587c1c",
    ),
    (
        "ast_walk_preorder:4/8",
        "286e652deaaf4cc0e4bb4cecde6e1a0296f81fb0e0c5a6d1ea5b7677263a465d",
    ),
    (
        "ast_walk_postorder_nodes:4/8",
        "77316f9e432d027843fc1c928b192bd4d15be064a4998242b492523fb4115e36",
    ),
    (
        "ast_walk_preorder:6/8",
        "e338c33a2088656013f475f4b4ce6536936d8683035b30244682106f5f5ef142",
    ),
    (
        "ast_walk_postorder_nodes:6/8",
        "78d8677f9876a05706f2a2d89217e26c87c7e021dc6a6b1367fd84a4a4c04a68",
    ),
    (
        "ast_walk_preorder:8/4",
        "59d319c45744defef232636d975cfe4a9a818e9b4e766163f39eb35d19e5af01",
    ),
    (
        "ast_walk_postorder_nodes:8/4",
        "399d5c021ae004525e4b6468e8b62f09ae451b7d275842309ee01bf655cc177a",
    ),
    (
        "ast_walk_preorder:8/16",
        "b603238a54bc2ad580d757b36497a37ef6a39e6763668055ce43d65fa6c9c108",
    ),
    (
        "ast_walk_postorder_nodes:8/16",
        "a17f38b0a7a164a7558b4ba5d3f70d25762c8f67ce810dced069cc5f5911e12e",
    ),
    (
        "ast_walk_preorder:8/1",
        "f8ddd3966f6b95d1f1bb5ebb7e8e569634f569b6df85b98774232c8f103be471",
    ),
    (
        "ast_walk_postorder_nodes:8/1",
        "5746bc5d86291d0e9189d1ef3def7f125bb89d674fda3799ad0cf904ad3bb966",
    ),
    (
        "ast_walk_postorder:0",
        "87a0886d2539e3ef5a5f41c38cbcec6427af8fdf787388815c38d7b796eab27b",
    ),
    (
        "ast_walk_postorder:1",
        "73ad74681617284bafcebb5314e1734bedc0302fe61b6d324fc015ced7344cb3",
    ),
    (
        "ast_walk_postorder:4",
        "88a0bfd385f1dd59e2badfaaf0b8a51b50b087c7b3d77a4d8966f97afd01163f",
    ),
    (
        "ast_walk_postorder:8",
        "50bd9700487098de330f4796cf14e12ef3a840b76ff229fd3aef4904705ef1d5",
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
