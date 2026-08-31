//! Multi-head Latent Attention (MLA).
//!
//! DeepSeek V4 Flash uses MLA with compressed KV cache. The key insight:
//! instead of caching full K and V tensors per head, MLA compresses them
//! into a low-rank latent vector c_t, then projects back at attention time.
//!
//! Formulation (simplified for single-token decode):
//!   c_t = W_DK @ h_t                    (compress, dim = kv_lora_rank)
//!   k_t = W_UK @ c_t + W_KR @ h_t       (decompress K, with decoupled RoPE)
//!   v_t = W_UV @ c_t                    (decompress V)
//!   o_t = softmax(q_t @ K^T / sqrt(d)) @ V
//!
//! The KV cache stores only c_t (and optionally the RoPE-decoupled key
//! component). For long context this reduces cache size by ~93%.
//!
//! Category A composition.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

use super::tiled_online_softmax::{
    scratch_index, tiled_online_softmax_body, TiledOnlineSoftmaxSpec,
};
use crate::nn::attention_stability::{bounded_exp_arg, bounded_score};

/// Buffer names and shape for one [`mla_decode`] build.
struct MlaDecodeSpec<'a> {
    /// Query vectors for the current token, `[num_heads, head_dim]`.
    q: &'a str,
    /// Compressed KV latents for prior tokens, `[seq_len, kv_lora_rank]`.
    kv_cache: &'a str,
    /// Decoupled RoPE keys for prior tokens, `[seq_len, qk_rope_head_dim]`.
    kr_cache: &'a str,
    /// K up-projection, `[kv_lora_rank, num_heads * head_dim]`.
    w_uk: &'a str,
    /// V up-projection, `[kv_lora_rank, num_heads * head_dim]`.
    w_uv: &'a str,
    /// Attention output, `[num_heads, head_dim]`.
    out: &'a str,
    /// Cached token count.
    seq_len: u32,
    /// Attention head count.
    num_heads: u32,
    /// Per-head feature width.
    head_dim: u32,
    /// Latent rank of the compressed KV cache.
    kv_lora_rank: u32,
    /// Leading dimensions carrying the decoupled RoPE key.
    qk_rope_head_dim: u32,
}

/// MLA single-token decode with compressed KV cache.
///
/// This computes one step of autoregressive decode: given the current
/// token's query and the full compressed KV cache, produce the attention
/// output for this token.
///
/// Shapes:
///   `q: [num_heads, head_dim]`  -  query vectors for current token
///   `kv_cache: [seq_len, kv_lora_rank]`  -  compressed KV for all prior tokens
///   `kr_cache: [seq_len, qk_rope_head_dim]`  -  decoupled RoPE keys for prior tokens
///   `w_uk: [kv_lora_rank, num_heads * head_dim]`  -  K up-projection
///   `w_uv: [kv_lora_rank, num_heads * head_dim]`  -  V up-projection
///   `out: [num_heads, head_dim]`  -  attention output
///
/// # Errors
/// Returns `Err` when any dimension is zero.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode(
    q: &str,
    kv_cache: &str,
    kr_cache: &str,
    w_uk: &str,
    w_uv: &str,
    out: &str,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
) -> Result<Program, String> {
    mla_decode_impl(&MlaDecodeSpec {
        q,
        kv_cache,
        kr_cache,
        w_uk,
        w_uv,
        out,
        seq_len,
        num_heads,
        head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
    })
}

/// MLA decode over the shared tiled online-softmax skeleton.
///
/// Only two fragments are MLA-specific: the score pass, which decompresses
/// `k_t` from the latent cache and folds in the decoupled RoPE component, and
/// the accumulator update, which decompresses `v_t`. The `(m, l, o_acc)`
/// recurrence around them comes from
/// [`tiled_online_softmax_body`](super::tiled_online_softmax::tiled_online_softmax_body).
fn mla_decode_impl(spec: &MlaDecodeSpec<'_>) -> Result<Program, String> {
    let MlaDecodeSpec {
        q,
        kv_cache,
        kr_cache,
        w_uk,
        w_uv,
        out,
        seq_len,
        num_heads,
        head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
    } = *spec;
    if seq_len == 0 || num_heads == 0 || head_dim == 0 || kv_lora_rank == 0 || qk_rope_head_dim == 0
    {
        return Err("Fix: mla_decode all dims must be > 0".to_string());
    }

    let workgroup_lanes = 64_u32;
    let tile_size = 64_u32;

    let head_stride = head_dim;
    let uv_stride = num_heads.checked_mul(head_dim).ok_or("overflow")?;

    let q_scratch_count = workgroup_lanes.checked_mul(head_dim).ok_or("overflow")?;
    let score_scratch_count = workgroup_lanes.checked_mul(tile_size).ok_or("overflow")?;
    let o_acc_count = workgroup_lanes.checked_mul(head_dim).ok_or("overflow")?;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let scale_expr = Expr::f32(scale);
    let num_tiles = seq_len.div_ceil(tile_size);

    // Scratch index helpers: each lane gets its own sub-slice.
    let q_idx = |local: Expr, d: Expr| scratch_index(head_dim, local, d);
    let score_idx = |local: Expr, j: Expr| scratch_index(tile_size, local, j);
    let o_idx = |local: Expr, d: Expr| scratch_index(head_dim, local, d);

    // ---- Compute all scores for the current tile ----
    let compute_tile_scores = vec![Node::loop_for(
        "tile_j",
        Expr::u32(0),
        Expr::var("tile_len"),
        vec![
            // decompress k_t on the fly and accumulate dot product
            Node::let_bind("dot_val", Expr::f32(0.0)),
            Node::loop_for(
                "dim",
                Expr::u32(0),
                Expr::u32(head_dim),
                vec![
                    Node::let_bind("k_val", Expr::f32(0.0)),
                    Node::loop_for(
                        "r",
                        Expr::u32(0),
                        Expr::u32(kv_lora_rank),
                        vec![Node::assign(
                            "k_val",
                            Expr::add(
                                Expr::var("k_val"),
                                Expr::mul(
                                    Expr::load(
                                        w_uk,
                                        Expr::add(
                                            Expr::mul(Expr::var("r"), Expr::u32(uv_stride)),
                                            Expr::add(
                                                Expr::mul(
                                                    Expr::var("head"),
                                                    Expr::u32(head_stride),
                                                ),
                                                Expr::var("dim"),
                                            ),
                                        ),
                                    ),
                                    Expr::load(
                                        kv_cache,
                                        Expr::add(
                                            Expr::mul(
                                                Expr::add(
                                                    Expr::var("tile_start"),
                                                    Expr::var("tile_j"),
                                                ),
                                                Expr::u32(kv_lora_rank),
                                            ),
                                            Expr::var("r"),
                                        ),
                                    ),
                                ),
                            ),
                        )],
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var("dim"), Expr::u32(qk_rope_head_dim)),
                        vec![Node::assign(
                            "k_val",
                            Expr::add(
                                Expr::var("k_val"),
                                Expr::load(
                                    kr_cache,
                                    Expr::add(
                                        Expr::mul(
                                            Expr::add(Expr::var("tile_start"), Expr::var("tile_j")),
                                            Expr::u32(qk_rope_head_dim),
                                        ),
                                        Expr::var("dim"),
                                    ),
                                ),
                            ),
                        )],
                    ),
                    Node::assign(
                        "dot_val",
                        Expr::add(
                            Expr::var("dot_val"),
                            Expr::mul(
                                Expr::load(
                                    "q_scratch",
                                    q_idx(Expr::var("local"), Expr::var("dim")),
                                ),
                                Expr::var("k_val"),
                            ),
                        ),
                    ),
                ],
            ),
            Node::let_bind(
                "raw_score",
                Expr::mul(Expr::var("dot_val"), scale_expr.clone()),
            ),
            Node::let_bind("score", bounded_score(Expr::var("raw_score"))),
            Node::store(
                "score_tile",
                score_idx(Expr::var("local"), Expr::var("tile_j")),
                Expr::var("score"),
            ),
        ],
    )];

    // ---- o_acc[d] = rescale * o_acc[d] + sum_j weight_j * v_t_j[d] ----
    let update_o_acc = vec![
        // rescale existing accumulator
        Node::loop_for(
            "rescale_d",
            Expr::u32(0),
            Expr::u32(head_dim),
            vec![Node::store(
                "o_acc",
                o_idx(Expr::var("local"), Expr::var("rescale_d")),
                Expr::mul(
                    Expr::var("rescale"),
                    Expr::load("o_acc", o_idx(Expr::var("local"), Expr::var("rescale_d"))),
                ),
            )],
        ),
        // iterate tokens in tile
        Node::loop_for(
            "v_j",
            Expr::u32(0),
            Expr::var("tile_len"),
            vec![
                Node::let_bind(
                    "weight",
                    Expr::UnOp {
                        op: UnOp::Exp,
                        operand: Box::new(bounded_exp_arg(Expr::sub(
                            Expr::load(
                                "score_tile",
                                score_idx(Expr::var("local"), Expr::var("v_j")),
                            ),
                            Expr::var("m_new"),
                        ))),
                    },
                ),
                // for each dimension, decompress v and accumulate
                Node::loop_for(
                    "v_dim",
                    Expr::u32(0),
                    Expr::u32(head_dim),
                    vec![
                        Node::let_bind("v_val", Expr::f32(0.0)),
                        Node::loop_for(
                            "r",
                            Expr::u32(0),
                            Expr::u32(kv_lora_rank),
                            vec![Node::assign(
                                "v_val",
                                Expr::add(
                                    Expr::var("v_val"),
                                    Expr::mul(
                                        Expr::load(
                                            w_uv,
                                            Expr::add(
                                                Expr::mul(Expr::var("r"), Expr::u32(uv_stride)),
                                                Expr::add(
                                                    Expr::mul(
                                                        Expr::var("head"),
                                                        Expr::u32(head_stride),
                                                    ),
                                                    Expr::var("v_dim"),
                                                ),
                                            ),
                                        ),
                                        Expr::load(
                                            kv_cache,
                                            Expr::add(
                                                Expr::mul(
                                                    Expr::add(
                                                        Expr::var("tile_start"),
                                                        Expr::var("v_j"),
                                                    ),
                                                    Expr::u32(kv_lora_rank),
                                                ),
                                                Expr::var("r"),
                                            ),
                                        ),
                                    ),
                                ),
                            )],
                        ),
                        Node::store(
                            "o_acc",
                            o_idx(Expr::var("local"), Expr::var("v_dim")),
                            Expr::add(
                                Expr::load("o_acc", o_idx(Expr::var("local"), Expr::var("v_dim"))),
                                Expr::mul(Expr::var("weight"), Expr::var("v_val")),
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ];

    let body = tiled_online_softmax_body(
        TiledOnlineSoftmaxSpec {
            q,
            out,
            item_var: "head",
            item_count: num_heads,
            seq_len,
            head_dim,
            tile_size,
            tile_count: num_tiles,
        },
        compute_tile_scores,
        update_o_acc,
    );

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(num_heads * head_dim),
            BufferDecl::storage(kv_cache, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(seq_len * kv_lora_rank),
            BufferDecl::storage(kr_cache, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(seq_len * qk_rope_head_dim),
            BufferDecl::storage(w_uk, 3, BufferAccess::ReadOnly, DataType::F32)
                .with_count(kv_lora_rank * uv_stride),
            BufferDecl::storage(w_uv, 4, BufferAccess::ReadOnly, DataType::F32)
                .with_count(kv_lora_rank * uv_stride),
            BufferDecl::workgroup("q_scratch", q_scratch_count, DataType::F32),
            BufferDecl::workgroup("score_tile", score_scratch_count, DataType::F32),
            BufferDecl::workgroup("o_acc", o_acc_count, DataType::F32),
            BufferDecl::output(out, 5, DataType::F32).with_count(num_heads * head_dim),
        ],
        [workgroup_lanes, 1, 1],
        vec![wrap_anonymous_region("vyre-libs::nn::mla_decode", body)],
    ))
}

/// MLA KV cache compression: `c_t = W_DK @ h_t`.
///
/// Computes the compressed latent vector for the current token
/// to be appended to the KV cache.
///
/// Shapes:
///   `h: [hidden_dim]`  -  current token hidden state
///   `w_dk: [hidden_dim, kv_lora_rank]`  -  down-projection weights
///   `c_out: [kv_lora_rank]`  -  compressed latent output
pub fn mla_compress_kv(
    h: &str,
    w_dk: &str,
    c_out: &str,
    hidden_dim: u32,
    kv_lora_rank: u32,
) -> Result<Program, String> {
    if hidden_dim == 0 || kv_lora_rank == 0 {
        return Err("Fix: mla_compress_kv all dims must be > 0".to_string());
    }

    let i = Expr::var("i");
    let body = vec![
        Node::let_bind("i", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::u32(kv_lora_rank)),
            vec![
                Node::let_bind("acc", Expr::f32(0.0)),
                Node::loop_for(
                    "j",
                    Expr::u32(0),
                    Expr::u32(hidden_dim),
                    vec![Node::assign(
                        "acc",
                        Expr::add(
                            Expr::var("acc"),
                            Expr::mul(
                                Expr::load(h, Expr::var("j")),
                                Expr::load(
                                    w_dk,
                                    Expr::add(
                                        Expr::mul(Expr::var("j"), Expr::u32(kv_lora_rank)),
                                        i.clone(),
                                    ),
                                ),
                            ),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: c_out.into(),
                    index: i,
                    value: Expr::var("acc"),
                },
            ],
        ),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(h, 0, BufferAccess::ReadOnly, DataType::F32).with_count(hidden_dim),
            BufferDecl::storage(w_dk, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(hidden_dim * kv_lora_rank),
            BufferDecl::output(c_out, 2, DataType::F32).with_count(kv_lora_rank),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::nn::mla_compress_kv",
            body,
        )],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_f32;

    #[test]
    fn mla_compress_kv_identity() {
        let h = [2.0f32, 3.0];
        let w_dk = [1.0f32, 0.0, 0.0, 1.0];
        let program = mla_compress_kv("h", "w_dk", "c", 2, 2).unwrap();
        let c = eval_f32("mla", &program, &[&h[..], &w_dk[..]], 2);
        assert_eq!(c, vec![2.0, 3.0]);
    }

    #[test]
    fn mla_decode_simple() {
        let q = [1.0f32, 0.0];
        let kv_cache = [1.0f32, 0.0];
        let kr_cache = [0.0f32, 0.0];
        let w_uk = [1.0f32, 0.0, 0.0, 1.0];
        let w_uv = [1.0f32, 0.0, 0.0, 1.0];

        let program = mla_decode(
            "q", "kv_cache", "kr_cache", "w_uk", "w_uv", "out", 1, 1, 2, 2, 2,
        )
        .unwrap();

        let out = eval_f32(
            "mla",
            &program,
            &[&q[..], &kv_cache[..], &kr_cache[..], &w_uk[..], &w_uv[..]],
            2,
        );
        assert!(
            (out[0] - 1.0).abs() < 1e-4,
            "mla_decode out[0] = {}",
            out[0]
        );
        assert!((out[1]).abs() < 1e-4, "mla_decode out[1] = {}", out[1]);
    }

    #[test]
    fn mla_decode_two_tokens() {
        // seq_len=2, num_heads=1, head_dim=2
        // q = [1.0, 0.0]
        // kv_cache = [[1,0], [0,1]]
        // w_uk = identity, w_uv = identity, kr_cache = zeros
        // k_0 = [1,0], k_1 = [0,1]
        // score_0 = dot([1,0],[1,0])/sqrt(2) = 1/sqrt(2)
        // score_1 = dot([1,0],[0,1])/sqrt(2) = 0
        // softmax: w0 ≈ 0.67, w1 ≈ 0.33
        // v_0 = [1,0], v_1 = [0,1]
        // out = [0.67, 0.33]
        let q = [1.0f32, 0.0];
        let kv_cache = [1.0f32, 0.0, 0.0, 1.0];
        let kr_cache = [0.0f32; 4];
        let w_uk = [1.0f32, 0.0, 0.0, 1.0];
        let w_uv = [1.0f32, 0.0, 0.0, 1.0];

        let program = mla_decode(
            "q", "kv_cache", "kr_cache", "w_uk", "w_uv", "out", 2, 1, 2, 2, 2,
        )
        .unwrap();

        let out = eval_f32(
            "mla",
            &program,
            &[&q[..], &kv_cache[..], &kr_cache[..], &w_uk[..], &w_uv[..]],
            2,
        );
        assert!(
            out[0] > 0.6 && out[0] < 0.7,
            "mla_decode out[0] = {}",
            out[0]
        );
        assert!(
            out[1] > 0.3 && out[1] < 0.4,
            "mla_decode out[1] = {}",
            out[1]
        );
    }

    #[test]
    fn mla_decode_zero_dim_errors() {
        for (batch, seq, kv_heads, head_dim, latent) in [
            (0, 1, 2, 2, 2),
            (1, 0, 2, 2, 2),
            (1, 1, 0, 2, 2),
            (1, 1, 2, 0, 2),
            (1, 1, 2, 2, 0),
        ] {
            let err = mla_decode(
                "q", "kv", "kr", "w_uk", "w_uv", "out", batch, seq, kv_heads, head_dim, latent,
            )
            .expect_err("zero dim must error");
            assert!(
                err.contains("mla_decode") && err.contains("> 0"),
                "mla_decode zero-dim ({batch},{seq},{kv_heads},{head_dim},{latent}): {err}"
            );
        }
    }
}
