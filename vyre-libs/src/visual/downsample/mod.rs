//! 2× box-filter downsample for half-resolution blur.
//!
//! Averages each 2×2 block of pixels into one output pixel.
//! Category A composition  -  pure IR. No Tier 2.5 primitives.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre-libs::visual::downsample";

/// Build a Program that 2× downsamples `input` into `output`.
///
/// - `input`:  `[u32; width * height]`  -  source pixels (packed RGBA)
/// - `output`: `[u32; (width/2) * (height/2)]`  -  downsampled result
/// - Width and height must be even.
#[must_use]
pub fn downsample_2x(input: &str, output: &str, width: u32, height: u32) -> Program {
    let out_w = width / 2;
    let out_h = height / 2;
    let input_count = width.saturating_mul(height);
    let output_count = out_w.saturating_mul(out_h);
    let (oy, ox) = crate::builder::stencil::decompose_index(&Expr::var("idx"), out_w);
    let [p00_idx, p10_idx, p01_idx, p11_idx] =
        crate::builder::stencil::downsample_2x_source_indices(
            &Expr::var("oy"),
            &Expr::var("ox"),
            width,
        );

    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_count),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(output_count),
        ],
        super::PIXEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(output_count)),
                    vec![
                        Node::let_bind("ox", ox),
                        Node::let_bind("oy", oy),
                        // Load 4 source pixels.
                        Node::let_bind("p00", Expr::load(input, p00_idx)),
                        Node::let_bind("p10", Expr::load(input, p10_idx)),
                        Node::let_bind("p01", Expr::load(input, p01_idx)),
                        Node::let_bind("p11", Expr::load(input, p11_idx)),
                        // Average each channel: (c0+c1+c2+c3+2) >> 2
                        // R channel
                        Node::let_bind(
                            "r",
                            Expr::shr(
                                Expr::add(
                                    Expr::add(
                                        Expr::add(
                                            Expr::bitand(Expr::var("p00"), Expr::u32(0xFF)),
                                            Expr::bitand(Expr::var("p10"), Expr::u32(0xFF)),
                                        ),
                                        Expr::add(
                                            Expr::bitand(Expr::var("p01"), Expr::u32(0xFF)),
                                            Expr::bitand(Expr::var("p11"), Expr::u32(0xFF)),
                                        ),
                                    ),
                                    Expr::u32(2),
                                ),
                                Expr::u32(2),
                            ),
                        ),
                        // G channel
                        Node::let_bind(
                            "g",
                            Expr::shr(
                                Expr::add(
                                    Expr::add(
                                        Expr::add(
                                            Expr::bitand(
                                                Expr::shr(Expr::var("p00"), Expr::u32(8)),
                                                Expr::u32(0xFF),
                                            ),
                                            Expr::bitand(
                                                Expr::shr(Expr::var("p10"), Expr::u32(8)),
                                                Expr::u32(0xFF),
                                            ),
                                        ),
                                        Expr::add(
                                            Expr::bitand(
                                                Expr::shr(Expr::var("p01"), Expr::u32(8)),
                                                Expr::u32(0xFF),
                                            ),
                                            Expr::bitand(
                                                Expr::shr(Expr::var("p11"), Expr::u32(8)),
                                                Expr::u32(0xFF),
                                            ),
                                        ),
                                    ),
                                    Expr::u32(2),
                                ),
                                Expr::u32(2),
                            ),
                        ),
                        // B channel
                        Node::let_bind(
                            "b",
                            Expr::shr(
                                Expr::add(
                                    Expr::add(
                                        Expr::add(
                                            Expr::bitand(
                                                Expr::shr(Expr::var("p00"), Expr::u32(16)),
                                                Expr::u32(0xFF),
                                            ),
                                            Expr::bitand(
                                                Expr::shr(Expr::var("p10"), Expr::u32(16)),
                                                Expr::u32(0xFF),
                                            ),
                                        ),
                                        Expr::add(
                                            Expr::bitand(
                                                Expr::shr(Expr::var("p01"), Expr::u32(16)),
                                                Expr::u32(0xFF),
                                            ),
                                            Expr::bitand(
                                                Expr::shr(Expr::var("p11"), Expr::u32(16)),
                                                Expr::u32(0xFF),
                                            ),
                                        ),
                                    ),
                                    Expr::u32(2),
                                ),
                                Expr::u32(2),
                            ),
                        ),
                        // A channel
                        Node::let_bind(
                            "a",
                            Expr::shr(
                                Expr::add(
                                    Expr::add(
                                        Expr::add(
                                            Expr::shr(Expr::var("p00"), Expr::u32(24)),
                                            Expr::shr(Expr::var("p10"), Expr::u32(24)),
                                        ),
                                        Expr::add(
                                            Expr::shr(Expr::var("p01"), Expr::u32(24)),
                                            Expr::shr(Expr::var("p11"), Expr::u32(24)),
                                        ),
                                    ),
                                    Expr::u32(2),
                                ),
                                Expr::u32(2),
                            ),
                        ),
                        // Pack RGBA.
                        Node::let_bind(
                            "packed",
                            crate::builder::stencil::pack_rgba(
                                Expr::var("r"),
                                Expr::var("g"),
                                Expr::var("b"),
                                Expr::var("a"),
                            ),
                        ),
                        // Write output.
                        Node::let_bind(
                            "oidx",
                            crate::builder::stencil::flat_index(
                                Expr::var("oy"),
                                out_w,
                                Expr::var("ox"),
                            ),
                        ),
                        Node::store(output, Expr::var("oidx"), Expr::var("packed")),
                    ],
                ),
            ],
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || downsample_2x("input", "output", 4, 4),
        Some(|| {
            // 4×4 all-white → 2×2 all-white
            let input = vec![0xFFFF_FFFFu32; 16];
            vec![vec![
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&input),
                vec![0u8; 16],
            ]]
        }),
        Some(|| {
            let expected = vec![0xFFFF_FFFFu32; 4];
            vec![vec![crate::visual::u32_word_bytes::u32_words_to_le_bytes(&expected)]]
        }),
    )
    .with_category("visual")
}
