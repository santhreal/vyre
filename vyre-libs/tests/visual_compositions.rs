//! Comprehensive test suite for vyre-libs visual compositions.
//!
//! Tests validate:
//! - **Identity transforms**: confirm no-op when params are neutral
//! - **Program structure**: verify buffer declarations and region tagging
//! - **Edge cases**: zero-radius, 1-pixel, max-radius
//! - **Algebraic properties**: energy conservation, symmetry, commutativity
//! - **Pixel math correctness**: fixed-point arithmetic, clamp boundaries

#![allow(deprecated)]

#[cfg(feature = "visual")]
#[path = "contract_cases/visual_compositions__cell_grid.rs"]
mod visual_compositions_cell_grid;
#[cfg(feature = "visual")]
#[path = "contract_cases/visual_compositions__default_params.rs"]
mod visual_compositions_default_params;
#[cfg(feature = "visual")]
#[path = "contract_cases/visual_compositions__glyph_grid.rs"]
mod visual_compositions_glyph_grid;
#[cfg(feature = "visual")]
#[path = "contract_cases/visual_compositions__program_has_correct_buffers.rs"]
mod visual_compositions_program_has_correct_buffers;
#[cfg(feature = "visual")]
#[test]
fn test_all_visual_program_builders_construct_valid_programs() {
    use vyre_libs::visual::*;

    let p_path = path_rasterize_segments("segs", "bg", "out", 4, 4, 1, 1, 0xFF00_00FF);
    assert_eq!(p_path.buffers().len(), 3);

    let p_text = text_run_blend("glyphs", 1, "atlas", 4, 4, "bg", "out", 4, 4);
    assert_eq!(p_text.buffers().len(), 4);

    let p_gray = rgba_to_grayscale("in", "out", 16);
    assert_eq!(p_gray.buffers().len(), 2);

    let p_rgba = grayscale_to_rgba("in", "out", 16);
    assert_eq!(p_rgba.buffers().len(), 2);

    let p_pm = premultiply_alpha("in", "out", 16);
    assert_eq!(p_pm.buffers().len(), 2);

    let p_unpm = unpremultiply_alpha("in", "out", 16);
    assert_eq!(p_unpm.buffers().len(), 2);

    let p_resample = bilinear_resample_rgba("in", 2, 2, "out", 4, 4);
    assert_eq!(p_resample.buffers().len(), 2);

    let p_scissor = apply_scissor_rect("in", "out", 4, 4, 1, 1, 3, 3);
    assert_eq!(p_scissor.buffers().len(), 2);

    let p_clip_mask = apply_clip_mask("in", "mask", "out", 16);
    assert_eq!(p_clip_mask.buffers().len(), 3);

    let p_blend = composite_blend("fg", "bg", "out", 16, BlendMode::SrcOver);
    assert_eq!(p_blend.buffers().len(), 3);

    let p_cull = cull_boxes_2d("boxes", 4, 0, 0, 100, 100, "mask");
    assert_eq!(p_cull.buffers().len(), 2);

    let p_scan = layout_prefix_scan_u32("sizes", "offsets", 16);
    assert_eq!(p_scan.buffers().len(), 2);

    let p_reduce = reduce_bounding_boxes_2d("boxes", 4, "bbox");
    assert_eq!(p_reduce.buffers().len(), 2);

    let p_patch_rgba = dirty_region_patch_rgba("atlas", "patch", 16, 16, 4, 4, 2, 2, "out");
    assert_eq!(p_patch_rgba.buffers().len(), 3);

    let p_patch_direct = dirty_region_patch_direct("patch", 4, 4, 2, 2, "target", 16, 16);
    assert_eq!(p_patch_direct.buffers().len(), 2);
}
