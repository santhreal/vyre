//! Domain 4: Interactive Graphics Whole-Graph Composition.
//!
//! Composes 2D viewport culling, analytical vector path rasterization, arbitrary
//! text run glyph accumulation, scissor clipping, and retained dirty-region atlas
//! patching into one unified, validated, schedule-free [`ProgramGraph`].

use vyre_foundation::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, ProgramGraph, ProgramGraphBuilder,
    ProgramGraphError, ShapeDim, ValueContract, ValueLifetime,
};

use crate::visual::{
    apply_scissor_rect, cull_boxes_2d, dirty_region_patch_rgba, path_rasterize_segments,
    text_run_blend,
};

/// Configuration parameters for the interactive graphics pipeline graph.
#[derive(Debug, Clone)]
pub struct InteractiveGraphicsPipelineParams {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Number of scene bounding boxes to cull.
    pub box_count: u32,
    /// Number of vector line segments to rasterize.
    pub segment_count: u32,
    /// Stroke half-width radius in pixels.
    pub stroke_radius: u32,
    /// Stroke RGBA color.
    pub stroke_color: u32,
    /// Number of glyph instances in the text run.
    pub glyph_count: u32,
    /// Glyph atlas width.
    pub atlas_w: u32,
    /// Glyph atlas height.
    pub atlas_h: u32,
    /// Scissor clip rectangle: `(min_x, min_y, max_x, max_y)`.
    pub clip_rect: (u32, u32, u32, u32),
    /// Patch width for retained dirty region update.
    pub patch_w: u32,
    /// Patch height for retained dirty region update.
    pub patch_h: u32,
    /// Destination offset `(dest_x, dest_y)` for the dirty patch.
    pub patch_dest: (u32, u32),
}

/// Build a validated representative interactive graphics frame [`ProgramGraph`].
///
/// Features exercised:
/// - 2D AABB scene graph culling
/// - Vector path analytical rasterization
/// - Subpixel text run glyph accumulation
/// - Scissor rectangle clipping
/// - Retained dirty-region resource blitting
pub fn build_interactive_graphics_pipeline(
    params: InteractiveGraphicsPipelineParams,
) -> Result<ProgramGraph, ProgramGraphError> {
    let mut builder = ProgramGraphBuilder::new();
    let pixel_count = (params.width * params.height) as u64;

    // 1. Graph Inputs
    let val_scene_boxes = builder.input(
        "scene_boxes",
        DataType::I32,
        vec![ShapeDim::Known((params.box_count * 4) as u64)],
    )?;
    let val_path_segments = builder.input(
        "path_segments",
        DataType::U32,
        vec![ShapeDim::Known((params.segment_count * 4) as u64)],
    )?;
    let val_text_glyphs = builder.input(
        "text_glyphs",
        DataType::U32,
        vec![ShapeDim::Known((params.glyph_count * 7) as u64)],
    )?;
    let val_glyph_atlas = builder.input(
        "glyph_atlas",
        DataType::U32,
        vec![ShapeDim::Known((params.atlas_w * params.atlas_h) as u64)],
    )?;
    let val_background_fb = builder.input(
        "background_fb",
        DataType::U32,
        vec![ShapeDim::Known(pixel_count)],
    )?;
    let val_patch_data = builder.input(
        "patch_data",
        DataType::U32,
        vec![ShapeDim::Known((params.patch_w * params.patch_h) as u64)],
    )?;

    // 2. Stage 1: Bounding Box Culling
    let p_cull = cull_boxes_2d(
        "scene_boxes",
        params.box_count,
        0,
        0,
        params.width as i32,
        params.height as i32,
        "visible_mask",
    );
    let _ = builder.add_node(
        "stage_1_cull",
        p_cull,
        vec![GraphInput {
            buffer: "scene_boxes".into(),
            value: val_scene_boxes,
            contract: ValueContract {
                dtype: DataType::I32,
                shape: vec![ShapeDim::Known((params.box_count * 4) as u64)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        }],
        vec![GraphOutput {
            buffer: "visible_mask".into(),
            name: "visible_mask_out".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(params.box_count as u64)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
            retained_successor_of: None,
        }],
    )?;

    // 3. Stage 2: Vector Path Rasterization
    let p_path = path_rasterize_segments(
        "path_segments",
        "background_fb",
        "layer_paths",
        params.width,
        params.height,
        params.segment_count,
        params.stroke_radius,
        params.stroke_color,
    );
    let (_, path_outs) = builder.add_node(
        "stage_2_path",
        p_path,
        vec![
            GraphInput {
                buffer: "path_segments".into(),
                value: val_path_segments,
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known((params.segment_count * 4) as u64)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
            GraphInput {
                buffer: "background_fb".into(),
                value: val_background_fb,
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(pixel_count)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
        ],
        vec![GraphOutput {
            buffer: "layer_paths".into(),
            name: "layer_paths_out".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(pixel_count)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    )?;

    // 4. Stage 3: Text Run Glyph Accumulation
    let p_text = text_run_blend(
        "text_glyphs",
        params.glyph_count,
        "glyph_atlas",
        params.atlas_w,
        params.atlas_h,
        "layer_paths",
        "layer_composite",
        params.width,
        params.height,
    );
    let (_, text_outs) = builder.add_node(
        "stage_3_text",
        p_text,
        vec![
            GraphInput {
                buffer: "text_glyphs".into(),
                value: val_text_glyphs,
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known((params.glyph_count * 7) as u64)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
            GraphInput {
                buffer: "glyph_atlas".into(),
                value: val_glyph_atlas,
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known((params.atlas_w * params.atlas_h) as u64)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
            GraphInput {
                buffer: "layer_paths".into(),
                value: path_outs[0],
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(pixel_count)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
        ],
        vec![GraphOutput {
            buffer: "layer_composite".into(),
            name: "layer_composite_out".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(pixel_count)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    )?;

    // 5. Stage 4: Scissor Rectangle Clipping
    let (c_min_x, c_min_y, c_max_x, c_max_y) = params.clip_rect;
    let p_clip = apply_scissor_rect(
        "layer_composite",
        "clipped_frame",
        params.width,
        params.height,
        c_min_x,
        c_min_y,
        c_max_x,
        c_max_y,
    );
    let (_, clip_outs) = builder.add_node(
        "stage_4_clip",
        p_clip,
        vec![GraphInput {
            buffer: "layer_composite".into(),
            value: text_outs[0],
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(pixel_count)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        }],
        vec![GraphOutput {
            buffer: "clipped_frame".into(),
            name: "clipped_frame_out".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(pixel_count)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    )?;

    // 6. Stage 5: Retained Resource Dirty Region Patching
    let (p_dx, p_dy) = params.patch_dest;
    let p_patch = dirty_region_patch_rgba(
        "clipped_frame",
        "patch_data",
        params.width,
        params.height,
        params.patch_w,
        params.patch_h,
        p_dx,
        p_dy,
        "final_presentation",
    );
    let _ = builder.add_node(
        "stage_5_patch",
        p_patch,
        vec![
            GraphInput {
                buffer: "clipped_frame".into(),
                value: clip_outs[0],
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(pixel_count)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
            GraphInput {
                buffer: "patch_data".into(),
                value: val_patch_data,
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known((params.patch_w * params.patch_h) as u64)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
        ],
        vec![GraphOutput {
            buffer: "final_presentation".into(),
            name: "final_presentation_out".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(pixel_count)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
            retained_successor_of: None,
        }],
    )?;

    builder.build()
}
