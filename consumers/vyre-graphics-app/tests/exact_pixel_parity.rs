//! Exact off-screen pixel parity and rasterization regression tests.
//!
//! Asserts exact byte-for-byte outputs for vector paths, text runs, and composite
//! framebuffers against the reference execution oracle.

#![forbid(unsafe_code)]

use vyre_graphics_app::{GraphicsRenderer, SceneGraph};
use vyre_libs::visual::{
    apply_scissor_rect, composite_blend, path_rasterize_segments, text_run_blend, BlendMode,
};
use vyre_reference::reference_eval;
use vyre_reference::value::Value;

#[test]
fn test_exact_path_rasterization_bytes() {
    // 2x2 canvas, 1 segment from (0,0) to (1,0), stroke radius 1, red color over black
    let p_path = path_rasterize_segments("segs", "bg", "out", 2, 2, 1, 1, 0xFF00_00FF);
    let segs = vec![0u32, 0, 1, 0];
    let bg = vec![0xFF00_0000u32; 4];

    let inputs = vec![
        Value::from(vyre_primitives::wire::pack_u32_slice(&segs)),
        Value::from(vyre_primitives::wire::pack_u32_slice(&bg)),
    ];
    let outputs = reference_eval(&p_path, &inputs).expect("reference eval must succeed");

    let expected_pixels = [
        0xFF00_00FFu32, // (0,0) - on path
        0xFF00_00FFu32, // (1,0) - on path
        0xFF00_0000u32, // (0,1) - background
        0xFF00_0000u32, // (1,1) - background
    ];

    let bytes = outputs[0].to_bytes();
    let actual_pixels = vyre_primitives::wire::decode_u32_le_bytes_all(&bytes);
    assert_eq!(
        actual_pixels.as_slice(),
        &expected_pixels,
        "Path rasterization must produce exact expected pixel values"
    );
}

#[test]
fn test_exact_text_run_bytes() {
    // 2x2 canvas, 1 glyph at (0,0) with size 1x1, sampled from atlas (0,0) with alpha 255
    let p_text = text_run_blend("glyphs", 1, "atlas", 2, 2, "bg", "out", 2, 2);
    let glyphs = vec![0u32, 0, 1, 1, 0, 0, 0xFF00_00FF]; // Blue glyph
    let atlas = vec![0xFF00_0000u32, 0, 0, 0];            // 255 alpha at (0,0)
    let bg = vec![0xFF00_0000u32; 4];                     // Black background

    let inputs = vec![
        Value::from(vyre_primitives::wire::pack_u32_slice(&glyphs)),
        Value::from(vyre_primitives::wire::pack_u32_slice(&atlas)),
        Value::from(vyre_primitives::wire::pack_u32_slice(&bg)),
    ];
    let outputs = reference_eval(&p_text, &inputs).expect("reference eval must succeed");

    let expected_pixels = [
        0xFF00_00FFu32, // (0,0) - blended blue glyph
        0xFF00_0000u32, // (1,0) - background
        0xFF00_0000u32, // (0,1) - background
        0xFF00_0000u32, // (1,1) - background
    ];

    let bytes = outputs[0].to_bytes();
    let actual_pixels = vyre_primitives::wire::decode_u32_le_bytes_all(&bytes);
    assert_eq!(
        actual_pixels.as_slice(),
        &expected_pixels,
        "Text run rasterization must produce exact expected pixel values"
    );
}

#[test]
fn test_exact_composite_and_clip_bytes() {
    // 2x2 canvas: blend foreground (red) over background (green) with Add mode, then scissor clip to (1,1,2,2)
    let p_blend = composite_blend("fg", "bg", "out", 4, BlendMode::Add);
    let fg = vec![0xFF00_00FFu32; 4]; // Red
    let bg = vec![0xFF00_FF00u32; 4]; // Green

    let inputs = vec![
        Value::from(vyre_primitives::wire::pack_u32_slice(&fg)),
        Value::from(vyre_primitives::wire::pack_u32_slice(&bg)),
    ];
    let outputs = reference_eval(&p_blend, &inputs).expect("blend eval must succeed");

    let bytes = outputs[0].to_bytes();
    let blended = vyre_primitives::wire::decode_u32_le_bytes_all(&bytes);

    // Red + Green = Yellow (0xFF00_FFFF)
    assert_eq!(blended[0], 0xFF00_FFFF);

    // Scissor clip keeping only pixel at (1,1)
    let p_clip = apply_scissor_rect("in", "out", 2, 2, 1, 1, 2, 2);
    let clip_inputs = vec![Value::from(vyre_primitives::wire::pack_u32_slice(&blended))];
    let clip_out = reference_eval(&p_clip, &clip_inputs).expect("clip eval must succeed");

    let expected_clipped = [
        0x0000_0000u32, // (0,0) - clipped out
        0x0000_0000u32, // (1,0) - clipped out
        0x0000_0000u32, // (0,1) - clipped out
        0xFF00_FFFFu32, // (1,1) - yellow inside scissor
    ];

    let clip_bytes = clip_out[0].to_bytes();
    let actual_pixels = vyre_primitives::wire::decode_u32_le_bytes_all(&clip_bytes);
    assert_eq!(
        actual_pixels.as_slice(),
        &expected_clipped,
        "Clipped composite must produce exact expected pixel values"
    );
}

#[test]
fn test_renderer_full_scene_exact_bytes() {
    let mut scene = SceneGraph::new_empty();
    scene.width = 2;
    scene.height = 2;
    scene.background = vec![0xFF00_0000u32; 4]; // Black
    scene.segments = vec![0, 0, 1, 0];          // Segment at y=0
    scene.stroke_radius = 1;
    scene.stroke_color = 0xFF00_00FF;           // Red stroke
    scene.clip_rect = (0, 0, 2, 2);

    let mut renderer = GraphicsRenderer::new(scene);
    let frame = renderer.render_frame().expect("render_frame must succeed");

    let expected = [
        0xFF00_00FFu32,
        0xFF00_00FFu32,
        0xFF00_0000u32,
        0xFF00_0000u32,
    ];
    assert_eq!(frame.as_slice(), &expected);
}
