//! Independent downstream interactive graphical application.
//!
//! Consumes only the public Vyre compiler seam and neutral `vyre-libs` compositions.
//! Exercises text/path rasterization, image filtering, clipping, compositing, culling,
//! layout scans, and retained resource updates under interactive constraints.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::time::Instant;

use vyre::compiler::{
    compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre_foundation::ir::{GraphValueId, ValueLifetime};
use vyre_libs::graph_compositions::{
    build_interactive_graphics_pipeline, InteractiveGraphicsPipelineParams,
};
use vyre_libs::visual::{
    apply_scissor_rect, cull_boxes_2d, dirty_region_patch_rgba, path_rasterize_segments,
    text_run_blend,
};
use vyre_reference::reference_eval;
use vyre_reference::value::Value;
/// Interactive user or system event.
#[derive(Debug, Clone)]
pub enum InteractiveEvent {
    /// Window/viewport resize.
    Resize { width: u32, height: u32 },
    /// Pointer movement / hover.
    PointerMove { x: u32, y: u32 },
    /// Pointer click / interaction.
    PointerClick { x: u32, y: u32 },
    /// Text input / glyph mutation.
    TextEdit { text: String, font_size: u32 },
    /// Localized dirty region update.
    DirtyRegionUpdate {
        patch_w: u32,
        patch_h: u32,
        dest_x: u32,
        dest_y: u32,
        pixels: Vec<u32>,
    },
    /// Burst of rapid input events.
    BurstInput { count: usize },
    /// Simulated sudden device loss.
    DeviceLoss,
    /// Memory pressure condition.
    MemoryPressure { allocation_mb: usize },
    /// Background compute interference.
    BackgroundInterference { job_count: usize },
}

/// Retained scene graph holding renderable items.
#[derive(Debug, Clone)]
pub struct SceneGraph {
    /// Frame dimensions.
    pub width: u32,
    pub height: u32,
    /// Bounding boxes for culling `[x0, y0, x1, y1]`.
    pub boxes: Vec<i32>,
    /// Path segments `[x0, y0, x1, y1]`.
    pub segments: Vec<u32>,
    /// Path stroke half-width radius.
    pub stroke_radius: u32,
    /// Path stroke color packed RGBA.
    pub stroke_color: u32,
    /// Text glyph instances `[gx, gy, gw, gh, atlas_u, atlas_v, color]`.
    pub glyphs: Vec<u32>,
    /// Glyph atlas texture buffer.
    pub glyph_atlas: Vec<u32>,
    pub atlas_w: u32,
    pub atlas_h: u32,
    /// Background framebuffer pixels.
    pub background: Vec<u32>,
    /// Scissor clip rectangle `(min_x, min_y, max_x, max_y)`.
    pub clip_rect: (u32, u32, u32, u32),
    /// Retained dirty patch data.
    pub patch: Vec<u32>,
    pub patch_w: u32,
    pub patch_h: u32,
    pub patch_dest: (u32, u32),
}

impl SceneGraph {
    /// Create an empty or tiny scene.
    #[must_use]
    pub fn new_empty() -> Self {
        Self {
            width: 1,
            height: 1,
            boxes: Vec::new(),
            segments: Vec::new(),
            stroke_radius: 1,
            stroke_color: 0,
            glyphs: Vec::new(),
            glyph_atlas: vec![0],
            atlas_w: 1,
            atlas_h: 1,
            background: vec![0],
            clip_rect: (0, 0, 1, 1),
            patch: Vec::new(),
            patch_w: 0,
            patch_h: 0,
            patch_dest: (0, 0),
        }
    }

    /// Create a standard interactive UI scene.
    #[must_use]
    pub fn new_ui_scene(width: u32, height: u32) -> Self {
        let pixel_count = (width * height) as usize;
        let mut background = vec![0xFF20_2020u32; pixel_count]; // Dark gray background

        // Draw a decorative background gradient or tint
        for y in 0..height {
            for x in 0..width {
                let idx = (x + y * width) as usize;
                let tint = (x * 255 / width.max(1)) & 0x3F;
                background[idx] = 0xFF20_2020u32 | tint;
            }
        }

        // Scene bounding boxes: a header bar, sidebar, and 10 content cards
        let mut boxes = vec![
            0, 0, width as i32, 40,             // Header
            0, 40, 200, height as i32,           // Sidebar
        ];
        for i in 0..10 {
            let cx = 220 + (i % 3) * 150;
            let cy = 60 + (i / 3) * 100;
            boxes.extend_from_slice(&[cx, cy, cx + 130, cy + 80]);
        }

        // Vector path segments: border lines and icon curves
        let segments = vec![
            0, 40, width, 40,                    // Header bottom border
            200, 40, 200, height,                // Sidebar right border
            250, 80, 300, 120,                   // Line 1
            300, 120, 350, 80,                   // Line 2
        ];

        // Glyph atlas: 16x16 with mock glyphs
        let atlas_w = 16u32;
        let atlas_h = 16u32;
        let mut glyph_atlas = vec![0u32; (atlas_w * atlas_h) as usize];
        // Populate atlas with some solid/semi-transparent pixels
        for y in 0..8 {
            for x in 0..8 {
                glyph_atlas[(x + y * atlas_w) as usize] = 0xFF00_0000;
            }
        }

        // Text glyphs: "Vyre Graphics UI"
        let glyphs = vec![
            10, 10, 8, 8, 0, 0, 0xFF00_FFFF,     // Yellow 'V'
            20, 10, 8, 8, 0, 0, 0xFF00_FFFF,     // 'y'
            30, 10, 8, 8, 0, 0, 0xFF00_FFFF,     // 'r'
            40, 10, 8, 8, 0, 0, 0xFF00_FFFF,     // 'e'
        ];

        // Patch: 4x4 cursor/badge at (100, 10)
        let patch_w = 4u32;
        let patch_h = 4u32;
        let patch = vec![0xFF00_00FFu32; 16]; // Red badge

        Self {
            width,
            height,
            boxes,
            segments,
            stroke_radius: 1,
            stroke_color: 0xFF00_FF00, // Green strokes
            glyphs,
            glyph_atlas,
            atlas_w,
            atlas_h,
            background,
            clip_rect: (0, 0, width, height),
            patch,
            patch_w,
            patch_h,
            patch_dest: (100, 10),
        }
    }

    /// Create a large scene with thousands of draw elements.
    #[must_use]
    pub fn new_large_scene(width: u32, height: u32, count: usize) -> Self {
        let mut scene = Self::new_ui_scene(width, height);
        scene.boxes.clear();
        scene.segments.clear();
        for i in 0..count {
            let x = (i * 17 % width.max(1) as usize) as i32;
            let y = (i * 31 % height.max(1) as usize) as i32;
            scene.boxes.extend_from_slice(&[x, y, x + 20, y + 20]);
            if i < 20 {
                scene.segments.extend_from_slice(&[
                    x as u32,
                    y as u32,
                    (x + 10) as u32,
                    (y + 10) as u32,
                ]);
            }
        }
        scene
    }
}

/// Graphics renderer managing retained session state, compilation, and event dispatch.
#[derive(Debug)]
pub struct GraphicsRenderer {
    pub scene: SceneGraph,
    pub is_device_lost: bool,
    pub device_recovery_count: usize,
    pub retained_framebuffer: Vec<u32>,
    pub total_frames_rendered: usize,
}

impl GraphicsRenderer {
    /// Create a new renderer for a scene.
    #[must_use]
    pub fn new(scene: SceneGraph) -> Self {
        let pixel_count = (scene.width * scene.height) as usize;
        Self {
            retained_framebuffer: vec![0; pixel_count],
            scene,
            is_device_lost: false,
            device_recovery_count: 0,
            total_frames_rendered: 0,
        }
    }

    /// Handle an incoming interactive event.
    pub fn handle_event(&mut self, event: InteractiveEvent) -> Result<(), String> {
        match event {
            InteractiveEvent::Resize { width, height } => {
                self.scene.width = width;
                self.scene.height = height;
                self.scene.background.resize((width * height) as usize, 0xFF20_2020);
                self.retained_framebuffer.resize((width * height) as usize, 0);
                self.scene.clip_rect = (0, 0, width, height);
            }
            InteractiveEvent::PointerMove { x, y } => {
                self.scene.patch_dest = (x.min(self.scene.width.saturating_sub(self.scene.patch_w)), y.min(self.scene.height.saturating_sub(self.scene.patch_h)));
            }
            InteractiveEvent::PointerClick { x, y } => {
                // Insert a click ripple / highlight segment
                self.scene.segments.extend_from_slice(&[x, y, x + 5, y + 5]);
            }
            InteractiveEvent::TextEdit { text, font_size } => {
                self.scene.glyphs.clear();
                let mut gx = 10u32;
                for _c in text.chars() {
                    self.scene.glyphs.extend_from_slice(&[
                        gx,
                        10,
                        font_size,
                        font_size,
                        0,
                        0,
                        0xFF00_FFFF,
                    ]);
                    gx += font_size + 2;
                }
            }
            InteractiveEvent::DirtyRegionUpdate {
                patch_w,
                patch_h,
                dest_x,
                dest_y,
                pixels,
            } => {
                self.scene.patch_w = patch_w;
                self.scene.patch_h = patch_h;
                self.scene.patch_dest = (dest_x, dest_y);
                self.scene.patch = pixels;
            }
            InteractiveEvent::BurstInput { count } => {
                for i in 0..count {
                    let offset = (i as u32) % 50;
                    self.scene.patch_dest = (100 + offset, 10 + offset);
                }
            }
            InteractiveEvent::DeviceLoss => {
                self.is_device_lost = true;
            }
            InteractiveEvent::MemoryPressure { allocation_mb } => {
                let _temp = vec![0u8; allocation_mb * 1024 * 1024];
            }
            InteractiveEvent::BackgroundInterference { job_count } => {
                // Simulate background compute kernels
                for _ in 0..job_count {
                    let mut dummy = [1u32, 2, 3, 4];
                    for x in &mut dummy {
                        *x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    }
                }
            }
        }
        Ok(())
    }

    /// Render current scene off-screen and return the resulting pixel buffer.
    pub fn render_frame(&mut self) -> Result<Vec<u32>, String> {
        if self.is_device_lost {
            // Recover from device loss
            self.recover_device_loss()?;
        }

        // Build execution parameters
        let box_count = (self.scene.boxes.len() / 4) as u32;
        let segment_count = (self.scene.segments.len() / 4) as u32;
        let glyph_count = (self.scene.glyphs.len() / 7) as u32;

        let _p_cull = if box_count > 0 {
            cull_boxes_2d(
                "boxes",
                box_count,
                0,
                0,
                self.scene.width as i32,
                self.scene.height as i32,
                "mask",
            )
        } else {
            cull_boxes_2d("boxes", 1, 0, 0, 1, 1, "mask")
        };

        // Evaluate rasterization stages via reference execution
        let mut framebuffer = self.scene.background.clone();

        // 1. Path rasterization
        if segment_count > 0 {
            let p_path = path_rasterize_segments(
                "segments",
                "bg",
                "out",
                self.scene.width,
                self.scene.height,
                segment_count,
                self.scene.stroke_radius,
                self.scene.stroke_color,
            );
            let inputs = vec![
                Value::from(vyre_primitives::wire::pack_u32_slice(&self.scene.segments)),
                Value::from(vyre_primitives::wire::pack_u32_slice(&framebuffer)),
            ];
            let out = reference_eval(&p_path, &inputs)
                .map_err(|e| format!("Path rasterization failed: {e:?}"))?;
            if let Some(val) = out.first() {
                let bytes = val.to_bytes();
                framebuffer = vyre_primitives::wire::decode_u32_le_bytes_all(&bytes);
            }
        }

        // 2. Text run rasterization
        if glyph_count > 0 {
            let p_text = text_run_blend(
                "glyphs",
                glyph_count,
                "atlas",
                self.scene.atlas_w,
                self.scene.atlas_h,
                "bg",
                "out",
                self.scene.width,
                self.scene.height,
            );
            let inputs = vec![
                Value::from(vyre_primitives::wire::pack_u32_slice(&self.scene.glyphs)),
                Value::from(vyre_primitives::wire::pack_u32_slice(&self.scene.glyph_atlas)),
                Value::from(vyre_primitives::wire::pack_u32_slice(&framebuffer)),
            ];
            let out = reference_eval(&p_text, &inputs)
                .map_err(|e| format!("Text rasterization failed: {e:?}"))?;
            if let Some(val) = out.first() {
                let bytes = val.to_bytes();
                framebuffer = vyre_primitives::wire::decode_u32_le_bytes_all(&bytes);
            }
        }

        // 3. Scissor clipping
        let (c_min_x, c_min_y, c_max_x, c_max_y) = self.scene.clip_rect;
        let p_clip = apply_scissor_rect(
            "in",
            "out",
            self.scene.width,
            self.scene.height,
            c_min_x,
            c_min_y,
            c_max_x,
            c_max_y,
        );
        let inputs = vec![Value::from(vyre_primitives::wire::pack_u32_slice(&framebuffer))];
        let out = reference_eval(&p_clip, &inputs)
            .map_err(|e| format!("Clipping failed: {e:?}"))?;
        if let Some(val) = out.first() {
            let bytes = val.to_bytes();
            framebuffer = vyre_primitives::wire::decode_u32_le_bytes_all(&bytes);
        }

        // 4. Dirty patch update
        if self.scene.patch_w > 0 && self.scene.patch_h > 0 && !self.scene.patch.is_empty() {
            let (p_dx, p_dy) = self.scene.patch_dest;
            let p_patch = dirty_region_patch_rgba(
                "atlas",
                "patch",
                self.scene.width,
                self.scene.height,
                self.scene.patch_w,
                self.scene.patch_h,
                p_dx,
                p_dy,
                "out",
            );
            let inputs = vec![
                Value::from(vyre_primitives::wire::pack_u32_slice(&framebuffer)),
                Value::from(vyre_primitives::wire::pack_u32_slice(&self.scene.patch)),
            ];
            let out = reference_eval(&p_patch, &inputs)
                .map_err(|e| format!("Dirty region patch failed: {e:?}"))?;
            if let Some(val) = out.first() {
                let bytes = val.to_bytes();
                framebuffer = vyre_primitives::wire::decode_u32_le_bytes_all(&bytes);
            }
        }

        self.retained_framebuffer = framebuffer.clone();
        self.total_frames_rendered += 1;
        Ok(framebuffer)
    }

    /// Recover from simulated device loss by resetting state and invalidating retained resources.
    pub fn recover_device_loss(&mut self) -> Result<(), String> {
        self.is_device_lost = false;
        self.device_recovery_count += 1;
        // Re-allocate / re-initialize retained framebuffer
        let pixel_count = (self.scene.width * self.scene.height) as usize;
        self.retained_framebuffer = vec![0; pixel_count];
        Ok(())
    }
}

/// Latency and performance benchmark report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkReport {
    pub total_samples: usize,
    pub input_latency_ns_p50: u64,
    pub input_latency_ns_p90: u64,
    pub input_latency_ns_p95: u64,
    pub input_latency_ns_p99: u64,
    pub input_latency_ns_p999: u64,
    pub frame_time_ns_p50: u64,
    pub frame_time_ns_p90: u64,
    pub frame_time_ns_p95: u64,
    pub frame_time_ns_p99: u64,
    pub frame_time_jitter_ns: f64,
    pub cpu_submission_time_ns: u64,
    pub compile_pipeline_time_ns: u64,
    pub missed_deadlines_60hz: usize,
    pub missed_deadlines_120hz: usize,
    pub missed_deadlines_144hz: usize,
    pub missed_deadlines_240hz: usize,
    pub empty_work_latency_ns: u64,
    pub device_loss_recovery_latency_ns: u64,
    pub total_memory_bytes: usize,
}

/// Benchmark harness for the interactive graphical application.
pub struct BenchmarkHarness;

impl BenchmarkHarness {
    /// Run the full interactive graphics measurement suite.
    pub fn run_suite(samples: usize) -> BenchmarkReport {
        let mut renderer = GraphicsRenderer::new(SceneGraph::new_ui_scene(64, 64));

        // 1. Measure compile/pipeline time
        let compile_start = Instant::now();
        let params = InteractiveGraphicsPipelineParams {
            width: 32,
            height: 32,
            box_count: 8,
            segment_count: 4,
            stroke_radius: 1,
            stroke_color: 0xFF00_00FF,
            glyph_count: 4,
            atlas_w: 16,
            atlas_h: 16,
            clip_rect: (0, 0, 32, 32),
            patch_w: 4,
            patch_h: 4,
            patch_dest: (2, 2),
        };
        let graph = build_interactive_graphics_pipeline(params).expect("graph build");
        let mut facts = ExternalFacts::new(Digest([42; 32]), BTreeMap::new());
        for (v_id, v) in graph.values().iter().enumerate() {
            if v.contract.lifetime == ValueLifetime::Constant {
                facts.constant_identities.insert(GraphValueId(v_id as u32), Digest([42; 32]));
            }
        }
        let request = CompileRequest::new(
            graph,
            facts,
            DeviceFacts::unknown(),
            SearchBudget::new(1, 1, 1, 0, 100_000),
            CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
        )
        .validate()
        .expect("validate");
        let _artifact = compile(&request).expect("compile");
        let compile_pipeline_time = compile_start.elapsed();

        // 2. Measure empty work latency
        let mut empty_renderer = GraphicsRenderer::new(SceneGraph::new_empty());
        let empty_start = Instant::now();
        let _ = empty_renderer.render_frame();
        let empty_work_latency = empty_start.elapsed();

        // 3. Measure device loss recovery latency
        let dev_loss_start = Instant::now();
        renderer.handle_event(InteractiveEvent::DeviceLoss).unwrap();
        let _ = renderer.render_frame();
        let dev_loss_recovery_latency = dev_loss_start.elapsed();

        // 4. Sample interactive event-to-frame latencies
        let mut latencies = Vec::with_capacity(samples);
        let mut frame_times = Vec::with_capacity(samples);
        let mut cpu_sub_times = Vec::with_capacity(samples);

        for i in 0..samples {
            let event = if i % 10 == 0 {
                InteractiveEvent::Resize {
                    width: 32 + (i as u32 % 32),
                    height: 32 + (i as u32 % 32),
                }
            } else if i % 3 == 0 {
                InteractiveEvent::TextEdit {
                    text: format!("Text {i}"),
                    font_size: 8,
                }
            } else {
                InteractiveEvent::PointerMove {
                    x: (i as u32 * 7) % 32,
                    y: (i as u32 * 11) % 32,
                }
            };

            let t0 = Instant::now();
            let sub_t0 = Instant::now();
            renderer.handle_event(event).unwrap();
            let sub_time = sub_t0.elapsed();
            cpu_sub_times.push(sub_time.as_nanos() as u64);

            let frame_t0 = Instant::now();
            let _ = renderer.render_frame();
            let frame_time = frame_t0.elapsed();
            let total_latency = t0.elapsed();

            latencies.push(total_latency.as_nanos() as u64);
            frame_times.push(frame_time.as_nanos() as u64);
        }

        latencies.sort_unstable();
        frame_times.sort_unstable();

        let p50_idx = samples * 50 / 100;
        let p90_idx = samples * 90 / 100;
        let p95_idx = samples * 95 / 100;
        let p99_idx = samples * 99 / 100;
        let p999_idx = (samples * 999 / 1000).min(samples.saturating_sub(1));

        // Jitter (std dev)
        let mean_frame = frame_times.iter().sum::<u64>() as f64 / samples as f64;
        let variance = frame_times
            .iter()
            .map(|&t| (t as f64 - mean_frame).powi(2))
            .sum::<f64>()
            / samples as f64;
        let jitter = variance.sqrt();

        // Deadlines: 60Hz = 16.67ms, 120Hz = 8.33ms, 144Hz = 6.94ms, 240Hz = 4.17ms
        let d60 = 16_666_667u64;
        let d120 = 8_333_333u64;
        let d144 = 6_944_444u64;
        let d240 = 4_166_667u64;

        let missed_60 = frame_times.iter().filter(|&&t| t > d60).count();
        let missed_120 = frame_times.iter().filter(|&&t| t > d120).count();
        let missed_144 = frame_times.iter().filter(|&&t| t > d144).count();
        let missed_240 = frame_times.iter().filter(|&&t| t > d240).count();

        BenchmarkReport {
            total_samples: samples,
            input_latency_ns_p50: latencies[p50_idx],
            input_latency_ns_p90: latencies[p90_idx],
            input_latency_ns_p95: latencies[p95_idx],
            input_latency_ns_p99: latencies[p99_idx],
            input_latency_ns_p999: latencies[p999_idx],
            frame_time_ns_p50: frame_times[p50_idx],
            frame_time_ns_p90: frame_times[p90_idx],
            frame_time_ns_p95: frame_times[p95_idx],
            frame_time_ns_p99: frame_times[p99_idx],
            frame_time_jitter_ns: jitter,
            cpu_submission_time_ns: cpu_sub_times.iter().sum::<u64>() / samples as u64,
            compile_pipeline_time_ns: compile_pipeline_time.as_nanos() as u64,
            missed_deadlines_60hz: missed_60,
            missed_deadlines_120hz: missed_120,
            missed_deadlines_144hz: missed_144,
            missed_deadlines_240hz: missed_240,
            empty_work_latency_ns: empty_work_latency.as_nanos() as u64,
            device_loss_recovery_latency_ns: dev_loss_recovery_latency.as_nanos() as u64,
            total_memory_bytes: renderer.retained_framebuffer.len() * 4,
        }
    }
}
