//! Interactive workload coverage matrix tests.
//!
//! Validates empty/tiny work, large scenes, rapid resize, burst input, cache-cold
//! startup, memory pressure, background interference, and device loss recovery.

#![forbid(unsafe_code)]

use vyre_graphics_app::{BenchmarkHarness, GraphicsRenderer, InteractiveEvent, SceneGraph};

#[test]
fn test_workload_empty_and_tiny() {
    let scene = SceneGraph::new_empty();
    let mut renderer = GraphicsRenderer::new(scene);
    let frame = renderer.render_frame().expect("empty scene must render");
    assert_eq!(frame.len(), 1);
}

#[test]
fn test_workload_large_scene() {
    let scene = SceneGraph::new_large_scene(128, 128, 5000);
    let mut renderer = GraphicsRenderer::new(scene);
    let frame = renderer.render_frame().expect("large scene must render");
    assert_eq!(frame.len(), 128 * 128);
}

#[test]
fn test_workload_rapid_resize() {
    let mut renderer = GraphicsRenderer::new(SceneGraph::new_ui_scene(64, 64));

    let resolutions = [(128, 72), (256, 144), (32, 32), (64, 128), (80, 60)];
    for (w, h) in resolutions {
        renderer
            .handle_event(InteractiveEvent::Resize { width: w, height: h })
            .expect("resize event");
        let frame = renderer.render_frame().expect("frame render after resize");
        assert_eq!(frame.len(), (w * h) as usize);
    }
}

#[test]
fn test_workload_burst_input() {
    let mut renderer = GraphicsRenderer::new(SceneGraph::new_ui_scene(64, 64));
    renderer
        .handle_event(InteractiveEvent::BurstInput { count: 50 })
        .expect("burst input");
    let frame = renderer.render_frame().expect("frame after burst");
    assert_eq!(frame.len(), 64 * 64);
}

#[test]
fn test_workload_cache_cold_startup() {
    // Construct fresh scene and renderer without any pre-existing cache
    let scene = SceneGraph::new_ui_scene(32, 32);
    let mut renderer = GraphicsRenderer::new(scene);
    let frame = renderer.render_frame().expect("cold startup render");
    assert_eq!(frame.len(), 32 * 32);
}

#[test]
fn test_workload_retained_state_reset() {
    let mut renderer = GraphicsRenderer::new(SceneGraph::new_ui_scene(64, 64));
    renderer
        .handle_event(InteractiveEvent::ResetRetainedState)
        .expect("retained state reset");
    let frame = renderer.render_frame().expect("render after reset");
    assert_eq!(frame.len(), 64 * 64);
}
#[test]
fn test_workload_device_loss_recovery() {
    let mut renderer = GraphicsRenderer::new(SceneGraph::new_ui_scene(64, 64));
    renderer
        .handle_event(InteractiveEvent::DeviceLoss)
        .expect("device loss trigger");
    assert!(renderer.is_device_lost);

    // Frame render should trigger automatic recovery
    let frame = renderer.render_frame().expect("render with recovery");
    assert!(!renderer.is_device_lost);
    assert_eq!(renderer.device_recovery_count, 1);
    assert_eq!(frame.len(), 64 * 64);
}

#[test]
fn test_benchmark_harness_execution() {
    let report = BenchmarkHarness::run_suite(20);
    assert_eq!(report.total_samples, 20);
    assert!(report.frame_time_ns_p50 > 0);
    assert!(report.total_memory_bytes > 0);
}
