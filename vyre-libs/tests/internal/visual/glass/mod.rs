use super::*;

#[test]
fn small_radius_falls_back_to_full_res() {
    let params = GlassParams {
        width: 16,
        height: 16,
        blur_radius: 4,
        blur_sigma: 1.5,
        tint_rgba: 0x0D_FFFFFF,
        brightness: 1.0,
        saturation: 0.75,
    };
    let pipeline = glass_stages_half_res("s", "o", "t", "h", "ht", &params);
    match &pipeline {
        GlassHalfResPipeline::FullRes { blur, filter } => {
            assert_eq!(blur.stage_count(), 2);
            assert_eq!(filter.buffers().len(), 1);
        }
        GlassHalfResPipeline::HalfRes { .. } => {
            panic!("radius 4 must select full-res path");
        }
    }
    assert_eq!(pipeline.stage_count(), 3);
    assert_eq!(pipeline.programs().len(), 3);
}

#[test]
fn small_dimensions_fall_back_to_full_res() {
    let params = GlassParams {
        width: 2,
        height: 2,
        blur_radius: 12,
        blur_sigma: 4.0,
        tint_rgba: 0x0D_FFFFFF,
        brightness: 1.0,
        saturation: 0.75,
    };
    let pipeline = glass_stages_half_res("s", "o", "t", "h", "ht", &params);
    assert!(matches!(pipeline, GlassHalfResPipeline::FullRes { .. }));
    assert_eq!(pipeline.stage_count(), 3);
}

#[test]
fn large_radius_builds_half_res_pipeline() {
    let params = GlassParams {
        width: 16,
        height: 16,
        blur_radius: 12,
        blur_sigma: 4.0,
        tint_rgba: 0x0D_FFFFFF,
        brightness: 1.0,
        saturation: 0.75,
    };
    let pipeline =
        glass_stages_half_res("scene", "out", "scratch", "half", "half_scratch", &params);
    match &pipeline {
        GlassHalfResPipeline::HalfRes {
            downsample,
            blur,
            upsample,
            filter,
        } => {
            assert_eq!(downsample.buffers().len(), 2);
            assert_eq!(downsample.buffers()[0].name(), "scene");
            assert_eq!(downsample.buffers()[1].name(), "half");

            assert_eq!(blur.stage_count(), 2);
            assert_eq!(blur.horizontal.buffers()[0].name(), "half");
            assert_eq!(blur.horizontal.buffers()[1].name(), "half_scratch");
            assert_eq!(blur.vertical.buffers()[0].name(), "half_scratch");
            assert_eq!(blur.vertical.buffers()[1].name(), "half");

            assert_eq!(upsample.buffers().len(), 2);
            assert_eq!(upsample.buffers()[0].name(), "half");
            assert_eq!(upsample.buffers()[1].name(), "out");

            assert_eq!(filter.buffers().len(), 1);
            assert_eq!(filter.buffers()[0].name(), "out");
        }
        GlassHalfResPipeline::FullRes { .. } => {
            panic!("radius 12 with 16x16 must select half-res path");
        }
    }
    assert_eq!(pipeline.stage_count(), 5);
    assert_eq!(pipeline.programs().len(), 5);
}
