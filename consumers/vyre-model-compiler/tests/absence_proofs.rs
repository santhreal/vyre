//! Proves the absence of physical kernels, target-specific source, schedule templates,
//! schedule hints, and compiler rewrites in the downstream model-compiler package.

use std::fs;
use std::path::PathBuf;

#[test]
fn proves_absence_of_physical_kernels_and_target_source() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");

    let banned_patterns = [
        "__global__",
        "__device__",
        "ptx",
        "nvvm",
        "spirv",
        "metal_stdlib",
        "wgsl",
        ".target sm_",
        ".visible .entry",
        "asm!",
        "core::arch::nvptx",
        "core::arch::x86",
    ];

    for entry in fs::read_dir(&src_dir).expect("read src dir") {
        let entry = entry.expect("valid entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            let content = fs::read_to_string(&path).expect("read source file");
            for pattern in banned_patterns {
                assert!(
                    !content.to_lowercase().contains(&pattern.to_lowercase()),
                    "Found banned physical kernel / target pattern '{pattern}' in {}",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn proves_absence_of_schedule_templates_and_hints() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");

    let banned_schedule_terms = [
        "ScheduleHint",
        "ScheduleTemplate",
        "set_tile_size",
        "set_workgroup_size",
        "force_fusion",
        "prevent_fusion",
        "unroll_factor",
    ];

    for entry in fs::read_dir(&src_dir).expect("read src dir") {
        let entry = entry.expect("valid entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            let content = fs::read_to_string(&path).expect("read source file");
            for term in banned_schedule_terms {
                assert!(
                    !content.contains(term),
                    "Found banned schedule hint / template '{term}' in {}",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn proves_absence_of_compiler_rewrites_and_passes() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");

    let banned_rewrite_terms = [
        "OptimizerPass",
        "apply_rewrite",
        "canonicalize_dag",
        "dead_code_elimination",
        "algebraic_simplification",
        "constant_fold_pass",
    ];

    for entry in fs::read_dir(&src_dir).expect("read src dir") {
        let entry = entry.expect("valid entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            let content = fs::read_to_string(&path).expect("read source file");
            for term in banned_rewrite_terms {
                assert!(
                    !content.contains(term),
                    "Found compiler rewrite / optimizer pass '{term}' in {}",
                    path.display()
                );
            }
        }
    }
}
