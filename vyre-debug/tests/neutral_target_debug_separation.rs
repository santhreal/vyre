//! Contract tests for neutral vs target-specific debug capability separation.
//!
//! Verifies Section 186.1:
//! - Descriptor dumps, diffs, source assignments, and carrier inspection operate
//!   without any concrete emitter or GPU runtime dependency.
//! - Naga/WGSL tracing is cleanly separated as an explicit target capability.
//! - Emitter crates do not depend on `vyre-debug` and binary emitters are inspected
//!   without being forced through a text-disassembly interface.

use vyre_debug::{
    carrier_summary, diff_descriptors, dump_descriptor, find_dangling_refs,
    neutral_debug_capabilities, source_assignments, DebugCapabilityKind,
    DescriptorDumpOptions, DEBUG_CAPABILITIES,
};
use vyre_lower::{BindingLayout, KernelBody, KernelDescriptor};

fn sample_descriptor() -> KernelDescriptor {
    KernelDescriptor {
        id: "debug_sample_kernel".to_string(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: vyre_lower::Dispatch {
            workgroup_size: [32, 1, 1],
        },
        body: KernelBody {
            ops: vec![],
            literals: vec![],
            child_bodies: vec![],
        },
    }
}

#[test]
fn neutral_capabilities_execute_without_emitter() {
    let desc_a = sample_descriptor();
    let desc_b = sample_descriptor();

    // 1. Descriptor dump
    let dump = dump_descriptor(&desc_a, &DescriptorDumpOptions::default());
    assert!(dump.text.contains("KernelDescriptor id="));
    assert!(dump.text.contains("dispatch: workgroup_size=[32, 1, 1]"));

    // 2. Descriptor diff
    let diff = diff_descriptors(&desc_a, &desc_b);
    assert!(diff.bindings_dropped.is_empty());

    // 3. Carrier summary
    let carriers = carrier_summary(&desc_a);
    assert_eq!(carriers.total_ops_observed, 0);

    // 4. Dangling refs
    let dangling = find_dangling_refs(&desc_a);
    assert!(dangling.is_empty());

    // 5. Source assignments
    let program = vyre_foundation::ir::Program::wrapped(vec![], [1, 1, 1], vec![]);
    let mut count = 0;
    source_assignments::walk_source_assigns(&program, |_name, _loop_path| {
        count += 1;
    });
    assert_eq!(count, 0);
}

#[test]
fn capability_registry_clearly_demarcates_target_inspection() {
    let neutral_caps = neutral_debug_capabilities();
    assert!(neutral_caps.len() >= 5);

    for cap in &neutral_caps {
        assert!(
            !cap.requires_target_emitter,
            "Fix: neutral capability `{}` must not require target emitter",
            cap.name
        );
        assert!(
            cap.is_binary_safe,
            "Fix: neutral capability `{}` must be binary safe",
            cap.name
        );
    }

    // Verify Naga/WGSL capabilities are explicitly marked as requiring target emitter
    let wgsl_caps: Vec<_> = DEBUG_CAPABILITIES
        .iter()
        .filter(|c| matches!(c.kind, DebugCapabilityKind::TargetNagaWgsl))
        .collect();

    assert!(!wgsl_caps.is_empty());
    for cap in wgsl_caps {
        assert!(
            cap.requires_target_emitter,
            "Fix: WGSL capability `{}` must require target emitter",
            cap.name
        );
    }
}
