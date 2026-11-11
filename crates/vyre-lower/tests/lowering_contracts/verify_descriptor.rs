//! Descriptor verification entry-point contracts.

use vyre_lower::analyses::AnalysisFacts;
use vyre_lower::descriptor_builder::{body, descriptor, lit};
use vyre_lower::*;

#[test]
fn valid_input_returns_descriptor_directly() {
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![lit(0, 0)],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        },
    };
    let out = verify_descriptor(&desc).unwrap();
    assert_eq!(out, desc);
}

#[test]
fn invalid_input_returns_input_failure() {
    // Descriptor with zero workgroup_size dim  -  caught by verify.
    let desc = descriptor("bad").dispatch(0, 1, 1).body(body()).build();
    let r = verify_descriptor(&desc);
    assert!(matches!(r, Err(VerifyFailure::Input(_))));
}

#[test]
fn full_report_runs_read_only_analyses() {
    let desc = KernelDescriptor {
        id: "fr".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![lit(0, 0), lit(0, 1)],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        },
    };
    let report = full_report(&desc, &AnalysisFacts::none());
    assert_eq!(report.descriptor_id, "fr");
    assert!(report.summary.contains("fr:"));
    assert_eq!(report.histogram.literal, 2);
    assert_eq!(report.perf.kernel_id, "fr");
    assert!(report.verify.is_ok());
    assert_eq!(report.verify_status(), "OK");
    assert!(report.fix_text.is_empty());
    let rendered = format!("{report}");
    assert!(rendered.contains("fr:"));
    assert!(rendered.contains("id fr"));
    assert!(rendered.contains("OK"));
}

#[test]
fn full_report_serializes_to_json() {
    let desc = KernelDescriptor {
        id: "fr".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![lit(0, 0)],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        },
    };
    let report = full_report(&desc, &AnalysisFacts::none());
    assert_eq!(report.descriptor_id, "fr");
    let json = serde_json::to_string(&report).expect("Fix: serialize");
    assert!(json.contains("\"descriptor_id\""));
    assert!(json.contains("\"summary\""));
    assert!(json.contains("\"histogram\""));
    assert!(json.contains("\"perf\""));
    assert!(json.contains("\"verify\""));
    assert!(json.contains("\"fix_text\""));

    // Round-trip back through Deserialize.
    let _back: FullReport = serde_json::from_str(&json).expect("Fix: round-trip");
}

#[test]
fn full_report_format_long_includes_all_sections() {
    let desc = KernelDescriptor {
        id: "fr".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![lit(0, 0)],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        },
    };
    let r = full_report(&desc, &AnalysisFacts::none());
    let long = r.format_long();
    assert!(long.contains("Kernel:"));
    assert!(long.contains("descriptor id: fr"));
    assert!(long.contains("Histogram:"));
    assert!(long.contains("Perf audit:"));
    assert!(long.contains("Verify:"));
    assert!(long.contains("OK"));
}

#[test]
fn full_report_records_verify_fix_text_for_bad_descriptor() {
    let desc = descriptor("bad").dispatch(0, 1, 1).body(body()).build();
    let report = full_report(&desc, &AnalysisFacts::none());
    assert_eq!(report.descriptor_id, "bad");
    assert_eq!(report.verify_status(), "FAIL");
    assert!(
        report
            .fix_text
            .contains("Fix: descriptor verification failed"),
        "Fix: invalid descriptor reports must carry operator-actionable verifier repair text."
    );
    let long = report.format_long();
    assert!(long.contains("Verify:"));
    assert!(long.contains("Fix:"));
}

#[test]
fn errors_accessor_yields_underlying() {
    let desc = descriptor("bad").dispatch(0, 1, 1).body(body()).build();
    let f = verify_descriptor(&desc).unwrap_err();
    assert_ne!(f.errors().len(), 0);
}
