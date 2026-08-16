//! Bind-result log loading contracts over the public `vyre_debug` surface.

use vyre_emit_naga::BindResultEntry;
use vyre_debug::*;

#[test]
fn load_bind_result_log_missing_file_returns_open_error() {
    let r = load_bind_result_log("/nonexistent/path/bind_log.jsonl");
    assert!(
        matches!(r, Err(BindResultLogError::Open(_))),
        "expected Err(Open(_)) for missing file, got {:?}",
        r.err().map(|e| format!("{e:?}"))
    );
}

#[test]
fn load_bind_result_log_valid_entries_parses_all() {
    // Build a minimal valid BindResultEntry JSON line.
    let entry = BindResultEntry {
        vyre_op_id: 7,
        op_kind: "Literal".to_string(),
        init_handle: 42,
        init_scalar_kind: Some("Uint".to_string()),
        child_body_depth: 0,
        value_types_at_call: None,
        publish_path: "root/op7".to_string(),
        local_allocated_ty: None,
    };
    let line = serde_json::to_string(&entry).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bind.jsonl");
    std::fs::write(&path, format!("{line}\n")).unwrap();

    let result = load_bind_result_log(path.to_str().unwrap()).unwrap();
    assert_eq!(result.len(), 1, "expected exactly 1 entry");
    assert_eq!(result[0].vyre_op_id, 7);
    assert_eq!(result[0].op_kind, "Literal");
    assert_eq!(result[0].init_handle, 42);
    assert_eq!(result[0].publish_path, "root/op7");
}

#[test]
fn load_bind_result_log_malformed_json_line_returns_parse_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.jsonl");
    // First line is valid, second is malformed.
    let entry = BindResultEntry {
        vyre_op_id: 1,
        op_kind: "Load".to_string(),
        init_handle: 0,
        init_scalar_kind: None,
        child_body_depth: 0,
        value_types_at_call: None,
        publish_path: "p".to_string(),
        local_allocated_ty: None,
    };
    let valid_line = serde_json::to_string(&entry).unwrap();
    std::fs::write(&path, format!("{valid_line}\nnot valid json\n")).unwrap();

    let r = load_bind_result_log(path.to_str().unwrap());
    assert!(
        matches!(r, Err(BindResultLogError::Parse(2, _))),
        "expected Err(Parse(2, _)) for malformed line 2, got {:?}",
        r.err().map(|e| format!("{e:?}"))
    );
}
