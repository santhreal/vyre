//! Smoke tests for the Rust nano-subset parser.

use vyre_frontend_rust::api::parse_rust_bytes;

#[test]
fn parse_trivial_function() {
    let source = r#"
fn main() {
    let x: i32 = 5;
}
"#;
    let summary = parse_rust_bytes(source.as_bytes()).unwrap();
    assert_eq!(summary.module.functions.len(), 1);
    assert_eq!(summary.module.functions[0].name, "ident@5");
}

#[test]
fn parse_function_with_params() {
    let source = r#"
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
"#;
    let summary = parse_rust_bytes(source.as_bytes()).unwrap();
    assert_eq!(summary.module.functions.len(), 1);
    assert_eq!(summary.module.functions[0].params.len(), 2);
}

#[test]
fn parse_if_else() {
    let source = r#"
fn max(a: i32, b: i32) -> i32 {
    if a < b {
        return b;
    } else {
        return a;
    }
}
"#;
    let summary = parse_rust_bytes(source.as_bytes()).unwrap();
    assert_eq!(summary.module.functions.len(), 1);
}

#[test]
fn parse_borrow() {
    let source = r#"
fn borrow(x: &i32) -> i32 {
    return *x;
}
"#;
    let summary = parse_rust_bytes(source.as_bytes()).unwrap();
    assert_eq!(summary.module.functions.len(), 1);
}
