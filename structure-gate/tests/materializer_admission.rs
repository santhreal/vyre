//! Contract: target-payload admission has exactly one implementation.
//!
//! WHY: the neutral half of materialization was copied into every concrete
//! backend, and the copies drifted. Two backends required a module entry point
//! named `main` and two accepted anything; one spelled shared rejections in its
//! own words. A payload one backend refused, another executed. Nothing failed
//! when that happened, which is the actual defect this file closes.
//!
//! The rule is checked against the live workspace and the member list is read
//! from the root manifest at run time, so a new `vyre-driver-*` backend that
//! hand-rolls admission turns this red without anyone editing a list here.
//!
//! What this does NOT catch: a backend that calls `materialize::admit` and then
//! re-checks the same properties itself. The behavioural contract for what
//! `admit` decides lives in
//! `vyre-driver-spirv/tests/target_payload_admission_contract.rs`.

use structure_gate::{materializer_admission_failures, scan, workspace_root};

/// WHY: a private `invalid_module` or `compile_error` is how the four copies
/// diverged in the first place.
#[test]
fn a_backend_that_redefines_a_shared_helper_is_a_violation() {
    let source = r#"
        fn invalid_module(reason: &str) -> BackendError { todo() }
        fn materialize(&self) { materialize::admit(a, p, t) }
    "#;
    let failures = materializer_admission_failures(&[(
        "vyre-driver-x/src/materializer.rs".into(),
        source.into(),
    )]);
    assert_eq!(failures.len(), 1, "expected one failure, got {failures:?}");
    assert!(
        failures[0].contains("defines its own `invalid_module`"),
        "failure must name the redefined helper, got `{}`",
        failures[0]
    );
}

/// WHY: the drift did not need a redefined helper. A backend that simply never
/// routes through the choke point decides admission by itself.
#[test]
fn a_backend_that_never_calls_admit_is_a_violation() {
    let source = "fn materialize(&self) { self.open_payload_my_own_way() }";
    let failures = materializer_admission_failures(&[(
        "vyre-driver-y/src/materializer.rs".into(),
        source.into(),
    )]);
    assert_eq!(failures.len(), 1, "expected one failure, got {failures:?}");
    assert!(
        failures[0].contains("does not admit its target payload"),
        "failure must name the missing choke point, got `{}`",
        failures[0]
    );
}

/// WHY: the rule must accept the shape every backend is supposed to have, or it
/// would be satisfied only by deleting materializers.
#[test]
fn a_backend_that_routes_through_admit_is_clean() {
    let source = r#"
        let admitted = materialize::admit(artifact, payload, target)?;
        return Err(materialize::invalid_module("bad dialect image"));
    "#;
    assert!(
        materializer_admission_failures(&[(
            "vyre-driver-z/src/materializer.rs".into(),
            source.into()
        )])
        .is_empty(),
        "a backend delegating to the shared module must pass"
    );
}

/// WHY: this is the live contract. It is the assertion that would have gone red
/// while the four copies existed, and it stays red if one comes back.
#[test]
fn every_concrete_backend_admits_through_the_shared_module() {
    let root = workspace_root();
    let workspace = scan(&root);
    assert!(
        !workspace.materializers.is_empty(),
        "no `vyre-driver-*` materializer was scanned; the rule would pass vacuously"
    );
    let failures = materializer_admission_failures(&workspace.materializers);
    assert!(
        failures.is_empty(),
        "target-payload admission must have one implementation:\n{}",
        failures.join("\n")
    );
}
