//! A reference oracle is reachable by name and never by preference.
//!
//! WHY: `acquire_preferred_dispatch_backend` is the only implicit backend
//! choice in the workspace. If it can return a reference oracle, a host with no
//! working GPU driver runs the user's program on the CPU and reports success, so
//! the failure that should have named a missing device instead shows up as a
//! wrong performance number or nothing at all. The refusal was a doc comment on
//! `acquire_preferred_dispatch_backend`; this is the executable form.
//!
//! This binary links exactly one dispatch-capable backend and that backend is a
//! reference oracle whose factory SUCCEEDS. There is nothing wrong with the
//! host, nothing to probe, and nothing to fail: the only reason to refuse is the
//! `reference_oracle` flag. Deleting the flag check in
//! `registry/acquire.rs` turns this red.
//!
//! Not caught here: a concrete driver crate that computes on the host inside its
//! own `dispatch`. That is a different contract, pinned by
//! `no_backend_crate_links_host_arithmetic.rs`.

use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_driver::backend::{
    acquire, acquire_preferred_dispatch_backend, BackendCapability, BackendPrecedence,
    BackendRegistration,
};
use vyre_driver::{BackendError, DispatchConfig, VyreBackend};
use vyre_foundation::ir::{OpId, Program};

const ORACLE_ID: &str = "fixture-reference-oracle";

struct OracleBackend;

impl vyre_driver::backend::private::Sealed for OracleBackend {}

impl VyreBackend for OracleBackend {
    fn id(&self) -> &'static str {
        ORACLE_ID
    }

    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(Vec::new())
    }
}

fn acquire_oracle() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(OracleBackend))
}

fn no_supported_ops() -> &'static HashSet<OpId> {
    static OPS: LazyLock<HashSet<OpId>> = LazyLock::new(HashSet::new);
    &OPS
}

inventory::submit! {
    BackendRegistration {
        id: ORACLE_ID,
        target_id: vyre_foundation::operation::TargetId::expect_valid(ORACLE_ID),
        payload_format: None,
        reference_oracle: true,
        factory: acquire_oracle,
        supported_ops: no_supported_ops,
        semantic_operations: no_supported_ops,
        target_compiler: None,
        materializer: None,
    }
}

inventory::submit! {
    BackendCapability {
        id: ORACLE_ID,
        dispatches: true,
    }
}

// Rank 0 is the best rank in the table. A reference oracle at the front of the
// precedence order must still lose, because precedence orders eligible
// backends and an oracle is not one.
inventory::submit! {
    BackendPrecedence {
        id: ORACLE_ID,
        rank: 0,
    }
}

#[test]
fn preferred_dispatch_refuses_a_host_whose_only_backend_is_a_reference_oracle() {
    let error = match acquire_preferred_dispatch_backend() {
        Ok(backend) => panic!(
            "Fix: preferred dispatch must refuse a host whose only dispatch-capable backend is a \
             reference oracle. It returned `{}`, which silently runs the program on the CPU.",
            backend.id()
        ),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains(ORACLE_ID),
        "Fix: the refusal must name the backend it skipped, got: {message}"
    );
    assert!(
        message.contains("only reference oracle backend(s) were available"),
        "Fix: the refusal must say the available backends were reference oracles, not report a \
         probe failure, got: {message}"
    );
    assert!(
        message.contains("Fix:"),
        "Fix: the refusal must state the corrective action, got: {message}"
    );
}

#[test]
fn the_same_reference_oracle_is_reachable_by_explicit_id() {
    let backend = acquire(ORACLE_ID).expect(
        "Fix: a reference oracle must stay reachable by name for conformance and parity work; \
         the preference filter is not a ban.",
    );
    assert_eq!(
        backend.id(),
        ORACLE_ID,
        "Fix: explicit acquisition must return the backend that was asked for"
    );
}
