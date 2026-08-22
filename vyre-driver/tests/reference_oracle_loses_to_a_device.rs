//! Precedence orders eligible backends; a reference oracle is not eligible.
//!
//! WHY: the sibling gate `reference_oracle_is_never_implicit.rs` proves the
//! refusal when an oracle is the only choice. That leaves the inverted case: an
//! oracle ranked ahead of a real device. If the oracle were merely last in the
//! precedence order rather than excluded, a rank edit or a new backend with a
//! worse rank would silently move host arithmetic to the front. This binary
//! ranks the oracle at 0, the best rank in the table, and the device at 500.
//!
//! Both factories succeed, so the only thing separating them is the
//! `reference_oracle` flag.

#[macro_use]
mod fixture_backend;

use fixture_backend::FixtureBackend;
use vyre_driver::acquire_preferred_dispatch_backend;
use vyre_driver::{BackendError, VyreBackend};

const ORACLE_ID: &str = "fixture-ranked-oracle";
const DEVICE_ID: &str = "fixture-ranked-device";

fn acquire_oracle() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(FixtureBackend(ORACLE_ID)))
}

fn acquire_device() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(FixtureBackend(DEVICE_ID)))
}

register_dispatchable_backend! {
    id: ORACLE_ID,
    oracle: true,
    rank: 0,
    factory: acquire_oracle,
}

register_dispatchable_backend! {
    id: DEVICE_ID,
    oracle: false,
    rank: 500,
    factory: acquire_device,
}

#[test]
fn a_best_ranked_reference_oracle_still_loses_to_a_worse_ranked_device() {
    let backend = acquire_preferred_dispatch_backend()
        .expect("Fix: preferred dispatch must select the linked non-oracle backend");
    assert_eq!(
        backend.id(),
        DEVICE_ID,
        "Fix: a reference oracle must never be the preferred dispatch target, whatever its \
         precedence rank. Precedence orders ELIGIBLE backends; an oracle is excluded before the \
         order is consulted."
    );
}
