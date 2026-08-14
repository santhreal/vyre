//! Registry/coverage closure for every `vyre-frontend-rust` program builder.
//!
//! WHY: this crate lowers Rust source to IR, and its `pub fn ... -> Program`
//! builders are the lexer substrate every consumer reaches for. A builder here
//! that is neither submitted through `inventory` nor pinned by a test is a
//! lowering nothing executes, so it can drift from the `rustc_lexer` oracle the
//! other targets in this directory hold it to. The contract, the enumerator, and
//! what this gate does not catch are stated once in `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // The lexer plan's `build` methods take a `&self` receiver, so the
    // enumeration counts only the free builders the plan delegates to. That
    // population is small: any floor below the measured count would be a floor
    // of zero here, and zero builders are what a broken scan reports, so the
    // floor is the measured count. Adding a builder never fails this; removing
    // one is a decision that has to be recorded on this line.
    floor: 2,
    // Empty, and it must stay that way by fixing builders rather than listing
    // them: the enumerator's stale and now-covered guards make this list
    // only-shrinkable, so anything added here is a debt with no scheduled payer.
    waiver: [],
}
