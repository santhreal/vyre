//! Contracts for reference modules whose tests reach only the public API.
//!
//! WHY: these suites lived beside the code they exercise but touched nothing
//! private, so they compiled into the library on every test build and could not
//! catch a symbol dropped from the public surface. One integration target keeps
//! them link-cheap and pins them to the API a consumer sees.

mod dialect_dispatch;
mod ieee754;
mod subgroup;
mod value;
