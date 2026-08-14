//! Registry-observing contracts, in one link unit.
//!
//! Every integration target in this crate links the whole operation registry,
//! its primitive catalog and every backend driver, so a second target costs a
//! second link of that surface. Both modules below judge the same live registry
//! and share the target instead.

#![forbid(unsafe_code)]

mod cli_docs;
mod operation_schema;
