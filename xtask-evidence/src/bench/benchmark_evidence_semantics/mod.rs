//! Semantic checks for benchmark release evidence.
//!
//! `data` owns the tables the checks are written against and the enums they
//! report. `logic` owns the interpretation. Callers see one module either way.

mod data;
mod logic;

pub(crate) use data::*;
pub(crate) use logic::*;
