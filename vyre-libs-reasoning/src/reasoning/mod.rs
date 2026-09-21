//! Logic, causal reasoning, categorical rewrites, and knowledge compilation.

pub mod adjustment_set_pass_dependency;
#[cfg(test)]
pub(crate) mod dnnf;
pub mod do_calculus_change_impact;
#[cfg(test)]
pub(crate) mod finite_category;
pub mod functorial_pass_composition;
pub mod string_diagram_ir_rewrite;
#[cfg(test)]
pub(crate) mod zx_diagram;
