//! Procedural macros for the [`vyre`](https://docs.rs/vyre) GPU compute IR
//! compiler.
//!
//! This crate is compile-time only. Downstream users import from
//! `vyre::optimizer::vyre_pass` rather than depending on this crate directly.
//!
//! The macro surface contains the foundation-owned AST registry generator and
//! the canonical semantic optimizer pass registration attribute.

mod arg_parsers;
mod ast_registry;
mod pass;

use proc_macro::TokenStream;

/// Generates the declarative IR AST core plus serialization and visitor traits.
#[proc_macro]
pub fn vyre_ast_registry(item: TokenStream) -> TokenStream {
    ast_registry::vyre_ast_registry_impl(item)
}
/// Register a unit struct as a `vyre::optimizer::ProgramPass`.
#[proc_macro_attribute]
pub fn vyre_pass(args: TokenStream, item: TokenStream) -> TokenStream {
    pass::vyre_pass_impl(args, item)
}
#[cfg(test)]
mod tests;
