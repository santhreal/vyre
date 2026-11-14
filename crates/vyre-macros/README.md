# vyre-macros

Procedural macros for compile-time pass and AST registry declaration in vyre.

## Purpose

- Implements `#[vyre_pass]` and `#[vyre_ast_registry]` procedural macros for compile-time metadata and registration generation.
- Expands attribute declarations at compile time without runtime reflection overhead.

## Boundaries

- `proc-macro = true` crate depending on zero workspace crates (`proc-macro2`, `quote`, `syn` only).
- Contains syntax expansion only; must not contain IR types, validation logic, runtime state, or I/O.
- Depended on by `vyre-foundation` (rank 0 `foundation` layer).
- Must never depend on any workspace crate in production dependencies.
