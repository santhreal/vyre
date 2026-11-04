# tests/SKILL.md, vyre-macros

One test file per contract. A file is named for the contract it proves, and
the directory has no catch-all target. `docs/testing/TESTING.toml` holds the
workspace-level default command, hardware expectation and failure behavior
for every package.

## Purpose

`vyre-macros` provides two procedural macros. `vyre_ast_registry!` generates
the declarative IR AST core with its serialization and visitor traits.
`#[vyre_pass]` registers a unit struct as an optimizer `ProgramPass`. It
builds against `proc-macro2`, `quote` and `syn` and no workspace crate, so
nothing it emits can depend on a type it can see. The whole contract is
compile-time, so most of the suite is compile behavior rather than runtime
behavior.

## Critical invariants

- A malformed invocation is a compile error with an actionable message. The
  19 cases under `tests/ui` pin the exact diagnostic in a `.stderr` file, and
  the ones that name a correction carry a `Fix:` sentence. Changing a
  diagnostic is a test change, not a silent edit.
- Expansion is usable from a crate that renames itself. The `integration` and
  `pass_matrix` targets expand under `extern crate self as vyre`, which is
  how a generated path resolves inside `vyre-foundation`.
- A registered pass carries its declared metadata into the generated
  registration. `pass_matrix` and `generated_metadata_matrix` prove the
  argument values reach the emitted `ProgramPass`.

## Adversarial surface

The `tests/ui` cases cover the rejection classes:

- Duplicate enum or duplicate variant in an `vyre_ast_registry!` body.
- `#[vyre_pass]` on an enum, a named struct or a tuple struct, where only a
  unit struct is legal.
- A missing `name` argument, an unknown argument, or a duplicated argument.
- A duplicate entry in `requires`, `invalidates` or `requires_caps`.
- An out-of-vocabulary `phase`, `analyze_mode`, `boundary_class` or
  `cost_model_family`.
- A `requires` entry that is not a string, and a `preserves_abi` that is not
  a bool.

## Cross-crate contracts

- `vyre_ast_registry!` is consumed by `vyre-foundation` for the `Node` and
  `Expr` declarations.
- `#[vyre_pass]` is consumed by the `vyre-foundation` optimizer for pass
  registration.

## Bench targets

The crate declares no bench target. Expansion happens during compilation, so
the cost shows up in the compile time of the crates that invoke the macros,
not in a runtime harness.

## Fuzz targets

The crate declares no fuzz target. The rejection classes above are enumerated
rather than sampled, because the input is a token stream with a fixed
argument vocabulary and the vocabulary is what has to be total.

## What NOT to test here

- Runtime semantics of the generated IR. Those live in `vyre-foundation`
  tests.
- Pass scheduling. That lives in `vyre-foundation` tests.

## Running

```bash
./cargo_full test -p vyre-macros
./cargo_full test -p vyre-macros --test adversarial
./cargo_full test -p vyre-macros --test integration
```
