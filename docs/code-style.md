# Vyre Code Style

Applies to Vyre 0.7.2.

Use the surrounding module as the first example. Run `cargo fmt --all` after a
coherent edit batch.

## Modules and files

- Name Rust modules in `snake_case`.
- Give each module one responsibility and one clear owner.
- Group peers under a plural module when that improves navigation.
- Keep domain logic independent of CLI, transport, and UI layers.
- Use explicit re-exports. Do not add wildcard public re-exports.
- Extend an existing canonical helper before adding a second implementation.

A cohesive file may be long. Split when responsibilities or ownership diverge,
not at an arbitrary line count.

## Public APIs

- Use `CamelCase` for types and traits, and `snake_case` for functions.
- Make non-public items `pub(crate)` or private.
- Treat every exported signature and behavior as a compatibility contract.
- Add public items through the owning crate's existing facade.
- Run the public API snapshot gate when an exported surface changes.

## Errors

Return a concrete error type at a public boundary. Include the failed operation,
relevant identity, and a concrete fix. Do not log secret inputs. Do not convert
an unsupported backend or device request into a silent fallback.

## Tests

Put new behavior tests under the owning crate's `tests/` directory. Update an
existing inline test only when the changed contract already lives there.

Each regression test:

1. names the observable behavior,
2. carries a doc comment explaining the bug it prevents,
3. asserts exact values or state transitions,
4. includes the relevant negative or adversarial boundary,
5. runs deterministically in the full suite.

## Performance work

Measure the real path before and after the change. Use
`docs/optimization/BENCH_TARGETS.toml` for release targets and keep backend
counter collection inside the owning concrete driver or benchmark crate.

## Comments

Comment the constraint or reason, not the syntax. Do not leave TODO, FIXME,
placeholder, or deferral markers in shipped code.
