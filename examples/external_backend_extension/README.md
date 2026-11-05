# external_backend_extension

A dispatch backend that lives outside the workspace. It implements
`vyre_driver::VyreBackend`, submits one `BackendRegistration`, one
`BackendCapability` and one `BackendPrecedence`, and `vyre_driver::acquire`
serves it exactly as it serves a driver crate shipped in the workspace.

Program execution stays in `vyre-reference`. This backend translates buffers
and delegates, so it adds no second host interpreter.

## Modules

- `src/lib.rs`: the backend, its registration, capability and precedence
  submissions, the supported-operation sets, and `dispatch_probe`.
- `src/main.rs`: acquires the backend by id through the registry and prints the
  probe output.
- `tests/backend_probe.rs`: the registration reaches `acquire` and the probe
  returns what the reference interpreter computes.

## Entry points

- `BACKEND_ID` and `TARGET_ID`: the identity the registry serves.
- `ExternalBackend`: the `VyreBackend` implementation.
- `dispatch_probe`: builds a program, dispatches it and returns the output
  words.

## Run its tests

This crate is not a workspace member and carries its own lock file.

```sh
cd examples/external_backend_extension
../../cargo_full test --locked
```
