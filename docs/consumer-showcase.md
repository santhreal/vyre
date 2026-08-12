# Named external integration

Last verified: 2026-08-04

Applies to Vyre 0.7.2.

Use [`docs/consumer-integration.md`](consumer-integration.md) for the generic
downstream-analyzer contract. It is the local authority for the platform
boundary: consumers depend on Vyre; Vyre must not depend back on consumers.

## Keyhog

[Keyhog](https://github.com/santhreal/keyhog) is a separately released
integration that uses Vyre. Its repository owns its supported inputs, rule
language, configuration, severity policy, deployment instructions, benchmarks,
and release status.

Use Keyhog when you need a concrete external example of:

- selecting a Vyre backend for scan-style workloads
- composing library operations into product pipelines
- keeping consumer policy outside the Vyre platform crates

Do not treat Keyhog private or product-specific paths as Vyre platform
documentation. Vyre's local conformance and backend evidence prove only the
platform surfaces named by the Vyre release gate.

## Platform boundary checklist for any consumer

1. Depend on published Vyre crates or the workspace path pins your product owns.
2. Keep product rules, severity, and CLI UX in the consumer repo.
3. Use Vyre diagnostics codes for automation; map them to product messages at
   the edge.
4. Never add a Vyre → consumer production dependency. The
   `platform-boundary` gate rejects consumer names in platform docs and code
   comments when configured.
5. If you need a new shared primitive, promote it into `vyre-primitives` or
   `vyre-libs` with the tier rules instead of reaching into private modules.

## Related docs

- [`consumer-integration.md`](consumer-integration.md)
- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`support.md`](support.md)
