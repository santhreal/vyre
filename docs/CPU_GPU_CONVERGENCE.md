# CPU/GPU Convergence Lens  -  Security Ops

Tracks #79 F-A5 (real CPU/GPU convergence lens for security ops).

## The claim

Every security-flavoured op in `vyre-libs::security` must
produce byte-equivalent output on CPU reference interpretation
and GPU dispatch for every input in its witness corpus. No
"ULP-aware" or "transcendental-exempt" shortcut applies  -  these
ops are integer-only (taint flow, dominator tree, path
reconstruct, flows_to), so byte equivalence is the only
acceptable parity.

## Why these rows were exempted historically

The `UniversalDiffExemption` registry carried security ops because the
original graph substrate did not expose enough reusable Tier-2.5
programs to test the security shims directly.

## Status

The live generated catalog is `docs/catalog/security.md`. It currently
records seven security op registrations, with `path_reconstruct`
carrying witness + expected-output fixtures and the remaining dataflow
family still covered by UniversalDiffExemption rows. The source gate
`composition_discipline::every_op_has_test_fixtures_or_is_explicitly_exempt`
does not maintain a separate security exemption list.

The open source-change findings are:

- remove the security UniversalDiffExemption rows when each shim has
  direct witness and expected-output coverage;
- wire a convergence lens for the dataflow family so one-hop graph
  steps are compared under the same fixpoint semantics that callers
  rely on;
- keep the generated catalog and the op registry in sync in the same
  patch.

## Operating rule

Security ops must not re-enter an ad hoc exemption list. An op that
loses its fixture pair must fail the fixture gate until the source patch
restores a real witness and expected-output pair.
