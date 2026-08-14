# vyre threat model

Applies to Vyre 0.7.2.

Vyre is GPU compute infrastructure. Consumers include inference
servers, edge AI platforms, multi-tenant GPU services, and CI
pipelines that execute untrusted ML models. This document specifies
the attacker capabilities vyre defends against and the invariants
the defense rests on.

## Attackers in scope

- **Untrusted IR submitter.** A consumer of a vyre-hosted service
  uploads a `vyre::Program`. Attacker can construct arbitrary valid
  (validator-passing) IR.
- **Malformed-wire submitter.** Attacker sends arbitrary bytes
  claiming to be a vyre wire envelope (`VYRE` magic). Decoder must not
  panic, must not loop, must return a structured error with bounded work.
- **Resource-exhaustion attacker.** Submits IR intended to consume
  maximum GPU memory, time, or pipeline-cache slots.
- **Cross-tenant attacker.** In a multi-tenant deploy, attacks via
  shared state (buffer pool, pipeline cache) to read or corrupt
  another tenant's data.

## Out of scope

- **Physical side channels.** Power analysis, electromagnetic
  leakage, thermal inference. Defense is deployment-level
  (physical security, rate limiting, not GPU architecture).
- **GPU driver exploits.** wgpu / naga / Vulkan driver bugs are
  upstream concerns; vyre reports them but does not patch them.
- **Trusted-host attacks.** If the machine running vyre is
  compromised, vyre cannot defend the model weights or results.

## Invariants vyre defends

| Invariant | Defense mechanism |
| --- | --- |
| **No panic on any input byte stream** | Wire decoder returns structured `Error::WireFormatValidation`; fuzz corpus at `fuzz/corpus/<target>/` regression-tests every historic crash |
| **Bounded wire-format decode work** | Length prefixes capped at `MAX_ARGS * 1024` bytes; depth bounded via `Reader::depth` with recursion guard |
| **No program executes without passing validation** | `WgpuBackend::validate_with_cache` runs on every dispatch; per-backend capability cache ensures cross-backend programs re-validate |
| **No unbounded resource allocation** | Buffer pool power-of-two sized with `MAX_RETAINED_BYTES` cap; pipeline cache bounded at 256 entries with LRU eviction; validation cache bounded at 1024 entries |
| **Cross-tenant isolation** | Each `WgpuBackend` instance owns its own device, pipeline cache, buffer pool; the MPS-style shared singleton (`cached_device`) is test-only per its doc-warning |
| **Output byte-identity across backends** | `conform` proof runs every op × every backend; byte-divergence is a hard error |
| **Dispatch deadline honored** | `DispatchConfig.timeout` triggers structured cancellation; `tracing::warn` event for observability |

## Open source-change findings

- **Cancellation is post-hoc.** The `timeout` check runs after the
  GPU returns; true mid-flight cancellation requires queue-drain
  plumbing.
- **Cross-tenant bind-group reuse through `moka` cache.** Two
  tenants with identical buffer contents would share a bind-group;
  this is a perf optimization, not a correctness hole, but a
  hardened multi-tenant deploy should wrap vyre with a per-tenant
  backend instance.
- **Fuzz corpus seed coverage.** `fuzz/corpus/<target>/` exists per
  SKILL.md; KAT seeding requires source/test data changes.

## Disclosure

See `SECURITY.md` for the coordinated-disclosure process. Embargo
period: 90 days from acknowledgment; extensions negotiated case-by-
case.

## Regression discipline

Every confirmed vulnerability:

1. Gets a `FINDING-VYRE-<n>` entry, with a blocker and a fix plan, in the
   owning crate's findings register. `vyre-libs/findings.toml` is the only
   register that exists today; a crate that needs one creates it beside its
   manifest under the same name.
2. Has the minimized input that reproduces it committed as a fixture the
   regression test reads, under `<crate>/tests/fixtures/`.
3. Has a regression test under `<crate>/tests/`, named for
   the class it closes, that fails on the pre-fix commit and passes on the
   post-fix commit.
4. Has a fragment in `release/changes/unreleased.toml`, which is what the
   changelog is generated from.

No finding is "documented and left open" without an explicit CEO-approval
record in its crate's findings register. LAW 9 applies.
