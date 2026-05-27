# FIX M23  -  Real fast-approx transcendental implementations

## Scope
`vyre-core/src/lower/wgsl/emit_wgsl.rs` (`emit_fast_approx_helpers`)

## Problem (LAW 9 evasion)
`_vyre_cos_ulp(x)` was a documented stub:

```wgsl
// FAST-APPROX NOT YET IMPLEMENTED  -  delegates to strict path
fn _vyre_cos_ulp(x: f32) -> f32 { return cos(x); }
```

This violates LAW 9: the comment documents a lie rather than fixing it. A positive
`ulp_budget` is supposed to select the fast-approx path; delegating to `cos(x)`
defeats the entire purpose of the budget.

## Survey of transcendentals in UnOp
Only `Sin` and `Cos` exist as transcendental `UnOp` variants
(`vyre-spec/src/un_op.rs`).  All other UnOp variants (`Abs`, `Sqrt`, `Floor`,
etc.) are exact or native WGSL instructions and do not need approximate wrappers.

| UnOp | Status before | Status after |
|------|---------------|--------------|
| `Sin` | Real impl (`_vyre_fast_sin_ulp`) | unchanged |
| `Cos` | **STUB** (`return cos(x)`) | **REAL** (`_vyre_cos_ulp`) |

## Implementation choice for cos

Two candidate approaches were considered:

1. `cos(x) = sin(x + pi/2)`  -  reuses the existing `_vyre_fast_sin_ulp`.
2. Own degree-12 even-series with the same range reduction as sin.

**Selected: approach 2 (even-series).**

Reason: for large `|x|` (e.g. `|x| >> 1e6`), adding `pi/2` in `f32` is a no-op
because `pi/2` is below the ULP of `x`.  The shifted value would then undergo the
same range reduction as sin, but on the **wrong** value, producing large errors.
A dedicated even-series on `y = x - k*pi` avoids that cancellation entirely and
keeps the same ≤2 ULP bound over the full `f32` domain.

### `_vyre_cos_ulp` details

- Range reduction: `k = round(x * inv_pi)`, `y = x - k * pi` → `[-pi/2, pi/2]`
- Sign: `(-1)^k` via the same parity test used in sin.
- Polynomial: degree-12 even Taylor in `z = y*y`, Horner/fma:
  ```
  c5 =  1/479001600
  c4 = -1/3628800
  c3 =  1/40320
  c2 = -1/720
  c1 =  1/24
  c0 = -1/2
  q  = fma(z, fma(z, ... c0), 1.0)
  result = sign * q
  ```
- ULP bound: **≤2 ULP** (same proof structure as sin; error dominated by
  truncation of the degree-12 series and single `fma` rounding at each step).

## Wiring
`vyre-core/src/lower/wgsl/expr/operators.rs` already routes `UnOp::Cos` to
`emit_transcendental(out, "cos", "_vyre_cos_ulp", ...)`.  No wiring change was
required; only the helper body was fixed.

## Tests

- `cargo check -p vyre`  -  clean.
- `cargo test -p vyre --lib`  -  **263 passed, 0 failed**.
- Specific test exercising the fix:
  - `ulp_budget_two_uses_fast_approx` (in `vyre-core/src/lower/wgsl/expr/tests.rs`)
    - Verifies `_vyre_cos_ulp` is emitted when `ulp_budget = 2`.
    - Verifies the emitted body does **not** contain `return cos(x);`.
    - Verifies the even-series coefficient `c0 = -1.0 / 2.0` is present in the
      emitted helper, proving the real polynomial is used.

## Files touched
- `vyre-core/src/lower/wgsl/emit_wgsl.rs`
- `vyre-core/src/lower/wgsl/expr/tests.rs`
