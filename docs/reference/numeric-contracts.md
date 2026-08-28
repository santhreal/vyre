# Numeric contracts

A numeric contract states what a computed value is allowed to be. Every
operation registration carries one, every region of a validated graph derives
one, and schedule legality refuses a transform whose new order produces an error
the contract does not admit.

```rust
use vyre_foundation::numeric::{NumericContract, Reassociation, ScalarFormat};

let contract = NumericContract::of(ScalarFormat::F32)
    .within_ulp(1)
    .reassociating(Reassociation::WithinBudget);
```

## What a contract states

| Field | Meaning |
|---|---|
| `measure` | how far the result may sit from the exact one |
| `reassociation` | whether a transform may reorder the combines |
| `storage` | the format a value is held in between regions |
| `intermediate` | the format one operation computes in |
| `accumulator` | the format a reduction accumulates in |
| `rounding` | the rounding the result is produced under |
| `overflow` | what happens when a value leaves the representable range |
| `nan` | what happens to a NaN |
| `infinity` | what happens to an infinity |
| `subnormal` | what happens to a subnormal |
| `determinism` | whether two runs agree bit for bit |
| `atomic_order` | whether the result depends on atomic landing order |
| `approximation` | whether an approximate native instruction is admitted |

`NumericContract::of` reads rounding, overflow, NaN, infinity and subnormal
behavior from the format's own semantics, so a contract cannot state behavior
its storage format does not have. `check` returns the first field that
disagrees.

## Measures

An `ErrorMeasure` is `Exact`, a count of units in the last place, an absolute
distance, or a relative fraction. `relative_error` reads a measure as a fraction
of the exact magnitude and refuses an absolute bound, which has no such reading
without a proven magnitude. `ulp_budget` reads it as a comparison window in
units of the storage format and answers `None` for the same case.

A codebook format such as NF4 rounds but has no uniform step, so
`ulp_fraction` answers `None` and a ULP bound over it is refused rather than
converted.

## Composition

`compose` states the contract a value carries after one region is followed by
another: errors add, the weaker determinism wins, and the composed contract
keeps the second region's formats. `graph_budget` composes an ordered sequence
of region contracts into the budget an output carries, and `budget_admits`
compares a declared ceiling against what a graph proves.

Two shapes are priced differently:

- A reduction of `n` terms accumulates the per-step error `n - 1` times in the
  stated order and `log2(n)` times in a tree.
- A recurrence of `n` steps compounds: `(1 + e)^n - 1`. One chain has one order,
  so `over_recurrence` forbids reassociation whatever the contract stated.

A reduction accumulating in a format narrower than its storage is priced at the
accumulator's step, because every partial sum rounds to it.

## Range proofs

`prove` prices one schedule choice against a magnitude range and returns a
`RangeProof` carrying the choice, the resulting range and the error it
contributes. A choice is `StoreAs`, `AccumulateIn`, `Approximate`,
`Reassociate` or `ChunkReduction`. A choice whose result leaves the
representable range of its format is refused with `RangeUnproven` rather than
priced.

## What legality does with a budget

`CompileRequest::with_numeric_budget` states the ceiling a caller admits for the
values a compile produces. Without one, every transform that changes a combine
order over a rounding accumulation is eliminated during search. With one, a
transform is admitted where the reordered contract fits the declared measure.

The same question is asked once more after ranking. A resident route lets
invocations reach a shared accumulator in an order the schedule does not fix, so
a plan selects one only where the combines reassociate or the stated budget
covers the new order.

## What the artifact records

`SelectedPlan::numeric` is a `NumericRecord`: the contract version, the declared
budget when the caller stated one, the composed budget the outputs carry, the
per-region contracts, and the regions this plan combines in an order the program
did not state.

## What lowering records

A contract is proven on the IR. A module is what the device runs, so each
emitted module carries a `ModuleNumericRecord` derived from the lowered program
and the frozen schedule phase: every storage format the module holds, every
format a conversion produces, every operation a backend may lower to an
approximate native instruction, and the chunk width when a phase combines more
than one element per invocation.

The record is derived, never declared. `ModuleNumericRecord::of` takes the
lowered program and the phase, so a consumer holding a bundle and the artifact
it was emitted from re-derives the record and compares.
