# vyre IR statement semantics

Applies to Vyre 0.7.2.

This document specifies how each `Node` variant evaluates. It is the semantic
contract; the byte encoding is a separate concern and lives in
[wire-format.md](wire-format.md).

For an operation and backend pair marked supported by the operation matrix, the
backend result must agree with the registered reference oracle within the
operation's declared exact or ULP tolerance. Unsupported pairs fail visibly.

Existing variant semantics remain stable within the 0.7 series. A breaking
semantic change requires the compatibility process in
[`docs/semver-policy.md`](semver-policy.md).

One naming note before the rules. `Node::forever` is a constructor, not a node
variant. It builds a `Node::Loop` whose upper bound is `u32::MAX`. Wherever this
document says `forever`, the encoded node is an ordinary `Loop`.

## Variable lifecycle: `Let`, `Assign`, scope

`Node::Let { name, value }` introduces a new binding in the current scope. The
binding is live from the immediately following statement until the enclosing
scope exits.

Shadowing is rejected by default. A second `Let` with the same name anywhere in
the visible scope chain, current or enclosing, is a `V008` validation error:

```text
V008: duplicate local binding `x` shadows an outer scope. Fix: choose a unique
local name, or opt into nested shadowing with ValidationOptions::with_shadowing(true).
```

As the message says, this is a default and not a law. `ValidationOptions::with_shadowing(true)`
accepts nested shadowing. Leave it off unless you have a reason: the default
keeps statement IR easy to reason about, removes a class of SSA-conversion edge
cases, and matches WGSL's explicit no-shadowing discipline. Autodiff and
canonical-form passes rely on it, so a pass that renames must produce globally
unique names.

`Node::Assign { name, value }` mutates the most recent `Let` binding
of `name` in scope. It is an error (surfaced by the validator) to
`Assign` to a name that has not been `Let`-bound in scope.

**This is the contract every Cat-A composition depends on:**

- `Let(acc, 0)` then `Assign(acc, acc + x)` inside a loop accumulates
  across iterations (not a fresh binding per iteration).
- `Let(state, 0)` at the outer scope then a sequence of `Assign(state, …)`
  inside an inner scope observably mutates the outer binding from the
  perspective of every later statement in the outer scope.
- `Assign` does not create a new binding.
- `Assign` to a name with no visible `Let` is a validation error.

### Scope rules

| Construct | Opens new scope? | Inherits parent? |
| --- | --- | --- |
| `Node::Block` | yes | yes |
| `Node::Loop` | yes (per-iteration scope is child of loop header scope) | yes |
| `Node::If { then, otherwise }` | yes (one per branch) | yes |
| `Node::Region { body, … }` | yes | yes |

Region preserves the same child-scope behavior as `Node::Block`. An `Assign`
inside a region still resolves a visible outer binding when no inner binding
shadows it. Composition-preserving rewrites must keep that lookup behavior.

### Why Assign exists at all

vyre IR is deliberately not SSA. Algebraic-law passes, autodiff
transform, and canonical-form normalization all run over a statement
IR whose mutation is local to a named binding. SSA can be derived on
demand via `transform::ssa::to_ssa`; it is not the canonical form.

**Future-proofing:**

- The validator freezes this contract in code. Any future pass that
  wants to assume immutability must call `transform::ssa::to_ssa`
  first.
- Wire format: `Node::Let` and `Node::Assign` have distinct tag bytes
  (`0x02` and `0x03`). The validator treats stray `Assign` as a hard
  error, not a warning, so downstream tools can trust the shape.
- The conformance runner generates proptest cases that exercise
  every combination of shadowing, scope exit, and Region wrapping,
  and every backend must match the CPU reference byte-for-byte.

## Execution order within a `Vec<Node>`

Statements execute top-to-bottom. Control-flow statements (`If`,
`Loop`, `Return`, `Barrier`) have their usual meaning. `Return`
unwinds all scopes and exits the current entry-point invocation.
`Barrier` is a workgroup synchronization point; passing through a
divergent barrier is `V010` (validation error).

## Iteration semantics for `Node::Loop`

`Node::Loop { var, from, to, body }` evaluates `from` and `to` once
at loop entry. The loop variable `var` is `from`, `from+1`, …,
`to-1` (half-open). The body runs once per iteration in a fresh
child scope. An `Assign` to the loop variable itself is `V011`,
because loop variables are immutable; rename instead.

`Node::forever(body)` is sugar for `Loop { var: "__forever__",
from: 0, to: u32::MAX, body }`, chosen so existing passes process
it without a new variant.

## Region semantics recap

`Node::Region { generator, source_region, body }` is a scoped debug wrapper.
Evaluation matches `Node::Block(body)`: the body executes in a child scope, and
the metadata fields do not affect values. Backend lowering and region-inlining
must preserve the same binding behavior.

## `Node::Opaque` semantics

`Node::Opaque(extension)` requires explicit support from the selected execution
path. The reference interpreter rejects an opaque node until it is lowered to
core nodes or receives a reference evaluator. A backend that cannot honor the
extension returns an actionable unsupported-capability error.

## Error codes introduced by this contract

| Code | Rule | Fix |
| --- | --- | --- |
| `V008` | A `Let` name already bound in the visible scope chain | Choose a unique local name, or pass `ValidationOptions::with_shadowing(true)` if nested shadowing is intended |

`Assign` to a name that was never `Let`-bound is also rejected, but the
validator reports it without a `V` code:

```text
assignment to undeclared variable `x`. Fix: add `let x = ...;` before this assignment.
```

Do not confuse this with `V033`, which is unrelated: `V033` is emitted when
expression nesting exceeds `DEFAULT_MAX_EXPR_DEPTH`. See
[error-codes.md](error-codes.md) for the full code list.
