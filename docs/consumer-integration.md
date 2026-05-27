# Consumer integration spec

Downstream analyzers should lower into a small set of true graph and
predicate primitives, then compose higher-level rules in their own rule
libraries. This document pins the tier contract that keeps platform semantics
inside Vyre while leaving product-specific policy outside the platform crates.

## Tier 2.5 substrate: `vyre-primitives`

The substrate every downstream analyzer lowers into. Every feature-gated domain
ships a CPU reference, a `fn(...) -> Program` builder, and `OpEntry`
registration.


| domain      | feature     | purpose                                                                                                                                                                   |
| ----------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graph`     | `graph`     | canonical ProgramGraph ABI (5-buffer CSR), `csr_forward_traverse`, `csr_backward_traverse`, `path_reconstruct`, `scc_decompose`, `toposort`, `reachable`                  |
| `bitset`    | `bitset`    | `and` / `or` / `not` / `xor` / `popcount` / `any` / `contains` over packed u32 bitsets                                                                                    |
| `fixpoint`  | `fixpoint`  | `bitset_fixpoint`  -  deterministic ping-pong convergence driver                                                                                                            |
| `reduce`    | `reduce`    | `count` / `min` / `max` / `sum` over bitsets + u32 ValueSets                                                                                                              |
| `label`     | `label`     | `resolve_family`  -  `node_tags` AND `family_mask` → NodeSet                                                                                                                |
| `predicate` | `predicate` | 10 frozen primitive predicates (`call_to`, `return_value_of`, `arg_of`, `size_argument_of`, `edge`, `in_function`, `in_file`, `in_package`, `literal_of`, `node_kind_eq`) |


## Canonical ProgramGraph ABI

`vyre_primitives::graph::program_graph::ProgramGraphShape` declares:


| binding | name                | access   | purpose                                     |
| ------- | ------------------- | -------- | ------------------------------------------- |
| 0       | `pg_nodes`          | ReadOnly | per-node `NodeKind` tag                     |
| 1       | `pg_edge_offsets`   | ReadOnly | CSR row pointers (`node_count + 1` entries) |
| 2       | `pg_edge_targets`   | ReadOnly | CSR column (`edge_count` entries)           |
| 3       | `pg_edge_kind_mask` | ReadOnly | per-edge `EdgeKind` bitmask                 |
| 4       | `pg_node_tags`      | ReadOnly | per-node tag bitmap (`TagFamily`)           |


Primitives use binding indices 5+ for their own frontier, output, and scratch
buffers. A downstream analyzer's emitted `Program` fills the five canonical
buffers with CSR bytes assembled at scan time.

## Downstream lowering path

The expected lowering shape is:

1. **User-defined predicate** (AST `PredicateDef`) → inline the body
   and recurse.
2. **Frozen primitive** (one of the 10) → delegate to
   `vyre_primitives::predicate::<name>`. Single dispatch, fixed
   contract.
3. **Analysis-library composition** (`flows_to`, `sanitized_by`, `taint_flow`,
   `bounded_by_comparison`, `dominates`, `label_by_family`,
   `path_reconstruct`, and their aliases) → delegate to
   `vyre_libs::security::<name>` which ships as Tier-3 shims over
   the Tier-2.5 primitives.
4. Everything else → a hard lowering error with an actionable diagnostic.

## Tier-3 shim policy

`vyre-libs::security::`* is an API-stability layer: the op ids stay
stable for external consumers, and each body is a one-call delegation
to the matching Tier-2.5 primitive. The real semantics live at
Tier 2.5; downstream policy composition, including fixpoint orchestration,
sanitizer exclusion, and path rebuild rules, lives in the consuming analyzer's
Tier B rule database.

## Edge-kind, tag-family, and node-kind constants

Downstream analyzers and Vyre agree on these sentinels via
`vyre_primitives::predicate::{edge_kind, tag_family, node_kind}`.
Adding a new `EdgeKind` requires appending a new bit to the module, teaching the
relevant primitive predicates, and registering any downstream rule keyword. No
changes to the 5-buffer ABI are required.
