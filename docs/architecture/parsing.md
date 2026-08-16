# Parsing

```text
source bytes
  -> lexer Program            (registered operations over raw bytes)
  -> structural extraction    (registered operations over token records)
  -> packed AST (VAST)
```

Parsing in vyre is a domain library of registered semantic operations, not
a host parser with a device accelerator bolted on. Each stage is a
`Program` over raw source bytes or over the token records the previous
stage wrote, so the whole pipeline is IR the compiler can fuse and the
optimizer can see through.

## The substrate is language-neutral

`vyre-libs/src/parsing/core` owns the pieces no language owns: AST node
kinds, delimiter handling, bracket matching, and full-grammar
shunting-yard AST generation. A frontend composes these; it does not
reimplement them.

`vyre-libs/src/parsing/vast.rs` is the packed AST wire format and its host
walks, re-exported from `vyre-foundation`. A frontend's output is VAST, so
a consumer reads one shape regardless of which language produced it.

`vyre-libs/src/parsing/lr_tables` holds precomputed LR(1) action and goto
tables plus `parse_lr`. The tables are `&'static [u32]` slices from a
manual SLR(1) construction of the C expression grammar. `parse_lr` is a
host reference parser: it is the differential oracle for the device
pipeline, and no shipping path routes a parse through it.

`vyre-grammar-gen` is the host-side generator for the lexer DFA and the
LR(1) tables. It runs on the developer's machine and its output is
committed, so a build does not depend on it.

## A frontend is a feature

| Feature | Frontend |
|---|---|
| `go-parser` | Go 1.21: byte lexer plus structural extraction for declarations |
| `python-parser` | Python 3.12: byte lexer plus structural extractors |

`parsing` enables every frontend at once. None is in `default`.

A frontend keeps the runtime path device-native: the lexer and every
structural extractor return a `Program` over raw source bytes. Tests may
use host helpers to compute reference expectations, and the shipping path
links no host parser.

## Caching is content-keyed

`source_cache` is a content-hash LRU over parsed source artifacts. A
pipeline opts in through `ParsedSourceLru::get_or_parse`, so the key is the
bytes and nothing else: the same source hits the same entry regardless of
path, timestamp or which frontend asked.

`parallel_parse` fans `get_or_parse` across cores and preserves input
order. It parallelizes the cache lookups over a corpus. It does not
execute IR on the host.

## Why the stages are separate registered operations

A lexer that is one registered operation and an extractor that is another
can be fused by the whole-program compiler when the graph makes that legal,
and can be reused by a second language when the byte-level work is the
same. A monolithic parse operation forecloses both, and neither the
optimizer nor a second frontend can recover the structure afterwards.

The shunting-yard statement pass is a named boundary inside one operation
rather than a second registered operation, so it carries the
`anonymous::` generator prefix. See
[the placement rule](../lego-block-rule.md) for what that prefix means to
the composition gate.
