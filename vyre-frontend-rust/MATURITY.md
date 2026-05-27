# vyre-frontend-rust maturity gate

Current maturity: **experimental / v0.0.1 scaffold**.

This crate is the thin pipeline driver for a GPU-first Rust compiler.
All reusable GPU primitives live in `vyre-libs::parsing::rust`.

## Scope boundary

`vyre-frontend-rust` owns Rust-source ingestion, nano-subset parsing,
GPU-reachable lexing evidence, and eventually borrow-check evidence.

It does not own the Vyre platform release proof.

## Supported nano-subset (v0.0.1)

- Functions: `fn name(param: Type) -> Type { body }`
- Let bindings: `let mut? name: Type = expr;`
- Return: `return expr;`
- Types: `i32`, `bool`, `&T`, `&mut T`
- Expressions: integer/bool literals, `+ - * / == <`, `if/else`, blocks,
  function calls, `&expr`, `*expr`
- NO: generics, traits, impls, macros, modules, structs, enums, closures,
  async, const, static, lifetimes, attributes

## Promotion gate

Promotion out of experimental requires:

| Gate | Required evidence |
|---|---|
| Lexer correctness | `tests/lexer_oracle.rs` passes against `rustc_lexer` on a corpus of Rust source files. |
| GPU lexing | `tests/gpu_lex.rs` proves the GPU lexer path matches the CPU oracle byte-for-byte. |
| Parser correctness | `tests/parse_smoke.rs` and `tests/parse_oracle.rs` pass on the nano-subset. |
| Borrow check | `tests/borrow_oracle.rs` validates Weir borrow analysis against rustc borrow checker on nano-subset programs. |
| No silent fallback | If GPU probe fails, the error is loud and actionable. |

## Production criteria

TBD. This crate is not on the Vyre 0.4.2 release train.
