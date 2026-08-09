# vyre-frontend-c

`vyre-frontend-c` ingests C translation units and lowers the supported `kernel`
entry subset to backend-neutral Vyre IR.

```rust
use vyre_frontend_c::lower_source;

let program = lower_source(
    "void kernel(const unsigned int *input, unsigned int *output) { \
     output[0] = input[0] + 1u; }",
)?;
# Ok::<(), vyre_frontend_c::CFrontendError>(())
```

The crate owns source validation, syntax parsing, source diagnostics, and typed
IR construction. It does not select a backend, compile artifacts, materialize
target modules, execute programs, emit object files, or link binaries.

## Supported lowering subset

- Exactly one function named `kernel`.
- A scalar `int` or `unsigned int` return with no parameters, or a `void`
  function whose parameters are scalar pointers.
- Integer literals, buffer subscripts, parentheses, and `+`, `-`, `*`, `&`,
  `|`, and `^` expressions.
- A scalar return or direct assignments to writable pointer parameters.

Valid C outside this subset returns `CFrontendError::Unsupported`. Malformed
syntax and invalid byte input return source-located errors. Input is bounded by
`MAX_SOURCE_BYTES`.

Use `parse_source` or `parse_source_bytes` when parsing and lowering must be
separate. Use `lower_translation_unit` to lower an accepted
`ParsedTranslationUnit`. Use `lower_source` for the combined source-to-IR path.
