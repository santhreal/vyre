# {{crate_name}}

Community Category-A op dialect for
[`vyre-libs`](https://docs.rs/vyre-libs).

Generated from the scaffold at `examples/libs-template`. Ships a skeleton
`example_op` demonstrating the five-step Cat-A authoring recipe (see
[AUTHORING.md](https://github.com/santhreal/vyre/blob/main/vyre-libs/AUTHORING.md)).

## Quickstart

```sh
cargo generate --git https://github.com/santhreal/vyre examples/libs-template \
    --name {{crate_name}}
cd {{crate_name}}
cargo test
```

## Layout

```
{{crate_name}}/
├── Cargo.toml
├── src/
│   └── lib.rs         # your op lives here
└── tests/
    └── cat_a_conform.rs  # byte-identity witness tests
```

## Contributing a second op

1. Add a new module under `src/` with your builder and free function.
2. Submit one `OperationRegistration` so the canonical registry and conformance harness discover it.
3. Add a witness in `tests/cat_a_conform.rs`.
4. Run `cargo test`.

## Publishing

Community dialect crates ship on crates.io directly, with no pull request
into `vyre-libs`. Use a `vyre-libs-` prefix in the crate name so discovery
is easy, for example `vyre-libs-quant` or `vyre-libs-llm`. A consumer
reaches your ops by depending on your crate and calling
`vyre_foundation::operation::OperationRegistry::global()`, which serves
every operation linked into the binary.
