# vyre-bench architecture

`vyre-bench` separates benchmark execution from command-line presentation. The library owns benchmark cases, registration, execution, probes, release matrices, and report data. The `vyre-bench` binary owns argument parsing, report-file commands, dashboard generation, and the line-oriented evolution server.

## Dependency direction

```text
binary CLI
    |
    v
benchmark library
    |
    v
engine and backend crates
```

Library modules do not import the binary CLI. CLI modules may call the public library API. Dashboard and server code stays private to the binary and is not part of the library API.

## Cargo features

The `cli` feature enables `clap`, `env_logger`, and the `vyre-bench` binary. It remains a default feature so existing `cargo run -p vyre-bench -- ...` commands keep working.

Library consumers that do not need the command-line application can omit its dependencies:

```toml
vyre-bench = { path = "../vyre-bench", default-features = false }
```

A library-only build uses:

```bash
cargo check -p vyre-bench --lib --no-default-features
```

## Ownership

| Responsibility | Owner |
| --- | --- |
| Case schema and registry | `api/`, `registry/` |
| Case implementations | `cases/` |
| Measurement and execution | `runner/`, `probes/` |
| Report data and serializers | `report/` |
| Release coverage | `release_matrix.rs` |
| CLI commands and presentation | `cli.rs`, `cli/` |
| Process entry point | `main.rs` |

Keep service loops, file-oriented presentation, and argument parsing under the binary boundary. Add reusable measurement behavior to the library instead of calling CLI modules from benchmark cases.
