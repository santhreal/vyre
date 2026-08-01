# vyre-conform  -  Configurability

Tier A and Tier B knobs for the `vyre-conform` binary. Both tiers are
required; together they form the entire surface that operators and the
community use to extend conformance without touching Rust.

## Tier A  -  operational config

CLI flags and environment variables. Tier A controls *how* a run executes.

| Flag / env                       | Applies to                | Default                       | Purpose                                                                 |
|----------------------------------|---------------------------|-------------------------------|-------------------------------------------------------------------------|
| `--backend <id>`                 | every subcommand          | `auto` (dispatch), `all`      | Backend to dispatch against, or `all` to cover every registered backend. |
| `--ops <all\|op_id>`             | every subcommand          | `all`                         | Restrict the run to one op id.                                          |
| `--out <path>`                   | `plan`, `prove`, `merge`  | see `--help`                  | Where the plan, certificate or merged certificate is written.           |
| `--certificates <dir>`           | `prove`                   | `certs/`                      | Directory the signed certificate lands in.                             |
| `--shard <index>/<count>`        | `plan`, `prove`           | unsharded                     | Run one shard of the op set, for parallel proving across machines.     |
| env `VYRE_BACKEND`               | `plan`, `prove`           | unset                         | Default value for `--backend`; the flag still wins.                    |
| env `VYRE_CONFORM_PROOF_WORKERS` | `prove`                   | detected parallelism, min 8   | Worker threads used to prove op pairs.                                 |
| env `VYRE_CONFORM_PROOF_TIMING`  | `prove`                   | off                           | `1`/`true`/`yes`/`on` logs per-stage proof timing.                     |
| env `VYRE_CONFORM_PROOF_PAIR_TIMING_MS` | `prove`            | `250`                         | Only pairs slower than this many milliseconds are reported.            |
| env `VYRE_CONFORM_PROOF_PAIR_START` | `prove`                | off                           | `1`/`true`/`yes`/`on` logs each pair as it starts.                     |

Run `vyre-conform --help` for the same surface. This table is the whole Tier A
surface: a flag that is not listed here is rejected with `unknown flag`.

Precedence is compiled default, then environment variable, then CLI flag. The
flag always wins. There is no config file: the runner reads no
`vyre-conform.toml`, and this document used to say it did.

## Tier B  -  the witness corpus

Every conformance witness the runner executes comes from the
`inventory`-registered op harnesses in `vyre-libs`, `vyre-intrinsics` and
`vyre-primitives`. `unified_entries` chains those three catalogs, and each entry
carries the program builder, its test inputs and its expected output. That is
the whole corpus: `vyre-conform dispatch --ops all` runs exactly the ops those
three crates register.

To add a witness pair today, register an `OpEntry` next to the op it covers:

```rust
inventory::submit! {
    crate::harness::OpEntry::new(
        MY_OP_ID,
        || my_op("input", "output", 2, 2),
        Some(|| vec![vec![/* input bytes */]]),
        Some(|| vec![vec![/* expected output bytes */]]),
    )
}
```

Earlier revisions of this document described a TOML corpus under `rules/kat/`
that the runner auto-loaded, with a `rules/SCHEMA.md` schema-of-truth. Neither
path is in this repository and no code reads them, so the description is
removed rather than left standing as an extension route that resolves to
nothing. A data-driven corpus remains the intended shape for this layer; it
lands as a documented loader plus a real `rules/` tree, not as prose.
