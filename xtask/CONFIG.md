# xtask  -  Configurability

xtask is the workspace's internal build-task runner. It is intentionally
`publish = false`. Its Tier A surface is exposed as subcommands; Tier B
is the op catalog xtask emits and consumes.

## Tier A  -  operational config

| Subcommand / env             | Default | Purpose                                                                   |
|------------------------------|---------|---------------------------------------------------------------------------|
| `xtask catalog`              |  -        | Emit the op catalog TOML manifest.                                        |
| `xtask perf-inventory wave1` |  -        | Run wave-1 perf inventory.                                                |
| `xtask lego-audit`           |  -        | Walk vyre-libs primitives and audit the LEGO surface.                    |
| `xtask publish-dryrun`       |  -        | Dry-run cargo publish across every workspace member.                     |
| env `XTASK_PARALLELISM`      | `nproc` | Host-side parallelism for catalog walk / lego-audit.                      |
| env `XTASK_VERBOSE`          | `0`     | `1` = log per-crate work; `2` = log per-file.                             |
| env `XTASK_TARGET_DIR`       | `target-xtask` | Override the per-agent target dir to avoid contention with cargo-fleet. |

xtask is meant to be invoked from the workspace root; running it from a
sub-directory is a usage bug, not a configuration choice.

## Tier B  -  community knowledge

xtask emits no op corpus. Its data files sit beside its manifest and each has
one reader: `gate-baselines.toml` is the sweep's pin per gate,
`dup-baseline.toml` is `dup-scan`, `feature-isolation.toml` is
`feature-isolation`, `public-api-paths.toml` is `public-api-paths`, and
`unsafe-budget.txt` is `lint-unsafe-budget`. Adding a dimension to one of them
is a code change in its reader and a data change in the file, in the same patch.

The launch rule layout is owned by `xtask/src/rule_tree`, which
`audit_rule_contracts` and `scaffold_rule` both resolve through, so the auditor
and the scaffolder cannot disagree about where a rule lives.

A workspace-wide op corpus and its schema were described here and have never
existed in this repository. The rule corpora that do exist are per crate:
`vyre-libs/rules`, `vyre-lower/rules` and `vyre-lints/rules`.
