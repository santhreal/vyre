# vyre-scan

Canonical scan program compilation, paging, residency, execution, and readback for Vyre.

## Ownership

`vyre-libs::scan` owns substrate-neutral builders such as DFA, NFA, literal-set,
and region-evidence programs. `vyre-scan` owns product workflows over those
programs: codec validation, execution sessions, paging, immutable matcher
residency, dispatch, and decoded matches.

The public facade delegates `vyre::scan` to this crate. Concrete target
compilation and materialization remain registered driver facets.

## Main APIs

- `GpuLiteralSet`: compile literal patterns and prepare scan resources.
- `ScanProgram`: one neutral regex/NFA program and its immutable tables.
- `ScanSession`: execution and codec lifecycle for one `ScanProgram`.
- `MatchScan`: common scan workflow contract.
- `ResidentLiteralScan` and `ResidentScanSession`: immutable matcher residency.
- `scan_paged_fused` and `scan_pattern_sharded`: bounded corpus and rule sharding.
- `RegionEvidencePipeline`: region admission and evidence extraction.
