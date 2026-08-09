# vyre-scan

Canonical scan program compilation, artifact materialization, paging, residency, typed submission, and readback for Vyre.

## Ownership

`vyre-libs::scan` owns substrate-neutral builders such as DFA, NFA, literal-set,
and region-evidence programs. `vyre-scan` owns product workflows over those
programs: codec validation, authenticated artifact sessions, paging, immutable
matcher residency, typed submission, and decoded matches.

The public facade delegates `vyre::scan` to this crate. Registered driver facets
own target compilation and materialization.

## Main APIs

- `ScanProgram`: one neutral regex/NFA program and its immutable tables.
- `ScanSession`: neutral construction, codec lifecycle, and independent reference scan.
- `MaterializedScanSession`: authenticated target payload, artifact instance, typed bindings, submission, and readback.
- `ScanArtifactSession`: reusable canonical artifact lifecycle for scan products.
- `GpuLiteralSet`: literal pattern compilation and prepared scan resources.
- `scan_paged_fused` and `scan_pattern_sharded`: bounded corpus and rule sharding.
- `RegionEvidencePipeline`: region admission and evidence extraction.
