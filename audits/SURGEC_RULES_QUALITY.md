# SURGEC_RULES_QUALITY Audit

**Scope:** `libs/tools/surgec/rules/`  -  all 101 `.srg` files  
**Active rules:** 94 (excluding 8 stdlib predicate libraries and 1 shape template)  
**Auditor:** automated source analysis + manual cross-reference  
**Date:** 2026-04-24

---

## Executive Summary

| Metric | Value |
|---|---|
| Active rules | 94 |
| Both positive + negative fixtures | 39 (41.5 %) |
| No fixtures at all | 55 (58.5 %) |
| Exact duplicate rule pairs | 5 |
| Rules referencing undefined label families | 50+ |
| Rules missing `confidence` | 94 (100 %) |
| Hardcoded function names (not families) | 3 files |
| Unreachable predicates (always false) | 1 |
| Pointless `flows_to` (gpumatch redundant) | 2 |
| Wrong sanitizer family | 1 |
| Tautological requirement | 1 |
| Rule name / directory slug mismatch | 5 |

---

## Findings

### 1. Exact duplicate across classes  -  copy_from_user_without_bound
**CRITICAL** | [CLOSED 2026-04-24] `kernel/copy_from_user_without_bound.srg` deleted; canonical lives at `launch/uninit_read_copy_to_user/rule.srg`. Regression test `no_byte_for_byte_duplicate_rule_bodies` in `tests/conformance.rs` walks every non-stdlib `.srg` and asserts byte-uniqueness so a future copy-paste fails CI.

### 2. Exact duplicate across classes  -  integer_overflow_to_alloc
**CRITICAL** | [CLOSED 2026-04-24] `memory/integer_overflow_to_alloc.srg` deleted; canonical lives at `launch/integer_overflow_alloc_arith/rule.srg`. Covered by `no_byte_for_byte_duplicate_rule_bodies`.

### 3. Exact duplicate across classes  -  remote_heap_overflow
**CRITICAL** | [CLOSED 2026-04-24] `memory/remote_heap_overflow.srg` deleted; canonical lives at `launch/remote_heap_overflow/rule.srg`. `tests/conformance.rs::document_level_lowering_collects_all_rules` redirected to the canonical path. Covered by `no_byte_for_byte_duplicate_rule_bodies`.

### 4. Exact duplicate across classes  -  toctou_filesystem
**CRITICAL** | [CLOSED 2026-04-24] `memory/toctou_filesystem.srg` deleted; canonical lives at `launch/toctou_privilege/rule.srg`. Covered by `no_byte_for_byte_duplicate_rule_bodies`.

### 5. Exact duplicate across classes  -  use_after_free
**CRITICAL** | [CLOSED 2026-04-24] `memory/use_after_free.srg` deleted; canonical lives at `launch/use_after_free_error_path/rule.srg`. Covered by `no_byte_for_byte_duplicate_rule_bodies`.

### 6. Unreachable predicate  -  reaches used on dataflow-value nodes
**CRITICAL** | [CLOSED 2026-04-24] `binary/reflective_loader.srg:30` switched `reaches(...)` → `flows_to(...)` so the predicate traverses dataflow edges instead of CFG control edges; the rule can now actually fire when buffer-source bytes reach a memory-map-exec sink.

### 7. Undefined label families  -  auth rules completely broken
**[CLOSED 2026-04-24]** All 8 auth families shipped (`password_check_family`, `credential_source_family`, `jwt_decode_family`, `alg_none_patterns`, `jwt_secure_alg`, `authentication_success_family`, `session_set_family`, `session_regenerate_family`). Regression test `every_label_reference_in_rules_resolves_to_a_label_toml` in `tests/label_loading.rs` walks every `.srg` under `rules/{auth,launch,malware,memory,web,tls,binary,crypto,kernel,ai}` and asserts every `@label` token resolves to a `LabelSet` entry  -  future copy-paste of an undefined `@label` fails CI immediately. Closure shared with #8/#9/#10/#11/#12.

**CRITICAL** | `auth/hardcoded_credential.srg:7` | References `@password_check_family` which does not exist in `labels/*.toml`. The rule cannot resolve the family and will never match a node. | Fix: create `labels/password_check_family.toml` with the relevant API names per language, or replace with an existing family.

**CRITICAL** | `auth/hardcoded_credential.srg:11` | References `@credential_source_family` (undefined). Same file, second broken family. | Fix: define `labels/credential_source_family.toml` or remove the clause.

**CRITICAL** | `auth/jwt_alg_none.srg:8` | References `@jwt_decode_family` (undefined). | Fix: create `labels/jwt_decode_family.toml`.

**CRITICAL** | `auth/jwt_alg_none.srg:8` | References `@alg_none_patterns` (undefined). | Fix: create `labels/alg_none_patterns.toml` with the literal patterns (e.g., `"none"`, `"None"`).

**CRITICAL** | `auth/jwt_alg_none.srg:12` | References `@jwt_secure_alg` (undefined). | Fix: create `labels/jwt_secure_alg.toml`.

**CRITICAL** | `auth/session_fixation.srg:7` | References `@authentication_success_family` (undefined). | Fix: create `labels/authentication_success_family.toml`.

**CRITICAL** | `auth/session_fixation.srg:7` | References `@session_set_family` (undefined). | Fix: create `labels/session_set_family.toml`.

**CRITICAL** | `auth/session_fixation.srg:14` | References `@session_regenerate_family` (undefined). | Fix: create `labels/session_regenerate_family.toml`.

### 8. Undefined label families  -  launch rules non-functional
**[CLOSED 2026-04-24]** All launch families landed. See finding #7 closure for the regression test.

**CRITICAL** | `launch/path_traversal_decode_chain/rule.srg:22` | References `@archive_entry_family` (undefined). | Fix: create `labels/archive_entry_family.toml`.

**CRITICAL** | `launch/path_traversal_decode_chain/rule.srg:26` | References `@decoder_family` (undefined). | Fix: create `labels/decoder_family.toml`.

**CRITICAL** | `launch/path_traversal_decode_chain/rule.srg:31` | References `@filesystem_read_family` (undefined). | Fix: create `labels/filesystem_read_family.toml`.

**CRITICAL** | `launch/path_traversal_decode_chain/rule.srg:32` | References `@filesystem_unlink_family` (undefined). | Fix: create `labels/filesystem_unlink_family.toml`.

**CRITICAL** | `launch/path_traversal_decode_chain/rule.srg:43` | References `@canonicalize_family` (undefined). | Fix: create `labels/canonicalize_family.toml`.

**CRITICAL** | `launch/path_traversal_decode_chain/rule.srg:47` | References `@prefix_guard_family` (undefined). | Fix: create `labels/prefix_guard_family.toml`.

**CRITICAL** | `launch/path_traversal_decode_chain/rule.srg:51` | References `@pathname_sanitize_family` (undefined). | Fix: create `labels/pathname_sanitize_family.toml`.

**CRITICAL** | `launch/side_channel_spectre_v1/rule.srg:27` | References `@comparison_family` (undefined). | Fix: create `labels/comparison_family.toml`.

**CRITICAL** | `launch/side_channel_spectre_v1/rule.srg:33` | References `@array_index_family` (undefined). | Fix: create `labels/array_index_family.toml`.

**CRITICAL** | `launch/side_channel_spectre_v1/rule.srg:52` | References `@index_mask_family` (undefined). | Fix: create `labels/index_mask_family.toml`.

**CRITICAL** | `launch/side_channel_spectre_v1/rule.srg:48` | References `@speculation_barrier_family` (undefined). | Fix: create `labels/speculation_barrier_family.toml`.

**CRITICAL** | `launch/decompression_bomb_chain/rule.srg:23` | References `@decompressor_family` (undefined). | Fix: create `labels/decompressor_family.toml`.

**CRITICAL** | `launch/decompression_bomb_chain/rule.srg:38` | References `@decompression_cap_family` (undefined). | Fix: create `labels/decompression_cap_family.toml`.

**CRITICAL** | `launch/decompression_bomb_chain/rule.srg:53` | References `@total_output_cap_family` (undefined). | Fix: create `labels/total_output_cap_family.toml`.

**CRITICAL** | `launch/integer_trunc_undersize_alloc/rule.srg:19` | References `@narrowing_cast_family` (undefined). | Fix: create `labels/narrowing_cast_family.toml`.

**CRITICAL** | `launch/parser_differential_length_lies/rule.srg:25` | References `@length_extractor_family` (undefined). | Fix: create `labels/length_extractor_family.toml`.

**CRITICAL** | `launch/parser_differential_length_lies/rule.srg:26` | References `@length_reconciliation_family` (undefined). | Fix: create `labels/length_reconciliation_family.toml`.

**CRITICAL** | `launch/parser_differential_length_lies/rule.srg:27` | References `@stream_advance_family` (undefined). | Fix: create `labels/stream_advance_family.toml`.

**CRITICAL** | `launch/double_free_concurrent/rule.srg:33` | References `@reassign_family` (undefined). | Fix: create `labels/reassign_family.toml`.

**CRITICAL** | `launch/double_free_concurrent/rule.srg:34` | References `@pointer_move_family` (undefined). | Fix: create `labels/pointer_move_family.toml`.

**CRITICAL** | `launch/driver_ioctl_size_ptr/rule.srg:28` | References `@ioctl_ptr_extract_family` (undefined). | Fix: create `labels/ioctl_ptr_extract_family.toml`.

**CRITICAL** | `launch/driver_ioctl_size_ptr/rule.srg:29` | References `@ioctl_size_extract_family` (undefined). | Fix: create `labels/ioctl_size_extract_family.toml`.

**CRITICAL** | `launch/signed_unsigned_boundary/rule.srg:21` | References `@comparison_family` (undefined). | Fix: create `labels/comparison_family.toml`.

**CRITICAL** | `launch/type_confusion/rule.srg:22` | References `@pointer_cast_family` (undefined). | Fix: create `labels/pointer_cast_family.toml`.

**CRITICAL** | `launch/uninit_read_copy_to_user/rule.srg:9` | References `@user_input_family` (undefined). | Fix: create `labels/user_input_family.toml`.

### 9. Undefined label families  -  malware matrix (30 rules) completely non-functional
**[CLOSED 2026-04-24]** All 10 source families + 5 sink families shipped. See finding #7 closure for the regression test. The duplication of the 30 rule files themselves is tracked separately under finding #20.

**CRITICAL** | `malware/buffer_to_exec.srg:6` | References `@buffer_source` and `@exec_sink` (both undefined). The entire malware directory (30 rules) uses source/sink pairs that have no TOML definitions. | Fix: define the 10 source families (`buffer_source`, `cli_source`, `credential_source`, `file_source`, `network_input_source`, `npm_script_source`, `sensitive_file_source`, `shell_source`, `system_source`) and 5 sink families (`exec_sink`, `file_sink`, `network_sink`, `sql_sink`, `xss_sink`) in `labels/*.toml`, or generate the matrix from a build script instead of shipping 30 broken rules.

### 10. Undefined label families  -  memory rules
**[CLOSED 2026-04-24]** All memory families shipped. See finding #7 closure.

**CRITICAL** | `memory/oob_read.srg:13` | References `@array_access_family` (undefined). | Fix: create `labels/array_access_family.toml`.

**CRITICAL** | `memory/oob_read.srg:17` | References `@range_check_family` (undefined). | Fix: create `labels/range_check_family.toml`.

**CRITICAL** | `memory/oob_read.srg:21` | References `@rust_safe_indexing_family` (undefined). | Fix: create `labels/rust_safe_indexing_family.toml`.

**CRITICAL** | `memory/uninit_read.srg:9` | References `@rust_DA_family` (undefined). | Fix: create `labels/rust_DA_family.toml`.

### 11. Undefined label families  -  web rules
**[CLOSED 2026-04-24]** All web families shipped (including `http_route_handler_family.toml` with single-quoted PHP namespaces to preserve `\` correctly). See finding #7 closure.

**CRITICAL** | `web/missing_auth.srg:12` | References `@http_route_handler_family` (undefined). | Fix: create `labels/http_route_handler_family.toml`.

**CRITICAL** | `web/missing_auth.srg:13` | References `@privileged_op_family` (undefined). | Fix: create `labels/privileged_op_family.toml`.

**CRITICAL** | `web/missing_auth.srg:17` | References `@auth_check_family` (undefined). | Fix: create `labels/auth_check_family.toml`.

### 12. Undefined label families  -  tls rules
**[CLOSED 2026-04-24]** TLS + crypto families shipped (`http_client_family`, `verify_arg_slot`, `false_literal`, `insecure_context_literal`, `aes_cipher_family`, `ecb_mode_literal`, `weak_hash_family`). See finding #7 closure.

**CRITICAL** | `tls/cert_verification_disabled.srg:7` | References `@http_client_family` (undefined). | Fix: create `labels/http_client_family.toml`.

**CRITICAL** | `tls/cert_verification_disabled.srg:8` | References `@verify_arg_slot` (undefined). | Fix: create `labels/verify_arg_slot.toml`.

**CRITICAL** | `tls/cert_verification_disabled.srg:10` | References `@false_literal` (undefined). | Fix: create `labels/false_literal.toml`.

**CRITICAL** | `tls/cert_verification_disabled.srg:11` | References `@insecure_context_literal` (undefined). | Fix: create `labels/insecure_context_literal.toml`.

**CRITICAL** | `crypto/ecb_mode.srg:9` | References `@aes_cipher_family` (undefined). | Fix: create `labels/aes_cipher_family.toml`.

**CRITICAL** | `crypto/ecb_mode.srg:10` | References `@ecb_mode_literal` (undefined). | Fix: create `labels/ecb_mode_literal.toml`.

**CRITICAL** | `crypto/weak_password_hash.srg:9` | References `@password_input_family` (undefined). | Fix: create `labels/password_input_family.toml`.

**CRITICAL** | `crypto/weak_password_hash.srg:9` | References `@weak_hash_family` (undefined). | Fix: create `labels/weak_hash_family.toml`.

### 13. Missing confidence field  -  corpus-wide
**HIGH** | `web/sqli.srg:4` (representative) | Not a single active rule declares `confidence`. Confidence is required for downstream triage, risk scoring, and adaptive speculation. | Fix: add `confidence = <float>` to every rule metadata, calibrated by historical true-positive / false-positive rates.

### 14. Hardcoded function names instead of label families
**HIGH** | [CLOSED 2026-04-24] `launch/uninit_read_copy_to_user/rule.srg` and `launch/driver_ioctl_size_ptr/rule.srg` switched from `call_to("copy_from_user")` / `call_to("copy_to_user")` to `call_to(@copy_from_user_family)` / `call_to(@copy_to_user_family)`. New `labels/copy_from_user_family.toml` (Linux/BSD/Windows uaccess primitives) and `labels/copy_to_user_family.toml` shipped. The `kernel/copy_from_user_without_bound.srg` reference is moot  -  that file was deleted in finding #1.

### 15. Pointless dataflow  -  gpumatch already covers the literal
**HIGH** | [CLOSED 2026-04-24] `auth/jwt_alg_none.srg:10` rewritten: `flows_to($alg_none_literal, arg_of($decode, 1))` → direct `literal_of(arg_of($decode, 1)) is @alg_none_patterns`. The gpumatch literal pre-filter handles the literal-at-arg-slot match in O(1); the dataflow confirmer was paying for nothing.

### 16. Pointless dataflow  -  regex literal to compile call
**HIGH** | [CLOSED 2026-04-24] `web/redos.srg:21` rewritten: `flows_to($pattern, arg_of($regex, 0))` → `literal_of(arg_of($regex, 0), "regex")`. Same gpumatch-redundancy pattern as #15.

### 17. Wrong sanitizer family for the vulnerability class
**HIGH** | [CLOSED 2026-04-24] `web/template_injection.srg:22` switched sanitizer from `@html_escape_family` (HTML-escaped strings still parse as template expressions, no SSTI protection) to `@template_sandbox_family` (Jinja2 SandboxedEnvironment, Twig SandboxExtension, ERB safe_level  -  the actual SSTI boundary).

### 18. Tautological requirement adds no constraint
**MEDIUM** | [CLOSED 2026-04-24] `auth/hardcoded_credential.srg` replaced `require literal_of($auth) == $literal` (LHS == RHS by construction) with `require len($literal) > 0` plus the existing `not sanitized_by(... @credential_source_family)` clause  -  non-empty literal not sourced from a credential-loading family.

### 19. Rule name / directory slug mismatch
**MEDIUM** | [CLOSED 2026-04-24] All five rules renamed to match their directory/file slug: `integer_overflow_alloc_arith`, `toctou_privilege`, `uninit_read_copy_to_user`, `use_after_free_error_path`, `cert_verification_disabled`. CVSS calibration table in `surgec/docs/cvss-calibration.md` updated alongside.

### 20. Mass structural duplication in malware/
**MEDIUM** | `malware/buffer_to_exec.srg:6` (representative) | 30 malware rules share the exact same 4-line body (`taint_flow_unsanitized(@src, @snk)` + `require $flow` + `report`). They differ only in the source and sink family names. This is unmaintainable copy-paste rather than code generation. | Fix: delete the 30 files and generate them from `malware/_shape.srg` via a build script or a TOML matrix file that lists the source/sink pairs.

### 21. Fixture coverage below quality bar
**MEDIUM** | Corpus-wide | Only 39 of 94 active rules (41.5 %) have both positive and negative fixtures. 55 rules have zero fixtures. Without adversarial fixtures, regressions cannot be caught and rule quality is unverified. | Fix: add at least one positive and one negative fixture to every rule. Priority: `auth/*`, `crypto/*`, `kernel/*`, `memory/*` duplicates, and `malware/*`.

---

## Appendix A  -  Rules with no fixtures

The following 55 rules have **zero** positive or negative fixtures under `libs/tools/surgec/tests/fixtures/rules/`:

- `auth/hardcoded_credential`
- `auth/jwt_alg_none`
- `auth/session_fixation`
- `binary/reflective_loader`
- `binary/string_encryption_decode_loop`
- `crypto/ecb_mode`
- `crypto/ecdsa_nonce_reuse`
- `crypto/hardcoded_key_material`
- `crypto/insecure_curve`
- `crypto/insecure_random`
- `crypto/mac_then_encrypt`
- `crypto/rsa_pkcs1v15_oracle`
- `crypto/static_iv_reuse`
- `crypto/weak_kdf_iterations`
- `crypto/weak_password_hash`
- `crypto/weak_prime_group`
- `deserialize/pickle_of_untrusted`
- `kernel/copy_from_user_without_bound`
- `launch/decompression_bomb_chain`
- `launch/double_free_concurrent`
- `launch/driver_ioctl_size_ptr`
- `launch/integer_overflow_alloc_arith`
- `launch/integer_trunc_undersize_alloc`
- `launch/parser_differential_length_lies`
- `launch/path_traversal_decode_chain`
- `launch/remote_heap_overflow`
- `launch/side_channel_spectre_v1`
- `launch/signed_unsigned_boundary`
- `launch/toctou_privilege`
- `launch/type_confusion`
- `launch/uninit_read_copy_to_user`
- `launch/use_after_free_error_path`
- `malware/buffer_to_exec`
- `malware/buffer_to_file`
- `malware/buffer_to_network`
- `malware/buffer_to_sql`
- `malware/buffer_to_xss`
- `malware/cli_to_exec`
- `malware/cli_to_file`
- `malware/cli_to_network`
- `malware/cli_to_sql`
- `malware/credential_to_exec`
- `malware/credential_to_file`
- `malware/credential_to_network`
- `malware/file_to_exec`
- `malware/file_to_network`
- `malware/file_to_sql`
- `malware/network_input_to_exec`
- `malware/network_input_to_file`
- `malware/network_input_to_network`
- `malware/network_input_to_sql`
- `malware/network_input_to_xss`
- `malware/npm_script_to_exec`
- `malware/npm_script_to_file`
- `malware/npm_script_to_network`
- `malware/sensitive_file_to_exec`
- `malware/sensitive_file_to_file`
- `malware/sensitive_file_to_network`
- `malware/shell_to_exec`
- `malware/shell_to_network`
- `malware/system_info_to_exec`
- `malware/system_info_to_file`
- `malware/system_info_to_network`
- `memory/integer_overflow_to_alloc`
- `memory/oob_read`
- `memory/remote_heap_overflow`
- `memory/uninit_read`
- `memory/use_after_free`

---

## Appendix B  -  Clean items

| Check | Result |
|---|---|
| TODO / FIXME in metadata or body | **None found**  -  corpus is clean. |
| `find()` with unbound iteration | **None found**  -  no `find(` construct exists in any `.srg` file. Stdlib fixpoints are bounded by `max_iterations: 64`. |
| Missing `severity` in active rules | **None found**  -  every active rule declares severity. (Stdlib predicate files correctly omit it.) |

---

## Appendix C  -  Methodology notes

1. **Fixture mapping:** Rule slug derived from file path (`category/name.srg` → `name`; `category/name/rule.srg` → `name`). Cross-checked against `libs/tools/surgec/tests/fixtures/rules/<slug>/positive*` and `negative*`.
2. **Label-family validation:** Extracted all `@family` references from `.srg` files and compared against the basename of every `.toml` file in `libs/tools/surgec/rules/labels/`.
3. **Duplicate detection:** MD5 checksums of all `.srg` files; byte-for-byte identical pairs reported.
4. **Unreachable predicate detection:** Inspected all `reaches(` calls. `reflective_loader.srg` is the only file passing value-derived bindings (`return_value_of`, `arg_of`) to `reaches`, which traverses CFG control edges.
5. **Pointless dataflow detection:** Flagged `flows_to` where the source is a `literal_of(...)` constant  -  these are already covered by gpumatch literal pre-filters.
6. **Hardcoded strings:** Grepped for `call_to("` literal strings.
