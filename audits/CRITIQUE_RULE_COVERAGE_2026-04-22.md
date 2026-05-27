# surgec Rule Corpus Coverage Audit

Scope: `libs/tools/surgec/rules/{auth,crypto,deserialize,kernel,malware,memory,tls,web,chains,stdlib}/`

Audit target: shipped rule corpus coverage, duplicate semantics, CWE/MITRE mapping quality, supply-chain blind spots, and missing rule classes.

Evidence baseline:

- `libs/tools/surgec/rules/README.md:3-10` says the shipped corpus is the Tier-B moat and should grow by small `.srg` additions.
- `libs/tools/surgec/rules/README.md:28-40` shows the current detection taxonomy: `auth`, `crypto`, `deserialize`, `kernel`, `malware`, `memory`, `tls`, `web`, plus `chains`.
- `libs/tools/surgec/rules/README.md:123-129` sets the quality bar: positive and negative fixtures, and deduplication against `stdlib`.
- `libs/tools/surgec/rules/malware/_shape.srg:3-15` shows the malware folder is mostly a generic source-to-sink shape library.

## Executive Read

- The shipped corpus contains `69` `.srg` files total, but only `60` are actual detection rules. `8` are `stdlib` vocabulary predicates/rules and `1` (`malware/_shape.srg`) is a reusable shape, not a standalone detection.
- Only `32/60` detection rules declare any CWE at all. `28/60` are unmapped. In `malware/`, only `3/31` rules declare a CWE.
- There are `33` distinct explicit CWE IDs in the shipped detection set. That is narrow relative to the OWASP Top 10 and the SANS/CWE Top 25.
- `malware/` has no explicit MITRE ATT&CK metadata at all. Coverage is only inferable from source/sink shape, not declared.
- Supply-chain coverage is effectively `npm`-only. The only package-manager-specific rules are `npm_script_to_{exec,file,network}`.
- Authz coverage is a single direct rule, `web/missing_auth.srg`, plus adjacent auth/session bugs. There is no IDOR, mass assignment, privilege-overwrite, or broken access-control depth.
- Crypto coverage is three shallow rules. The corpus does not cover IV reuse, weak KDF parameterization, padding oracles, nonce reuse, insecure curves, weak primes, or key lifecycle misuse.
- There is no AI/LLM rule family and no binary-obfuscation rule family.

## Corpus Inventory

| Bucket | Files | Detection rules | Notes |
|---|---:|---:|---|
| `auth/` | 3 | 3 | All mapped to explicit CWEs |
| `crypto/` | 3 | 3 | All mapped to explicit CWEs |
| `deserialize/` | 1 | 1 | Only Python pickle |
| `kernel/` | 1 | 1 | Single kernel copy primitive |
| `malware/` | 32 | 31 | `31` detections plus `1` reusable shape |
| `memory/` | 8 | 8 | Good depth for current classes, narrow breadth |
| `tls/` | 1 | 1 | Only disabled certificate verification |
| `web/` | 12 | 12 | Classic web bug set, not modern authz/API abuse depth |
| `chains/` | 3 `.toml` | 3 chain definitions | No CWE / ATT&CK metadata |
| `stdlib/` | 8 | 0 practical detections | Vocabulary / primitives |

## Rule × CWE Coverage

### Auth

| Rule | Severity | CWE(s) | Coverage note |
|---|---|---|---|
| `auth/hardcoded_credential.srg` | critical | `CWE-798`, `CWE-259` | Hardcoded password/secret comparison |
| `auth/jwt_alg_none.srg` | critical | `CWE-347` | JWT algorithm trust bypass |
| `auth/session_fixation.srg` | high | `CWE-384` | Session identifier not regenerated |

### Crypto

| Rule | Severity | CWE(s) | Coverage note |
|---|---|---|---|
| `crypto/ecb_mode.srg` | high | `CWE-327` | AES ECB mode |
| `crypto/insecure_random.srg` | high | `CWE-338` | Weak RNG in token/ID path |
| `crypto/weak_password_hash.srg` | high | `CWE-916` | Weak password hash family |

### Deserialize

| Rule | Severity | CWE(s) | Coverage note |
|---|---|---|---|
| `deserialize/pickle_of_untrusted.srg` | critical | `CWE-502` | Untrusted pickle load |

### Kernel

| Rule | Severity | CWE(s) | Coverage note |
|---|---|---|---|
| `kernel/copy_from_user_without_bound.srg` | critical | `CWE-120`, `CWE-787` | Kernel copy without bound enforcement |

### Memory

| Rule | Severity | CWE(s) | Coverage note |
|---|---|---|---|
| `memory/integer_overflow_to_alloc.srg` | critical | `CWE-190`, `CWE-680` | Overflowed allocation sizing |
| `memory/oob_read.srg` | critical | `CWE-125` | Out-of-bounds read |
| `memory/race_on_shared_state.srg` | high | `CWE-362` | Shared state race |
| `memory/remote_heap_overflow.srg` | critical | `CWE-120`, `CWE-122`, `CWE-787` | Network-driven heap overflow |
| `memory/toctou_filesystem.srg` | medium | `CWE-367` | Filesystem-only TOCTOU |
| `memory/unbounded_recursion.srg` | high | `CWE-674` | Recursion DoS |
| `memory/uninit_read.srg` | high | `CWE-457` | Use of uninitialized memory |
| `memory/use_after_free.srg` | critical | `CWE-416` | UAF |

### TLS

| Rule | Severity | CWE(s) | Coverage note |
|---|---|---|---|
| `tls/cert_verification_disabled.srg` | critical | `CWE-295` | Disabled TLS cert validation |

### Web

| Rule | Severity | CWE(s) | Coverage note |
|---|---|---|---|
| `web/command_injection.srg` | critical | `CWE-78` | HTTP input to command sink |
| `web/log_injection.srg` | medium | `CWE-93` | CRLF-aware log injection |
| `web/missing_auth.srg` | high | `CWE-862` | Missing auth check before privileged op |
| `web/open_redirect.srg` | medium | `CWE-601` | Open redirect |
| `web/path_traversal.srg` | high | `CWE-22` | Path traversal |
| `web/prototype_pollution.srg` | high | `CWE-915` | JS prototype pollution |
| `web/redos.srg` | medium | `CWE-1333` | ReDoS |
| `web/sqli.srg` | critical | `CWE-89` | SQL injection |
| `web/ssrf.srg` | high | `CWE-918` | SSRF |
| `web/template_injection.srg` | high | `CWE-94` | Template injection / SSTI |
| `web/xss.srg` | high | `CWE-79` | XSS |
| `web/xxe.srg` | high | `CWE-611` | XXE |

### Malware

The malware folder is predominantly shape coverage, not taxonomy coverage. `malware/_shape.srg:3-15` defines the shared pattern. Only three rules claim a CWE.

| Rule | Severity | CWE(s) | Coverage note |
|---|---|---|---|
| `malware/buffer_to_exec.srg` | high |  -  | Decoded buffer to exec sink |
| `malware/buffer_to_file.srg` | high |  -  | Decoded buffer to file sink |
| `malware/buffer_to_network.srg` | high |  -  | Decoded buffer to network sink |
| `malware/buffer_to_sql.srg` | high |  -  | Decoded buffer to SQL sink |
| `malware/buffer_to_xss.srg` | high |  -  | Decoded buffer to XSS sink |
| `malware/cli_to_exec.srg` | high |  -  | CLI arg to exec sink |
| `malware/cli_to_file.srg` | high |  -  | CLI arg to file sink |
| `malware/cli_to_network.srg` | high |  -  | CLI arg to network sink |
| `malware/cli_to_sql.srg` | high |  -  | CLI arg to SQL sink |
| `malware/credential_to_exec.srg` | critical | `CWE-200` | Credential material to exec sink |
| `malware/credential_to_file.srg` | critical | `CWE-200` | Credential material to file sink |
| `malware/credential_to_network.srg` | critical | `CWE-200` | Credential exfil to network sink |
| `malware/file_to_exec.srg` | high |  -  | File content to exec sink |
| `malware/file_to_network.srg` | high |  -  | File content to network sink |
| `malware/file_to_sql.srg` | high |  -  | File content to SQL sink |
| `malware/network_input_to_exec.srg` | critical |  -  | Downloaded content to exec sink |
| `malware/network_input_to_file.srg` | critical |  -  | Downloaded content to file sink |
| `malware/network_input_to_network.srg` | critical |  -  | Downloaded content to network sink |
| `malware/network_input_to_sql.srg` | critical |  -  | Downloaded content to SQL sink |
| `malware/network_input_to_xss.srg` | critical |  -  | Downloaded content to XSS sink |
| `malware/npm_script_to_exec.srg` | critical |  -  | npm install script to exec sink |
| `malware/npm_script_to_file.srg` | critical |  -  | npm install script to file sink |
| `malware/npm_script_to_network.srg` | critical |  -  | npm install script to network sink |
| `malware/sensitive_file_to_exec.srg` | critical |  -  | Sensitive file to exec sink |
| `malware/sensitive_file_to_file.srg` | critical |  -  | Sensitive file to file sink |
| `malware/sensitive_file_to_network.srg` | critical |  -  | Sensitive file exfil to network sink |
| `malware/shell_to_exec.srg` | high |  -  | Shell output to exec sink |
| `malware/shell_to_network.srg` | high |  -  | Shell output to network sink |
| `malware/system_info_to_exec.srg` | high |  -  | Recon/system info to exec sink |
| `malware/system_info_to_file.srg` | high |  -  | Recon/system info to file sink |
| `malware/system_info_to_network.srg` | high |  -  | Recon/system info to network sink |

### Chains

| Chain | Category | Output | Metadata quality |
|---|---|---|---|
| `chains/command_exec.toml` | exploit step chain | `exec_sink` | No CWE, no ATT&CK, no OWASP |
| `chains/deserialization_gadget.toml` | exploit step chain | `code_exec` | No CWE, no ATT&CK, no OWASP |
| `chains/web_input_sql_exec.toml` | exploit step chain | `sql_sink` | No CWE, no ATT&CK, no OWASP |

## Duplicate and Near-Duplicate Semantics

Strict duplicate claims are limited. Most overlaps are “same CWE, different source or sink.” The highest-value problems are near-duplicates and over-broad taxonomy claims:

1. `malware/credential_to_exec.srg`, `credential_to_file.srg`, and `credential_to_network.srg` all claim `CWE-200` but represent materially different attack outcomes. The network case is exfiltration, the file case is local staging, and the exec case is command construction. One generic disclosure CWE is too coarse.
2. `memory/remote_heap_overflow.srg` and `kernel/copy_from_user_without_bound.srg` both claim `CWE-120` and `CWE-787`; they are not duplicates, but the shared taxonomy obscures the kernel-specific trust boundary.
3. `crypto/ecb_mode.srg` uses `CWE-327` and so would be indistinguishable at taxonomy level from many unrelated weak-crypto classes not actually covered.
4. `web/missing_auth.srg` claims only `CWE-862`; it partially overlaps broken access control, missing authorization, and default-allow policy bugs, but only one narrow shape is implemented.
5. The three `npm_script_to_*` rules are distinct sink variants, not duplicates. The problem is the absence of ecosystem peers (`pip`, `maven`, `crates`, `gems`, `docker`), not over-duplication.
6. The malware folder could be generated from a matrix. Manually shipping the sink cartesian product without ATT&CK/CWE disambiguation increases maintenance cost and makes coverage look broader than it is.

## OWASP Top 10 / SANS Top 25 Gap Summary

### Explicitly Covered Top-25 / OWASP-Relevant CWEs

Covered explicitly: `22`, `78`, `79`, `89`, `93`, `94`, `120`, `122`, `125`, `190`, `259`, `295`, `327`, `338`, `347`, `362`, `367`, `384`, `416`, `457`, `502`, `601`, `611`, `674`, `680`, `787`, `798`, `862`, `915`, `916`, `918`, `1333`.

### High-Value Gaps from SANS/CWE Top 25

Not explicitly covered by shipped rules:

| CWE | Class | Why it matters |
|---|---|---|
| `CWE-20` | Improper Input Validation | Root class for parser trust and API boundary bugs |
| `CWE-352` | CSRF | Still absent from web authz layer |
| `CWE-434` | Unrestricted File Upload | Common web exploit surface |
| `CWE-476` | NULL Pointer Dereference | Important memory-safety completeness gap |
| `CWE-287` | Improper Authentication | Broader than current `missing_auth` and session rules |
| `CWE-77` | Command Injection (generic special element neutralization) | Broader shell/OS command family gap |
| `CWE-119` | Improper Restriction of Operations within Bounds of a Memory Buffer | Parent memory-bounds shape not modeled |
| `CWE-306` | Missing Authentication for Critical Function | Distinct from privileged-op route domination |
| `CWE-269` | Improper Privilege Management | Role overwrite / privilege escalation missing |
| `CWE-863` | Incorrect Authorization | IDOR / tenant scoping / object-level auth absent |
| `CWE-276` | Incorrect Default Permissions | Supply-chain and deployment misconfig gap |

### OWASP 2021 Category Gaps

| OWASP bucket | Current state |
|---|---|
| `A01 Broken Access Control` | Only `missing_auth`, `open_redirect`, and path-centric bugs; no IDOR, mass assignment, tenant escape, role overwrite |
| `A02 Cryptographic Failures` | Three shallow rules; no IV/KDF/nonce/padding/curve/prime coverage |
| `A03 Injection` | Good classics, missing NoSQL/LDAP/host-header/HTTP response splitting/runtime format string |
| `A04 Insecure Design` | Little beyond recursion/race/TOCTOU |
| `A05 Security Misconfiguration` | Only cert-disable and XXE-adjacent disable checks |
| `A06 Vulnerable and Outdated Components` | No dependency / version / known-bad package rules |
| `A07 Identification and Authentication Failures` | Thin coverage |
| `A08 Software and Data Integrity Failures` | Pickle + npm install script only; no signed update tampering |
| `A09 Security Logging and Monitoring Failures` | `log_injection` exists, but absence of logging/alerting misuse not modeled |
| `A10 SSRF` | Covered |

## MITRE ATT&CK Mapping for `malware/`

There is no explicit ATT&CK metadata anywhere under `libs/tools/surgec/rules/malware/`. The mapping below is therefore inferred from source/sink behavior, not declared.

| Malware rule cohort | Inferred ATT&CK | Confidence | Note |
|---|---|---|---|
| `*_to_exec` | `T1059` Command and Scripting Interpreter | medium | Generic exec sink, no shell-family subtype metadata |
| `network_input_to_exec` | `T1105` Ingress Tool Transfer, `T1059` | medium | Download then execute |
| `npm_script_to_exec` | `T1195.002` Compromise Software Supply Chain, `T1059` | high | Install script execution |
| `buffer_to_*` | `T1140` Deobfuscate/Decode Files or Information | medium | Decoded buffer shape |
| `system_info_to_*` | `T1082` System Information Discovery | medium | Recon source only |
| `sensitive_file_to_*`, `file_to_*` | `T1005` Data from Local System | medium | Local file collection/exfil staging |
| `credential_to_*` | `T1552` Unsecured Credentials | low-to-medium | Depends on `@credential_source` breadth, not declared |
| `*_to_network` | `T1041` Exfiltration Over C2 Channel or `T1071` Application Layer Protocol | low | Too generic; no protocol-level disambiguation |
| `shell_to_network` | `T1059` + possible `T1071` | low | Shell output to network, not protocol-specific |

### Required ATT&CK coverage check

| Technique | Current state |
|---|---|
| `T1059` Command and Scripting Interpreter | Partial, only inferred from exec sinks |
| `T1566` Phishing | Missing |
| `T1190` Exploit Public-Facing Application | Missing |
| `T1552` Unsecured Credentials | Partial, only inferred via generic credential-source rules |
| `T1071` Application Layer Protocol | Weak partial at best; no HTTP/DNS/SMTP/WebSocket-specific malware rules |

Gap: the malware corpus covers generic source→sink danger, not ATT&CK-grade malware behavior. That means it cannot answer “which ATT&CK techniques do we detect?” with precision.

## Supply-Chain Coverage

### Ecosystem matrix

| Ecosystem | Shipped rules | Coverage status | Notes |
|---|---:|---|---|
| `npm` | 3 | partial | Only install-script taint (`npm_script_to_{exec,file,network}`) |
| `pip / PyPI` | 0 | missing | No `setup.py`, `pyproject`, wheel entry-point, or typosquat rules |
| `crates.io` | 0 | missing | No `build.rs`, proc-macro, cargo plugin, or registry confusion rules |
| `Docker Hub / Dockerfiles` | 0 | missing | No Docker build/install pipeline rules |
| `Maven / Gradle` | 0 | missing | No plugin, goal, wrapper, or `pom.xml` supply-chain rules |
| `RubyGems` | 0 | missing | No gem install hooks, extension build hooks, or typosquat patterns |

### What the current corpus catches

- npm install-script execution/staging/exfil (`npm_script_to_exec`, `npm_script_to_file`, `npm_script_to_network`).
- Generic credential/file/system/network flows that would also trigger inside malicious packages if the lowerer labels the source/sink correctly.

### What it does not catch as first-class supply-chain fingerprints

- SolarWinds-style signed binary / signed update backdoors.
- PyPI typosquat naming patterns, homoglyphs, and stale-maintainer impersonation.
- npm post-install credential harvesters that never touch a generic `@credential_source`.
- crates.io `build.rs` network exfiltration and compile-time downloader implants.
- Maven plugin / wrapper backdoors.
- Gem install hooks and native-extension compile abuse.
- Dockerfile curl-bash, stage-smuggling, and secret-exfil at build time.

## Completeness by Requested Domain

### Memory safety

| Class | Covered? | Evidence |
|---|---|---|
| UAF | yes | `memory/use_after_free.srg` |
| OOB read | yes | `memory/oob_read.srg` |
| OOB write | no | No write-specific rule |
| Uninitialized use | yes | `memory/uninit_read.srg` |
| Integer overflow | partial | `integer_overflow_to_alloc.srg` only |
| Race | partial | `race_on_shared_state.srg` generic race only |
| Double-free | no | Missing |
| Format string | no | Missing |
| Signed overflow | no | Missing |
| Shift past width | no | Missing |
| TOCTOU beyond filesystem | no | Filesystem only in `toctou_filesystem.srg` |
| Null deref | no | Missing |

### Authz

| Class | Covered? | Evidence |
|---|---|---|
| JWT `alg=none` | yes | `auth/jwt_alg_none.srg` |
| Session fixation | yes | `auth/session_fixation.srg` |
| Hardcoded creds | yes | `auth/hardcoded_credential.srg` |
| Broken access control | partial | Only `web/missing_auth.srg` |
| Privilege escalation via role overwrite | no | Missing |
| IDOR / object-level auth | no | Missing |
| Mass assignment | no | Missing |
| Tenant / scope escape | no | Missing |

### Crypto

| Class | Covered? | Evidence |
|---|---|---|
| ECB | yes | `crypto/ecb_mode.srg` |
| Weak random | yes | `crypto/insecure_random.srg` |
| Weak password hash | yes | `crypto/weak_password_hash.srg` |
| Static IV reuse | no | Missing |
| Weak KDF iterations | no | Missing |
| MAC-then-encrypt | no | Missing |
| RSA-PKCS-v1.5 padding oracle | no | Missing |
| ECDSA nonce reuse | no | Missing |
| Insecure curve (`secp192r1`) | no | Missing |
| Known-weak primes / DH groups | no | Missing |
| Hardcoded crypto keys | no | Missing |
| Hostname verification / protocol downgrade | no | Missing beyond full cert disable |

### LLM / AI-specific

| Class | Covered? | Evidence |
|---|---|---|
| Prompt injection | no | No AI rule directory or rule names under `rules/` |
| Tool poisoning | no | Missing |
| Training-data exfil via side-channel | no | Missing |
| RAG retrieval poisoning | no | Missing |
| Model-output-to-exec/network trust break | no | Missing |

### Binary / obfuscation

| Class | Covered? | Evidence |
|---|---|---|
| Packer detection | no | Missing |
| Control-flow flattening | no | Missing |
| Opaque predicates | no | Missing |
| String encryption / staged decode loops | no direct rule | `buffer_to_*` only catches the decoded-buffer consequence |
| Reflective loading / in-memory PE/ELF | no | Missing |
| Dynamic API resolution / hashed imports | no | Missing |

## Finding Ledger

Format requested by task: `severity | file:line | description | suggested fix`

### A. Taxonomy, metadata, and duplicate-quality findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| critical | `libs/tools/surgec/rules/malware/_shape.srg:3` | `malware/` is a source→sink matrix with no ATT&CK metadata, so the corpus cannot answer which malware behaviors it claims to cover. | Ship `malware/attack_metadata.toml` and require each malware rule to declare `attack = ["Txxxx"]`. Minimal sketch: `rule network_input_to_exec { severity = critical; attack = ["T1105","T1059"]; ... }`. |
| high | `libs/tools/surgec/rules/README.md:28` | The taxonomy has no `ai/`, `binary/`, `supply_chain/`, or `authz/` subtrees despite those being first-class modern detection domains. | Add new rule families: `rules/ai/`, `rules/binary/`, `rules/supply_chain/`, `rules/authz/`. Minimal sketch: `rule prompt_injection { severity = critical; let $p = call_to(@llm_prompt_family); ... }`. |
| high | `libs/tools/surgec/rules/chains/command_exec.toml:1` | `chains/*.toml` have no CWE/ATT&CK metadata, so exploit-graph findings lose taxonomy at the chain layer. | Extend chain schema with `cwe`, `owasp`, `attack`. Minimal sketch: `name="download_exec"; cwe=["CWE-494"]; attack=["T1105","T1059"]`. |
| high | `libs/tools/surgec/rules/auth/hardcoded_credential.srg:1` | `hardcoded_credential` conflates `CWE-798` and `CWE-259` in one rule without distinguishing inbound default credentials from embedded outbound service secrets. | Split into `hardcoded_password_check` and `embedded_service_secret`. Minimal sketch: `rule embedded_service_secret { severity = critical; let $lit = literal_of(call_to(@secret_use_family)); ... }`. |
| medium | `libs/tools/surgec/rules/malware/credential_to_exec.srg:1` | The `credential_to_{exec,file,network}` trio all claim only `CWE-200`, masking materially different behaviors. | Keep sink variants but add more precise secondary metadata. Minimal sketch: `rule credential_to_network { severity = critical; cwe=["CWE-200"]; attack=["T1552","T1041"]; ... }`. |
| medium | `libs/tools/surgec/rules/kernel/copy_from_user_without_bound.srg:1` | Kernel memory-copy abuse shares coarse CWEs with generic heap overflow rules, so reports will collapse unlike behaviors into the same bucket. | Add kernel-specific tags and CAPEC/ATT&CK-like trust-boundary metadata. Minimal sketch: `tags=["kernel","usercopy","memory_corruption"]`. |
| high | `libs/tools/surgec/rules/README.md:123` | The README requires positive and negative fixtures, but the malware folder’s large matrix encourages rule proliferation without taxonomy review or fixture depth. | Add a metadata lint: no new malware rule without `cwe|attack|ecosystem` and positive/negative corpus entries. Minimal sketch: `[[test_inputs]] name="buildrs_exfil_pos" expect_fires=true`. |
| high | `libs/tools/surgec/rules/crypto/ecb_mode.srg:1` | `CWE-327` is used as a catch-all weak crypto bucket, but the corpus implements only one specific mode misuse. | Add specific crypto rule families instead of over-crediting `CWE-327`. Minimal sketch: `rule static_iv_reuse { severity = high; let $iv = arg_of(call_to(@cipher_init_family),1); require literal_of($iv); ... }`. |

### B. OWASP / SANS Top-25 gap findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| critical | `libs/tools/surgec/rules/web/missing_auth.srg:6` | No rule covers `CWE-863` Incorrect Authorization / object-level authorization failure. | Ship `authz/idor_object_access.srg`. Minimal sketch: `rule idor_object_access { severity = critical; let $id = route_param("id"); let $fetch = call_to(@object_fetch_family); require flows_to($id,arg_of($fetch,0)); require not any(call_to(@ownership_check_family)==$c and dominates($c,$fetch): $c in all_nodes()); report { primary:$fetch } }`. |
| critical | `libs/tools/surgec/rules/web/missing_auth.srg:6` | No rule covers `CWE-269` improper privilege management via role overwrite or authority escalation. | Ship `authz/role_overwrite.srg`. Minimal sketch: `rule role_overwrite { severity = critical; let $src = call_to(@http_input_family); let $set = call_to(@role_assignment_family); require flows_to(return_value_of($src), arg_of($set,1)); report { primary:$set } }`. |
| high | `libs/tools/surgec/rules/web/missing_auth.srg:6` | No rule covers `CWE-287` improper authentication beyond one missing-check shape. | Ship `authz/auth_bypass_fallback.srg`. Minimal sketch: `rule auth_bypass_fallback { severity = high; let $guard = call_to(@auth_check_family); require return_value_of($guard) == literal(false); require reaches($guard, call_to(@privileged_op_family)); ... }`. |
| high | `libs/tools/surgec/rules/web/path_traversal.srg:1` | No rule covers `CWE-434` unrestricted file upload. | Ship `web/unrestricted_upload.srg`. Minimal sketch: `rule unrestricted_upload { severity = high; let $upload = call_to(@file_upload_family); require not any(call_to(@file_type_validate_family)==$v and dominates($v,$upload): $v in all_nodes()); report { primary:$upload } }`. |
| high | `libs/tools/surgec/rules/web/command_injection.srg:3` | No rule covers `CWE-352` CSRF on state-changing actions. | Ship `web/csrf_missing_token.srg`. Minimal sketch: `rule csrf_missing_token { severity = high; let $entry = call_to(@http_route_handler_family); let $state = call_to(@state_change_family); require reaches($entry,$state); require not any(call_to(@csrf_check_family)==$c and dominates($c,$state): $c in all_nodes()); report { primary:$state } }`. |
| high | `libs/tools/surgec/rules/memory/oob_read.srg:1` | No rule covers `CWE-476` NULL dereference. | Ship `memory/null_deref.srg`. Minimal sketch: `rule null_deref { severity = high; let $use = call_to(@pointer_use_family); require arg_of($use,0) == literal(null); report { primary:$use } }`. |
| high | `libs/tools/surgec/rules/web/command_injection.srg:3` | No explicit rule covers `CWE-77` generic command injection across non-shell interpreters. | Ship `web/interpreter_injection.srg`. Minimal sketch: `rule interpreter_injection { severity = critical; let $src = call_to(@http_input_family); let $snk = call_to(@interpreter_eval_family); require flows_to(return_value_of($src), arg_of($snk,0)); report { primary:$snk } }`. |
| medium | `libs/tools/surgec/rules/memory/remote_heap_overflow.srg:5` | No parent-boundary rule covers `CWE-119` generic bounds restriction failures outside the current hand-picked memory shapes. | Ship `memory/unchecked_buffer_op.srg`. Minimal sketch: `rule unchecked_buffer_op { severity = high; let $op = call_to(@pointer_use_family); require not bounded_by_comparison(arg_of($op,2),$op); report { primary:$op } }`. |

### C. MITRE ATT&CK and malware-behavior gap findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| critical | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:1` | No malware rule explicitly declares `T1059`, even though every `*_to_exec` rule is essentially interpreter execution. | Add ATT&CK metadata to all `*_to_exec` rules. Minimal sketch: `rule shell_to_exec { severity = high; attack=["T1059"]; ... }`. |
| critical | `libs/tools/surgec/rules/malware/network_input_to_exec.srg:1` | `network_input_to_exec` is an obvious `T1105` ingress-tool-transfer pattern but ships unmapped. | Ship `malware/download_then_exec.srg` with ATT&CK metadata. Minimal sketch: `rule download_then_exec { severity = critical; attack=["T1105","T1059"]; let $f = taint_flow_unsanitized(@network_input_source,@exec_sink); require $f; report { primary:$f } }`. |
| critical | `libs/tools/surgec/rules/malware/credential_to_network.srg:1` | `T1552` unsecured credentials is only weakly implied through generic credential sources, not declared or specialized. | Ship `malware/unsecured_credential_exfil.srg`. Minimal sketch: `rule unsecured_credential_exfil { severity = critical; attack=["T1552","T1041"]; let $f = taint_flow_unsanitized(@credential_file_source,@network_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/malware/system_info_to_network.srg:1` | System discovery is only implied. No ATT&CK-grade differentiation between `T1082` system-info discovery and generic telemetry. | Ship `malware/system_discovery_to_c2.srg`. Minimal sketch: `rule system_discovery_to_c2 { severity = high; attack=["T1082","T1071"]; let $f = taint_flow_unsanitized(@system_source,@http_c2_sink); require $f; report { primary:$f } }`. |
| critical | `libs/tools/surgec/rules/README.md:36` | No rule addresses `T1566` phishing delivery or lure-trigger execution. | Ship `malware/phishing_attachment_exec.srg`. Minimal sketch: `rule phishing_attachment_exec { severity = critical; attack=["T1566","T1204"]; let $open = call_to(@email_attachment_open_family); let $exec = call_to(@exec_sink); require reaches($open,$exec); report { primary:$exec } }`. |
| critical | `libs/tools/surgec/rules/web/command_injection.srg:3` | No rule addresses `T1190` exploit public-facing application as a malware chain, even though web exploitation is a canonical initial-access technique. | Ship `malware/public_facing_exploit_to_exec.srg`. Minimal sketch: `rule public_facing_exploit_to_exec { severity = critical; attack=["T1190","T1059"]; let $f = taint_flow_unsanitized(@http_input_family,@exec_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/malware/buffer_to_exec.srg:1` | Decoded-buffer execution is behaviorally close to `T1140` but the corpus loses that signal entirely. | Ship `malware/decode_then_exec.srg`. Minimal sketch: `rule decode_then_exec { severity = high; attack=["T1140","T1059"]; let $f = taint_flow_unsanitized(@buffer_source,@exec_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/malware/shell_to_network.srg:1` | No network rule distinguishes `T1071` application-layer C2 from generic networking. | Ship `malware/shell_output_to_http_dns.srg`. Minimal sketch: `rule shell_output_to_http_dns { severity = high; attack=["T1059","T1071"]; let $f = taint_flow_unsanitized(@shell_source,@application_protocol_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/malware/system_info_to_file.srg:1` | No persistence techniques are covered (`schtasks`, registry run keys, launch agents), even though existing incident tests exercise them. | Ship `malware/persistence_via_scheduler.srg`. Minimal sketch: `rule persistence_via_scheduler { severity = high; attack=["T1053"]; let $exec = call_to(@task_scheduler_family); require $exec; report { primary:$exec } }`. |
| high | `libs/tools/surgec/rules/malware/system_info_to_exec.srg:1` | No coverage exists for registry run-key persistence (`T1547.001`) or service creation (`T1543`). | Ship `malware/registry_runkey_persistence.srg`. Minimal sketch: `rule registry_runkey_persistence { severity = high; attack=["T1547.001"]; let $w = call_to(@registry_runkey_write_family); require $w; report { primary:$w } }`. |
| medium | `libs/tools/surgec/rules/malware/file_to_network.srg:1` | Local-file exfil is modeled, but no ATT&CK metadata distinguishes source types like browser stores, SSH keys, kubeconfigs, or CI tokens. | Ship `malware/ssh_key_to_network.srg`. Minimal sketch: `rule ssh_key_to_network { severity = critical; attack=["T1552","T1041"]; let $f = taint_flow_unsanitized(@ssh_key_source,@network_sink); require $f; report { primary:$f } }`. |
| medium | `libs/tools/surgec/rules/malware/network_input_to_network.srg:1` | No rule models relay/proxy malware behavior or beacon fan-out. | Ship `malware/beacon_relay.srg`. Minimal sketch: `rule beacon_relay { severity = high; attack=["T1071"]; let $f = taint_flow_unsanitized(@network_input_source,@network_sink); require $f; require protocol_of($f) in ["http","dns","ws"]; report { primary:$f } }`. |
| medium | `libs/tools/surgec/rules/malware/file_to_exec.srg:1` | No rule models signed-binary side-loading or living-off-the-land execution. | Ship `malware/lolbin_side_load.srg`. Minimal sketch: `rule lolbin_side_load { severity = high; attack=["T1218"]; let $f = taint_flow_unsanitized(@file_source,@signed_binary_exec_sink); require $f; report { primary:$f } }`. |
| medium | `libs/tools/surgec/rules/malware/cli_to_exec.srg:1` | CLI-driven malware is covered, but no rule maps `T1204` user execution or packaging-lure execution. | Ship `malware/user_execution_lure.srg`. Minimal sketch: `rule user_execution_lure { severity = high; attack=["T1204"]; let $src = call_to(@user_open_family); let $snk = call_to(@exec_sink); require reaches($src,$snk); report { primary:$snk } }`. |

### D. Supply-chain gap findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| critical | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:3` | Supply-chain coverage is JS-only. There is no `pip` / `PyPI` install-hook source. | Ship `supply_chain/pypi_setup_hook_to_exec.srg`. Minimal sketch: `rule pypi_setup_hook_to_exec { severity = critical; let $f = taint_flow_unsanitized(@pypi_install_hook_source,@exec_sink); require $f; report { primary:$f } }`. |
| critical | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:3` | No `pyproject.toml` / PEP 517 build-backend supply-chain coverage. | Ship `supply_chain/pep517_backend_to_network.srg`. Minimal sketch: `rule pep517_backend_to_network { severity = critical; let $f = taint_flow_unsanitized(@pep517_backend_source,@network_sink); require $f; report { primary:$f } }`. |
| critical | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:3` | No crates.io compile-time implant coverage for `build.rs`. | Ship `supply_chain/build_rs_to_network.srg`. Minimal sketch: `rule build_rs_to_network { severity = critical; let $f = taint_flow_unsanitized(@cargo_build_script_source,@network_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:3` | No proc-macro / compiler-plugin abuse coverage for Rust supply chain. | Ship `supply_chain/proc_macro_side_effect.srg`. Minimal sketch: `rule proc_macro_side_effect { severity = high; let $f = taint_flow_unsanitized(@proc_macro_source,@network_sink); require $f; report { primary:$f } }`. |
| critical | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:3` | No Maven/Gradle lifecycle hook coverage. | Ship `supply_chain/maven_plugin_exec.srg`. Minimal sketch: `rule maven_plugin_exec { severity = critical; let $f = taint_flow_unsanitized(@maven_plugin_source,@exec_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:3` | No Gradle init-script / buildSrc implant coverage. | Ship `supply_chain/gradle_init_to_network.srg`. Minimal sketch: `rule gradle_init_to_network { severity = high; let $f = taint_flow_unsanitized(@gradle_init_source,@network_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:3` | No RubyGems install-hook or native-extension build coverage. | Ship `supply_chain/gemspec_post_install_to_exec.srg`. Minimal sketch: `rule gemspec_post_install_to_exec { severity = high; let $f = taint_flow_unsanitized(@gem_install_hook_source,@exec_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/malware/npm_script_to_exec.srg:3` | No Docker build-time implant coverage. | Ship `supply_chain/docker_build_to_network.srg`. Minimal sketch: `rule docker_build_to_network { severity = high; let $f = taint_flow_unsanitized(@docker_build_stage_source,@network_sink); require $f; report { primary:$f } }`. |
| high | `libs/tools/surgec/rules/README.md:36` | No typosquat / homoglyph / namespace-confusion family exists even though this is a standard supply-chain attack class. | Ship `supply_chain/typosquat_manifest_anomaly.srg`. Minimal sketch: `rule typosquat_manifest_anomaly { severity = high; let $pkg = call_to(@package_manifest_family); require literal_of(arg_of($pkg,0)) is @typosquat_literal; report { primary:$pkg } }`. |
| high | `libs/tools/surgec/rules/README.md:36` | No rule covers maintainer/publisher impersonation or suspicious ownership transfer. | Ship `supply_chain/publisher_drift.srg`. Minimal sketch: `rule publisher_drift { severity = high; let $meta = call_to(@package_publish_metadata_family); require suspicious_publisher_change($meta); report { primary:$meta } }`. |
| critical | `libs/tools/surgec/rules/README.md:36` | No first-class rule covers SolarWinds-style signed-binary backdoor / signed update tampering. | Ship `supply_chain/signed_update_backdoor.srg`. Minimal sketch: `rule signed_update_backdoor { severity = critical; let $sig = call_to(@signature_verify_family); let $drop = call_to(@signed_binary_replace_family); require reaches($sig,$drop); report { primary:$drop } }`. |
| high | `libs/tools/surgec/rules/README.md:36` | No rule covers package-manifest curl-bash / remote installer strings when they do not flow through a labeled source. | Ship `supply_chain/manifest_remote_installer.srg`. Minimal sketch: `rule manifest_remote_installer { severity = high; let $cmd = literal_of(call_to(@manifest_script_family)); require regex(\"curl .*\\| *(sh|bash)\") near $cmd; report { primary:$cmd } }`. |
| high | `libs/tools/surgec/rules/malware/npm_script_to_network.srg:1` | No rule models credential harvesting from `.npmrc`, `.pypirc`, `.cargo/credentials`, `.gem/credentials`, or CI secret files as ecosystem-specific supply-chain abuse. | Ship `supply_chain/registry_token_exfil.srg`. Minimal sketch: `rule registry_token_exfil { severity = critical; let $f = taint_flow_unsanitized(@registry_credential_file_source,@network_sink); require $f; report { primary:$f } }`. |
| medium | `libs/tools/surgec/rules/README.md:36` | No rule covers dependency confusion against internal namespaces. | Ship `supply_chain/dependency_confusion_fetch.srg`. Minimal sketch: `rule dependency_confusion_fetch { severity = high; let $dep = call_to(@package_dependency_decl_family); require internal_namespace_shadow($dep); report { primary:$dep } }`. |

### E. Memory-safety gap findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| critical | `libs/tools/surgec/rules/memory/oob_read.srg:1` | There is no explicit OOB-write detector even though writes are generally more severe than reads. | Ship `memory/oob_write.srg`. Minimal sketch: `rule oob_write { severity = critical; let $w = call_to(@pointer_use_family); require writes_memory($w); require not bounded_by_comparison(arg_of($w,2),$w); report { primary:$w } }`. |
| critical | `libs/tools/surgec/rules/memory/use_after_free.srg:5` | Double-free is missing. | Ship `memory/double_free.srg`. Minimal sketch: `rule double_free { severity = critical; let $f1 = call_to(@deallocator_family); let $f2 = call_to(@deallocator_family); require arg_of($f1,0) == arg_of($f2,0); require dominates($f1,$f2); report { primary:$f2 related:[$f1] } }`. |
| high | `libs/tools/surgec/rules/memory/integer_overflow_to_alloc.srg:5` | Signed overflow is missing; only attacker-influenced alloc-size overflow is covered. | Ship `memory/signed_integer_overflow.srg`. Minimal sketch: `rule signed_integer_overflow { severity = high; let $op = binary_op(@signed_arith_family); require flows_to(return_value_of(call_to(@receive_family)),$op); require not sanitized_by($op,@overflow_check_family); report { primary:$op } }`. |
| high | `libs/tools/surgec/rules/memory/integer_overflow_to_alloc.srg:5` | Shift-past-width / undefined-bitshift bugs are not covered. | Ship `memory/shift_past_width.srg`. Minimal sketch: `rule shift_past_width { severity = high; let $op = binary_op(@shift_family); require arg_of($op,1) >= bit_width(arg_of($op,0)); report { primary:$op } }`. |
| high | `libs/tools/surgec/rules/memory/toctou_filesystem.srg:1` | TOCTOU is filesystem-only. There is no lock/state/IPC/db TOCTOU rule. | Ship `memory/toctou_lock_state.srg`. Minimal sketch: `rule toctou_lock_state { severity = high; let $check = call_to(@state_check_family); let $use = call_to(@state_use_family); require dominates($check,$use); require concurrent_write_between($check,$use); report { primary:$use related:[$check] } }`. |
| high | `libs/tools/surgec/rules/memory/race_on_shared_state.srg:11` | Generic race coverage does not cover atomic-ordering misuse or check-then-act races. | Ship `memory/check_then_act_race.srg`. Minimal sketch: `rule check_then_act_race { severity = high; let $check = call_to(@shared_access_family); let $act = call_to(@shared_access_family); require reaches($check,$act); require not synchronized_between($check,$act); report { primary:$act } }`. |
| high | `libs/tools/surgec/rules/memory/oob_read.srg:5` | Format-string bugs are absent despite being a classic sink-driven memory corruption / disclosure class. | Ship `memory/format_string.srg`. Minimal sketch: `rule format_string { severity = critical; let $fmt = call_to(@format_sink_family); require flows_to(return_value_of(call_to(@http_input_family)), arg_of($fmt,0)); report { primary:$fmt } }`. |
| medium | `libs/tools/surgec/rules/memory/uninit_read.srg:5` | No stack exhaustion / VLA / `alloca` abuse rule exists. | Ship `memory/unbounded_stack_alloc.srg`. Minimal sketch: `rule unbounded_stack_alloc { severity = high; let $a = call_to(@stack_alloc_family); require flows_to(return_value_of(call_to(@receive_family)), arg_of($a,0)); report { primary:$a } }`. |
| medium | `libs/tools/surgec/rules/memory/use_after_free.srg:5` | No invalid-free / free-of-non-heap-object rule exists. | Ship `memory/invalid_free.srg`. Minimal sketch: `rule invalid_free { severity = high; let $f = call_to(@deallocator_family); require not originates_from(arg_of($f,0), @allocator_family); report { primary:$f } }`. |
| medium | `libs/tools/surgec/rules/memory/remote_heap_overflow.srg:5` | No type-confusion / cast-mismatch memory corruption rule exists. | Ship `memory/type_confusion.srg`. Minimal sketch: `rule type_confusion { severity = high; let $cast = call_to(@unsafe_cast_family); let $use = call_to(@pointer_use_family); require reaches($cast,$use); report { primary:$use related:[$cast] } }`. |

### F. Authz / API abuse gap findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| critical | `libs/tools/surgec/rules/web/missing_auth.srg:3` | No IDOR coverage for route IDs, query IDs, or object keys. | Ship `authz/idor_route_param.srg`. Minimal sketch: `rule idor_route_param { severity = critical; let $id = route_param(\"id\"); let $fetch = call_to(@object_fetch_family); require flows_to($id,arg_of($fetch,0)); require not any(call_to(@ownership_check_family)==$c and dominates($c,$fetch): $c in all_nodes()); report { primary:$fetch } }`. |
| critical | `libs/tools/surgec/rules/web/missing_auth.srg:3` | No mass-assignment rule exists for binder/deserializer to privileged fields. | Ship `authz/mass_assignment.srg`. Minimal sketch: `rule mass_assignment { severity = critical; let $bind = call_to(@model_bind_family); require binds_sensitive_field($bind,@privileged_field_literal); report { primary:$bind } }`. |
| critical | `libs/tools/surgec/rules/web/missing_auth.srg:3` | No role overwrite / claim overwrite rule exists. | Ship `authz/role_claim_overwrite.srg`. Minimal sketch: `rule role_claim_overwrite { severity = critical; let $src = call_to(@http_input_family); let $set = call_to(@role_assignment_family); require flows_to(return_value_of($src), arg_of($set,1)); report { primary:$set } }`. |
| high | `libs/tools/surgec/rules/web/missing_auth.srg:3` | No multi-tenant scoping / account boundary rule exists. | Ship `authz/tenant_scope_bypass.srg`. Minimal sketch: `rule tenant_scope_bypass { severity = high; let $tenant = route_param(\"tenant\"); let $fetch = call_to(@tenant_object_fetch_family); require flows_to($tenant,arg_of($fetch,0)); require not any(call_to(@tenant_scope_check_family)==$c and dominates($c,$fetch): $c in all_nodes()); report { primary:$fetch } }`. |
| high | `libs/tools/surgec/rules/auth/jwt_alg_none.srg:1` | JWT coverage is narrow. No rule covers trusting user-controlled role claims without server-side revalidation. | Ship `authz/jwt_role_trust.srg`. Minimal sketch: `rule jwt_role_trust { severity = high; let $decode = call_to(@jwt_decode_family); let $authz = call_to(@privileged_op_family); require flows_to(claim_of($decode,\"role\"),$authz); require not any(call_to(@server_role_lookup_family)==$c and dominates($c,$authz): $c in all_nodes()); report { primary:$authz } }`. |
| high | `libs/tools/surgec/rules/web/missing_auth.srg:3` | No rule covers default-allow authorization fallthrough. | Ship `authz/default_allow_policy.srg`. Minimal sketch: `rule default_allow_policy { severity = high; let $policy = call_to(@authz_policy_family); require returns_default_allow($policy); report { primary:$policy } }`. |
| medium | `libs/tools/surgec/rules/auth/session_fixation.srg:1` | Session rule covers fixation only, not session reuse after privilege elevation. | Ship `authz/session_reuse_after_role_change.srg`. Minimal sketch: `rule session_reuse_after_role_change { severity = medium; let $elev = call_to(@role_assignment_family); let $set = call_to(@session_set_family); require reaches($elev,$set); require not any(call_to(@session_regenerate_family)==$r and dominates($elev,$r) and dominates($r,$set): $r in all_nodes()); report { primary:$set } }`. |
| medium | `libs/tools/surgec/rules/web/missing_auth.srg:3` | No rule covers method confusion / verb-only authorization differences (e.g. GET protected, PATCH not). | Ship `authz/method_specific_guard_gap.srg`. Minimal sketch: `rule method_specific_guard_gap { severity = medium; let $route = call_to(@http_route_handler_family); require route_method($route) in [\"PATCH\",\"PUT\",\"DELETE\"]; require not any(call_to(@auth_check_family)==$c and dominates($c,$route): $c in all_nodes()); report { primary:$route } }`. |

### G. Crypto and TLS gap findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| high | `libs/tools/surgec/rules/crypto/ecb_mode.srg:1` | No static-IV or IV-reuse rule exists. | Ship `crypto/static_iv_reuse.srg`. Minimal sketch: `rule static_iv_reuse { severity = high; let $init = call_to(@cipher_init_family); let $iv = arg_of($init,1); require literal_of($iv); report { primary:$init } }`. |
| high | `libs/tools/surgec/rules/crypto/weak_password_hash.srg:1` | No weak-KDF-iterations rule exists for PBKDF2/bcrypt/scrypt/Argon2 parameter misuse. | Ship `crypto/weak_kdf_iterations.srg`. Minimal sketch: `rule weak_kdf_iterations { severity = high; let $kdf = call_to(@kdf_family); require numeric_literal(arg_of($kdf,1)) < @min_kdf_iterations; report { primary:$kdf } }`. |
| high | `libs/tools/surgec/rules/crypto/ecb_mode.srg:1` | No MAC-then-encrypt / unauthenticated-encryption ordering rule exists. | Ship `crypto/mac_then_encrypt.srg`. Minimal sketch: `rule mac_then_encrypt { severity = high; let $mac = call_to(@mac_family); let $enc = call_to(@cipher_encrypt_family); require reaches($mac,$enc); require not uses_aead($enc); report { primary:$enc related:[$mac] } }`. |
| critical | `libs/tools/surgec/rules/crypto/ecb_mode.srg:1` | No RSA PKCS#1 v1.5 padding-oracle rule exists. | Ship `crypto/rsa_pkcs1v15_oracle.srg`. Minimal sketch: `rule rsa_pkcs1v15_oracle { severity = critical; let $dec = call_to(@rsa_pkcs1v15_decrypt_family); require oracle_error_flow($dec); report { primary:$dec } }`. |
| critical | `libs/tools/surgec/rules/crypto/insecure_random.srg:11` | No ECDSA/DSA nonce reuse rule exists. | Ship `crypto/ecdsa_nonce_reuse.srg`. Minimal sketch: `rule ecdsa_nonce_reuse { severity = critical; let $sign = call_to(@ecdsa_sign_family); require deterministic_nonce_disabled($sign) and weak_nonce_source($sign); report { primary:$sign } }`. |
| high | `libs/tools/surgec/rules/crypto/ecb_mode.srg:1` | No insecure-curve rule exists (`secp192r1`, low-security curves, deprecated groups). | Ship `crypto/insecure_curve.srg`. Minimal sketch: `rule insecure_curve { severity = high; let $curve = literal_of(call_to(@ecc_curve_select_family)); require $curve is @weak_curve_literal; report { primary:$curve } }`. |
| high | `libs/tools/surgec/rules/crypto/ecb_mode.srg:1` | No weak-prime / weak-DH-group coverage exists. | Ship `crypto/weak_prime_group.srg`. Minimal sketch: `rule weak_prime_group { severity = high; let $grp = call_to(@dh_group_select_family); require literal_of(arg_of($grp,0)) is @weak_prime_group_literal; report { primary:$grp } }`. |
| high | `libs/tools/surgec/rules/auth/hardcoded_credential.srg:1` | No hardcoded cryptographic key / static signing key rule exists. | Ship `crypto/hardcoded_key_material.srg`. Minimal sketch: `rule hardcoded_key_material { severity = critical; let $use = call_to(@crypto_key_use_family); let $lit = literal_of($use); require $lit; report { primary:$use related:[$lit] } }`. |
| high | `libs/tools/surgec/rules/tls/cert_verification_disabled.srg:1` | TLS coverage stops at outright cert disable. There is no hostname-verification or weak-protocol rule. | Ship `tls/hostname_verification_disabled.srg`. Minimal sketch: `rule hostname_verification_disabled { severity = high; let $tls = call_to(@tls_client_init_family); require literal_of(arg_of($tls,@verify_hostname_arg)) == false; report { primary:$tls } }`. |
| medium | `libs/tools/surgec/rules/tls/cert_verification_disabled.srg:1` | No TLS version downgrade / insecure protocol selection rule exists. | Ship `tls/legacy_protocol_enabled.srg`. Minimal sketch: `rule legacy_protocol_enabled { severity = medium; let $cfg = call_to(@tls_config_family); require literal_of(arg_of($cfg,@min_version_arg)) is @legacy_tls_literal; report { primary:$cfg } }`. |

### H. LLM / AI-specific gap findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| critical | `libs/tools/surgec/rules/README.md:28` | There is no prompt-injection rule at all. | Ship `ai/prompt_injection.srg`. Minimal sketch: `rule prompt_injection { severity = critical; let $src = call_to(@untrusted_prompt_source_family); let $llm = call_to(@llm_prompt_family); require flows_to(return_value_of($src), arg_of($llm,0)); report { primary:$llm } }`. |
| critical | `libs/tools/surgec/rules/README.md:28` | There is no tool-poisoning rule for agent/tool registries or tool call arguments. | Ship `ai/tool_poisoning.srg`. Minimal sketch: `rule tool_poisoning { severity = critical; let $src = call_to(@untrusted_tool_spec_source_family); let $call = call_to(@agent_tool_call_family); require flows_to(return_value_of($src), arg_of($call,0)); report { primary:$call } }`. |
| high | `libs/tools/surgec/rules/README.md:28` | There is no training-data or retrieval-data exfiltration side-channel rule. | Ship `ai/training_data_exfil.srg`. Minimal sketch: `rule training_data_exfil { severity = high; let $src = call_to(@embedding_store_source_family); let $net = call_to(@network_sink); require flows_to(return_value_of($src), arg_of($net,0)); report { primary:$net } }`. |
| high | `libs/tools/surgec/rules/README.md:28` | There is no rule for model output trusted directly as command, SQL, or network target. | Ship `ai/model_output_to_exec.srg`. Minimal sketch: `rule model_output_to_exec { severity = critical; let $m = call_to(@llm_response_family); let $exec = call_to(@exec_sink); require flows_to(return_value_of($m), arg_of($exec,0)); report { primary:$exec related:[$m] } }`. |

### I. Binary / obfuscation gap findings

| Severity | file:line | Description | Suggested fix |
|---|---|---|---|
| high | `libs/tools/surgec/rules/malware/buffer_to_exec.srg:1` | No packer signature / stub detection exists. | Ship `binary/packer_stub.srg`. Minimal sketch: `rule packer_stub { severity = high; let $sig = literal_of(call_to(@binary_magic_family)); require $sig is @packer_magic_literal; report { primary:$sig } }`. |
| high | `libs/tools/surgec/rules/malware/_shape.srg:8` | No control-flow-flattening or dispatcher-loop rule exists. | Ship `binary/control_flow_flattening.srg`. Minimal sketch: `rule control_flow_flattening { severity = high; let $loop = call_to(@dispatcher_loop_family); require opaque_switch_dispatch($loop); report { primary:$loop } }`. |
| high | `libs/tools/surgec/rules/malware/_shape.srg:8` | No opaque-predicate rule exists. | Ship `binary/opaque_predicate.srg`. Minimal sketch: `rule opaque_predicate { severity = high; let $pred = call_to(@branch_predicate_family); require always_true_or_false_under_known_domain($pred); report { primary:$pred } }`. |
| medium | `libs/tools/surgec/rules/malware/buffer_to_exec.srg:1` | String encryption is only caught after decode reaches a sink; encrypted-string storage itself is not modeled. | Ship `binary/string_encryption_decode_loop.srg`. Minimal sketch: `rule string_encryption_decode_loop { severity = medium; let $dec = call_to(@string_decode_family); require loop_dominates($dec); report { primary:$dec } }`. |
| high | `libs/tools/surgec/rules/malware/file_to_exec.srg:1` | No reflective-loader / in-memory PE or ELF mapping rule exists. | Ship `binary/reflective_loader.srg`. Minimal sketch: `rule reflective_loader { severity = high; let $mem = call_to(@memory_map_exec_family); let $buf = call_to(@buffer_source_family); require reaches($buf,$mem); report { primary:$mem related:[$buf] } }`. |

## Net Assessment

The current corpus is good at classic web bugs, a handful of C/C++ memory classes, and broad malware-taint shapes. It is not yet a comprehensive rule corpus. The two structural issues are:

1. Taxonomy debt: too many rules, especially in `malware/` and `chains/`, ship without CWE/OWASP/ATT&CK metadata.
2. Coverage debt: modern attack classes are absent entirely, especially supply-chain, authz/API abuse, AI/LLM abuse, binary obfuscation, and deeper cryptographic misuse.

If this corpus is intended to be the shipped Tier-B moat, the next milestone should be:

- add explicit metadata requirements for all detection rules;
- create `authz/`, `supply_chain/`, `ai/`, and `binary/` subtrees;
- and ship the missing classes above before claiming broad OWASP / ATT&CK coverage.
