# CRITIQUE: SURGE Language Surface + Stdlib Rule Quality

**Date:** 2026-04-22  
**Scope:**
1. `libs/surge/src/`  -  SURGE rule DSL (AST, parser, lexer, expr, rule modules)
2. `libs/tools/surgec/rules/stdlib/` + `libs/tools/surgec/rules/labels/`  -  shipped rule corpus
3. `libs/tools/surgec/AUTHORING.md` + `libs/performance/matching/vyre/rules/op/SCHEMA.md`  -  documented contract vs shipped AST

**Methodology:** Read-only static analysis of source, rules, and documentation. No source files were modified.

---

## 1. SURGE DSL Surface (`libs/surge/src/`)

### 1.1 AST / Public API Leaks

**1. CRITICAL** | `libs/surge/src/ast/expr.rs:50-57`  
AST doc comments for `Arrow` and `IsMember` leak lowering details (`stdlib flows_to` predicate, label bitset intersection) to surface-language consumers. Public AST docs should describe semantics, not compiler internals.  
**Fix:** Replace implementation-specific desugaring notes with semantic descriptions. Move lowering details to internal compiler architecture docs.

**2. CRITICAL** | `libs/surge/src/ast/expr.rs:97-112`  
AST doc comments for `LetIn` and `Quantifier` leak vyre IR concepts ("CSE-bound introduction", "no vyre opcode expansion", `any`/`all` lowering forms) in public API documentation.  
**Fix:** Redact all vyre-opcode, inliner, and Node-type references from AST doc comments. Document semantic behavior only.

**3. CRITICAL** | `libs/surge/src/ast/expr.rs:228-231`  
`Fixpoint::max_iterations` doc comment exposes the `Node::Loop` vyre IR concept in a public struct field.  
**Fix:** Document the semantic purpose (iteration bound to prevent infinite fixpoint computation) without naming vyre IR node types.

### 1.2 Parser Implementation Leaks

**4. HIGH** | `libs/surge/src/parser/expr.rs:98`  
Parse error string begins with `"internal:"`, exposing parser-internal invariants to end users. Error messages are part of the language surface.  
**Fix:** Rewrite as actionable user-facing error: `Fix: flow chain must contain at least one arrow operator (~> or ->).`

**5. HIGH** | `libs/surge/src/parser/expr.rs:166,295-301`  
Manual cursor arithmetic (`self.cursor + 1`, `self.cursor + 2`) leaks the token-vector implementation into expression parsing logic.  
**Fix:** Introduce a `peek_ahead(n)` abstraction on the parser token stream to hide cursor math.

**6. HIGH** | `libs/surge/src/parser/rules.rs:254-267`  
`parse_predicate_def` joins local bindings with `Expr::And` and documents that "the inliner treats [it] as sequenced." The AST is being used as a side-channel for sequencing, and inliner behavior is documented in the parser.  
**Fix:** Add a dedicated `Expr::Sequence` AST variant; remove inliner implementation details from parser comments.

**7. HIGH** | `libs/surge/src/parser/mod.rs:340-348`  
`bump()` destroys source-location information by replacing the current token with a dummy `SpannedToken` (zero span) via `std::mem::replace`.  
**Fix:** Retain source spans in a lookahead ring buffer instead of mutating the token stream in place.

### 1.3 Silent Fallbacks & Orphaned AST Nodes

**8. HIGH** | `libs/surge/src/corpus.rs:38,57`  
Empty match arms silently ignore unrecognized corpus lines (`_ => {}`) and trailing content after `}` (`Section::Done => {}`). This swallows malformed input without diagnostics.  
**Fix:** Replace silent drops with explicit `ParseError::UnexpectedToken` or `ParseError::TrailingContent` carrying line number and a `Fix:` directive.

**9. HIGH** | `libs/surge/src/parser/expr.rs:170`  
`_ => {}` in `parse_ident_primary` silently falls through to generic ident handling for unknown keywords instead of producing an error.  
**Fix:** Produce `ParseError::UnknownKeyword` with a `Fix:` directive listing valid keywords.

**10. HIGH** | `libs/surge/src/parser/rules.rs:154`  
`shape: None` is hardcoded; the parser never produces `Rule::shape` even though `ShapeInvocation` exists in the AST. The AST node is orphaned.  
**Fix:** Implement shape-invocation parsing in rule headers or delete the orphaned `ShapeInvocation` AST node.

**11. HIGH** | `libs/surge/src/parser/mod.rs` (structural)  
`Document::uses` (`Vec<UseDecl>`) is never populated; there is no parser branch for `use` declarations. `UseDecl` is effectively dead code in the AST.  
**Fix:** Add `use` declaration parser production and populate `Document::uses`, or remove `UseDecl` from the AST.

**12. HIGH** | `libs/surge/src/parser/mod.rs` (structural)  
`Exemption` exists in `ast::exempt` but `Document` has no `exemptions` field and the parser has no `exemption` production.  
**Fix:** Add `exemptions: Vec<Exemption>` to `Document` and implement the `exemption` parser production per spec §8.

**13. MEDIUM** | `libs/surge/src/ast/use_decl.rs:30-31`  
`unwrap_or("")` silently yields an empty name for a `use` declaration with an empty path instead of erroring.  
**Fix:** Replace with `ok_or_else(|| ParseError::EmptyUsePath { fix: "provide a non-empty module path" })?`.

### 1.4 Heuristic & Deferred Validation

**14. MEDIUM** | `libs/surge/src/lib.rs:51-53`  
Corpus format is detected by substring search (`source.contains("strings:") && source.contains("condition:")`), which can misidentify v3 files that happen to contain those substrings in comments or strings.  
**Fix:** Parse the document header and inspect the declared version field instead of doing substring heuristics.

**15. MEDIUM** | `libs/surge/src/lexer.rs:355-366`  
`read_number` accepts letters, underscores, and hyphens inside numeric literals, deferring validation to the parser. This produces malformed tokens.  
**Fix:** Reject invalid number characters at lex time with `LexerError::InvalidNumericLiteral` carrying the offending character and a `Fix:` directive.

### 1.5 Parse Errors Without Actionable Fix Directives

**16. MEDIUM** | `libs/surge/src/parser/expr.rs:98`  
Error `"internal: flow chain with no arrows"` lacks a `Fix:` directive.  
**Fix:** Change to `ParseError::EmptyFlowChain { fix: "add at least one arrow operator (~> or ->) between nodes" }`.

**17. MEDIUM** | `libs/surge/src/parser/rules.rs:185`  
Error `"failed to parse binding \`{name}\` in rule \`{rule}\`: {e}"` lacks a `Fix:` directive.  
**Fix:** Wrap the inner error and prepend `fix: "check that the binding expression uses valid SURGE syntax"`.

**18. MEDIUM** | `libs/surge/src/parser/rules.rs:239`  
Error `"expected \`rec\` to produce a Fixpoint, got {other:?}"` lacks a `Fix:` directive.  
**Fix:** Add `fix: "replace the binding body with a recursive expression using the rec keyword"`.

**19. MEDIUM** | `libs/surge/src/parser/rules.rs:293`  
Error `"shape \`{name}\` has more than one \`report\` block."` lacks a `Fix:` directive.  
**Fix:** Add `fix: "merge the report blocks or remove the duplicate"`.

**20. MEDIUM** | `libs/surge/src/parser/mod.rs:121-124`  
Error `"attributes must precede a \`rule\`, \`predicate\`, or \`shape\` declaration."` lacks a `Fix:` directive.  
**Fix:** Add `fix: "move the attribute to appear immediately before the declaration it annotates"`.

**21. MEDIUM** | `libs/surge/src/parser/mod.rs:147-154`  
Error `"attribute \`{name}\` wants a string literal, got \`{other:?}\`."` lacks a `Fix:` directive.  
**Fix:** Add `fix: "provide a double-quoted string literal as the attribute value"`.

### 1.6 Missing Grammar Productions

**22. HIGH** | `libs/surge/src/lexer.rs` (structural)  
Block comments (`/* … */`) are not implemented; only `//` line comments and `#` shell-style comments exist.  
**Fix:** Add `/* … */` block comment lexer production.

**23. HIGH** | `libs/surge/src/lexer.rs` (structural)  
Regex flags (`/…/i`, `/…/m`) are discarded; the regex literal production captures only the body between slashes.  
**Fix:** Extend the regex literal lexer to capture trailing flag characters and expose them in `RegexLiteral`.

**24. HIGH** | `libs/surge/src/lexer.rs:315-321`  
`\u` Unicode escapes in string literals are not supported; only `\n`, `\r`, `\t`, `\"`, `\\` are handled.  
**Fix:** Add `\u{XXXX}` and `\uXXXX` escape sequences to the string lexer.

**25. HIGH** | `libs/surge/src/parser/expr.rs` (structural)  
`case … of … =>` conditional expressions are not implemented; the `Expr` AST has no `Case` variant.  
**Fix:** Add `Expr::Case { scrutinee, arms }` to the AST and implement the parser production.

**26. HIGH** | `libs/surge/src/parser/mod.rs` (structural)  
`flow` short-form declarations (`flow @src ~> @sink unless …`) have no parser support.  
**Fix:** Add `FlowDecl` AST node and parser production for flow shorthand.

**27. HIGH** | `libs/surge/src/parser/mod.rs` (structural)  
`label_decl` at document level has no parser support.  
**Fix:** Add `LabelDecl` to `Document` and implement inline label declaration parsing.

**28. HIGH** | `libs/surge/src/parser/expr.rs:424-427`  
Motif labeled edges (`$a ~kind~> $b`) hardcode `kind: "reaches"`; the lexer lacks support for custom edge kinds.  
**Fix:** Extend the lexer to tokenize `~ident~>` edge-kind syntax and parse it into `Expr::Arrow { kind: Option<Ident>, … }`.

**29. MEDIUM** | `libs/surge/src/lexer.rs` (structural)  
`union`, `intersect`, `complement` keywords are reserved but have no AST variants or parser branches.  
**Fix:** Either implement set-expression parsing for these keywords or remove them from the keyword table.

**30. MEDIUM** | `libs/surge/src/parser/expr.rs` (structural)  
`let … in …` expression (`Expr::LetIn`) exists in the AST but the parser never constructs it.  
**Fix:** Implement `let <binding> in <expr>` parser production.

**31. MEDIUM** | `libs/surge/src/ast/import_decl.rs` (structural)  
`ImportDecl` only stores `kind` and `path`; selective imports (`import { names } from "path"`) are not supported.  
**Fix:** Add `names: Option<Vec<Ident>>` to `ImportDecl` and parse selective import syntax.

**32. MEDIUM** | `libs/surge/src/lexer.rs` (structural)  
Hex (`0x`), octal (`0o`), and binary (`0b`) integer literals are not supported.  
**Fix:** Extend `read_number` to recognize radix prefixes.

**33. MEDIUM** | `libs/surge/src/lexer.rs` (structural)  
Scientific notation (`1e10`, `1.5e-3`) is not supported in numeric literals.  
**Fix:** Extend `read_number` to parse exponent components.

### 1.7 Missing Security / Malware / Supply-Chain Primitives

**34. CRITICAL** | `libs/surge/src/ast/expr.rs` (structural)  
No SBOM/dependency analysis primitives (`Package`, `Dependency`, `VersionConstraint`, `Purl`, `Cpe`). The language cannot express "flag if `log4j` < 2.15.0" or check lockfiles for known-vulnerable versions.  
**Fix:** Add AST literal variants for package identifiers and version constraints; add `dependency_of`, `package_name`, `version_less_than` predicates.

**35. CRITICAL** | `libs/surge/src/ast/expr.rs` (structural)  
No cryptographic integrity primitives (`HashLiteral`, `Signature`, `Certificate`, `Checksum`). The language cannot express code-signature verification, hash mismatch, or attestation checks.  
**Fix:** Add `HashLiteral` and `SignatureLiteral` expression variants with algorithm tagging.

**36. CRITICAL** | `libs/surge/src/ast/expr.rs` (structural)  
No malware binary heuristics (`EntropyLiteral`, `ImportHash`, `SectionName`, `Resource`). The language cannot express "PE section entropy > 7.0", YARA-style imphash matching, or MZ header checks.  
**Fix:** Add binary-section and entropy expression variants for YARA-style binary analysis.

**37. CRITICAL** | `libs/surge/src/ast/expr.rs` (structural)  
No network IoC literals (`IpAddrLiteral`, `UrlLiteral`, `DomainLiteral`). The lexer treats IPs/URLs as identifiers or numbers.  
**Fix:** Add dedicated lexer tokens and AST variants for IP addresses, URLs, and domain names.

**38. HIGH** | `libs/surge/src/ast/expr.rs` (structural)  
No permissions/capabilities primitives (`Permission`, `Capability`, `Entitlement`). The language cannot express "Android app requests `INTERNET` + `READ_SMS`" or iOS entitlement checks.  
**Fix:** Add `CapabilityLiteral` and `PermissionLiteral` expression variants for mobile/OS entitlement checking.

**39. HIGH** | `libs/surge/src/ast/expr.rs` (structural)  
No temporal/provenance primitives (`CommitHash`, `Provenance`, `Attestation`, `Age`). The language cannot express "finding introduced in commit abc123" or build-provenance gaps.  
**Fix:** Add `CommitHashLiteral` and `ProvenanceLiteral` variants for supply-chain provenance rules.

**40. HIGH** | `libs/surge/src/ast/expr.rs` (structural)  
No container/cloud primitives (`DockerfileInstruction`, `K8sResource`, `TerraformResource`). The language cannot express Dockerfile `FROM` latest-tag, K8s root-pod, or IaC misconfigurations.  
**Fix:** Add IaC-specific expression variants and predicates for Dockerfile, Kubernetes, and Terraform resources.

**41. HIGH** | `libs/surge/src/ast/expr.rs` (structural)  
No CI/CD build primitives (`BuildStep`, `CIStage`, `ArtifactProvenance`). The language cannot express "CI script downloads from untrusted URL" or unreproducible build steps.  
**Fix:** Add `BuildStepLiteral` and `CIStageLiteral` for pipeline analysis.

**42. HIGH** | `libs/surge/src/ast/exempt.rs` (structural)  
`Exemption` AST node is orphaned (no parser, no `Document` field), preventing native suppression authoring per spec §8.  
**Fix:** Wire `Exemption` into `Document` and implement the parser production.

---

## 2. Stdlib Rule Corpus (`libs/tools/surgec/rules/`)

### 2.1 Undefined Families → Dead Rules

**43. CRITICAL** | `libs/tools/surgec/rules/auth/hardcoded_credential.srg:7,11`  
References `@password_check_family` and `@credential_source_family` which have no label definition in `labels/`. The rule resolves to empty sets and will never fire.  
**Fix:** Create `labels/password_check_family.toml` and `labels/credential_source_family.toml` with language-mapped names, or delete the dead rule.

**44. CRITICAL** | `libs/tools/surgec/rules/auth/jwt_alg_none.srg:7,8,12`  
References `@jwt_decode_family`, `@alg_none_patterns`, `@jwt_secure_alg` with no label definitions.  
**Fix:** Create the three missing label TOMLs or inline the literal patterns into the rule.

**45. CRITICAL** | `libs/tools/surgec/rules/auth/session_fixation.srg:7,8,13`  
References `@authentication_success_family`, `@session_set_family`, `@session_regenerate_family` with no label definitions.  
**Fix:** Create the missing label TOMLs or rewrite the rule using existing families.

**46. CRITICAL** | `libs/tools/surgec/rules/crypto/ecb_mode.srg:7,10`  
References `@aes_cipher_family` and `@ecb_mode_literal` with no label definitions.  
**Fix:** Create `labels/aes_cipher_family.toml` and `labels/ecb_mode_literal.toml`.

**47. CRITICAL** | `libs/tools/surgec/rules/crypto/weak_password_hash.srg:7,8`  
References `@weak_hash_family` and `@password_input_family` with no label definitions.  
**Fix:** Create the missing label TOMLs.

**48. CRITICAL** | `libs/tools/surgec/rules/deserialize/pickle_of_untrusted.srg:8`  
References `@pickle_load_family` with no label definition.  
**Fix:** Create `labels/pickle_load_family.toml` mapping `pickle.load` etc. per language.

**49. CRITICAL** | `libs/tools/surgec/rules/kernel/copy_from_user_without_bound.srg:9`  
References `@user_input_family` which does not exist; should use `@receive_family` or `@http_input_family`.  
**Fix:** Replace `@user_input_family` with `@receive_family` or create the missing label TOML.

**50. CRITICAL** | `libs/tools/surgec/rules/memory/oob_read.srg:13,17,21`  
References `@array_access_family`, `@range_check_family`, `@rust_safe_indexing_family` with no label definitions.  
**Fix:** Create the three missing label TOMLs or rewrite using existing `pointer_use_family` / `length_clamp_family`.

**51. CRITICAL** | `libs/tools/surgec/rules/memory/uninit_read.srg:24`  
References `@rust_DA_family` with no label definition.  
**Fix:** Create `labels/rust_DA_family.toml` or remove the dead rule arm.

**52. CRITICAL** | `libs/tools/surgec/rules/tls/cert_verification_disabled.srg:7,8,10,11`  
References `@http_client_family`, `@verify_arg_slot`, `@false_literal`, `@insecure_context_literal` with no label definitions.  
**Fix:** Create the four missing label TOMLs or inline the checks using `literal_of` and `node_kind`.

**53. CRITICAL** | `libs/tools/surgec/rules/web/missing_auth.srg:12,13,17`  
References `@http_route_handler_family`, `@privileged_op_family`, `@auth_check_family` with no label definitions.  
**Fix:** Create the three missing label TOMLs or rewrite using existing families.

**54. CRITICAL** | `libs/tools/surgec/rules/malware/*.srg` (30 rules)  
All malware source/sink families (`@buffer_source`, `@cli_source`, `@credential_source`, `@file_source`, `@network_input_source`, `@npm_script_source`, `@sensitive_file_source`, `@shell_source`, `@system_source`, `@exec_sink`, `@file_sink`, `@network_sink`, `@sql_sink`, `@xss_sink`) are undefined.  
**Fix:** Create the 13 missing malware source/sink label TOMLs, or delete the 30 dead malware rules.

**55. CRITICAL** | `libs/tools/surgec/rules/stdlib/go_frontend.srg:4,12,20,28`  
References `@worker_family`, `@channel_family`, `@cleanup_family` with no label definitions.  
**Fix:** Create the three missing Go-family label TOMLs.

### 2.2 Structurally Broken Rules

**56. CRITICAL** | `libs/tools/surgec/rules/memory/uninit_read.srg:11-15`  
Logical contradiction: `$decl` is bound as `variable_decl` and `$read` as `variable_use`, then `require $decl == $read` demands the same node be both kinds simultaneously  -  impossible in any standard IR. This rule can never match.  
**Fix:** Rewrite the rule to relate the declaration and use via a dataflow predicate (`flows_to` or `dominates`) rather than node equality.

**57. CRITICAL** | `libs/tools/surgec/rules/web/redos.srg:17`  
Unbound variable `$n`: `let $pattern = literal_of($n, "regex")` references `$n`, which is never bound in this rule. Unbound variable → parse/evaluation failure.  
**Fix:** Bind `$n` in a preceding `let` (e.g., `let $n = all_nodes("regex_compile")`) or remove the rule.

### 2.3 Semantically Wrong Sanitizers

**58. HIGH** | `libs/tools/surgec/rules/web/log_injection.srg:21`  
Uses `@html_escape_family` as sanitizer for log injection. HTML escaping does **not** prevent CRLF/log injection.  
**Fix:** Replace with `@log_escape_family` or create a `labels/log_escape_family.toml` with language-mapped log sanitizers.

**59. HIGH** | `libs/tools/surgec/rules/web/prototype_pollution.srg:21`  
Uses `@html_escape_family` as sanitizer for prototype pollution. Wrong sanitizer family.  
**Fix:** Replace with `@object_freeze_family` or an object-sanitizer label; remove the incorrect sanitizer.

**60. HIGH** | `libs/tools/surgec/rules/web/redos.srg:23`  
Uses `@html_escape_family` as sanitizer for ReDoS. Wrong sanitizer family.  
**Fix:** Replace with `@regex_timeout_family` or a regex-sanitizer label; remove the incorrect sanitizer.

**61. HIGH** | `libs/tools/surgec/rules/web/template_injection.srg:22`  
Uses `@html_escape_family` as sanitizer for SSTI. Wrong sanitizer family.  
**Fix:** Replace with `@template_autoescape_family` or a template-sanitizer label.

**62. HIGH** | `libs/tools/surgec/rules/web/xxe.srg:21`  
Uses `@html_escape_family` as sanitizer for XXE. Wrong sanitizer family.  
**Fix:** Replace with `@xml_disable_dtd_family` or an XXE-specific sanitizer label.

### 2.4 Predicates Without Structural Lowering

**63. HIGH** | `libs/tools/surgec/rules/stdlib/dominates.srg:7,11`  
`dominates` and `post_dominates` are aliases for builtins (`dominator_tree_contains`, `post_dominator_tree_contains`) but have no structural lowering in the compiler.  
**Fix:** Lower `dominates($a, $b)` to a vyre dominator-tree query instead of relying on opaque builtins.

**64. HIGH** | `libs/tools/surgec/rules/stdlib/paths.srg:7`  
`exists_path` is an alias for `flows_to` with no structural lowering.  
**Fix:** Lower `exists_path` to a reachability query in the CFG adjacency matrix.

**65. HIGH** | `libs/tools/surgec/rules/stdlib/sanitized_by.srg:8`  
`sanitized_by` is a composition of `flows_to` + family membership with no structural lowering.  
**Fix:** Implement as a `flows_to($x, $y) && is_member($y, @sanitizer_family)` expansion in the lowerer.

**66. HIGH** | `libs/tools/surgec/rules/stdlib/taint_flow.srg:11,19`  
`taint_flow` and `taint_flow_unsanitized` are compositions with no structural lowering.  
**Fix:** Expand `taint_flow` to `flows_to($source, $sink) && !sanitized_by($source, $sink)` in the lowerer.

### 2.5 Coverage Gaps & Missing Metadata

**67. MEDIUM** | `libs/tools/surgec/rules/` (all `.srg` files)  
No rule file contains `test_inputs` blocks. Rules are shipped without positive/negative test coverage.  
**Fix:** Add `test_inputs { positive: [...], negative: [...] }` to every rule with representative code snippets.

**68. MEDIUM** | `libs/tools/surgec/rules/` (all `.srg` files)  
No `.srg` rule declares a `primitive` field; only `chains/*.toml` files declare primitives.  
**Fix:** Add `primitive = "..."` to each rule header or document that `.srg` rules derive primitives from chains.

**69. MEDIUM** | `libs/tools/surgec/rules/stdlib/` (structural)  
Stdlib provides zero predicates/rules for authz, crypto, concurrency, resource, numeric, binary, malware families, supply-chain, network, or cloud.  
**Fix:** Add category-specific stdlib predicates: `authz_missing`, `crypto_weak_mode`, `race_condition`, `resource_leak`, `numeric_overflow`, `binary_entropy`, `supply_chain_tamper`, `network_egress`, `cloud_misconfig`.

**70. LOW** | `libs/tools/surgec/rules/` (structural)  
`binary` and `cloud` vulnerability classes have zero rules anywhere in the rulebase.  
**Fix:** Author at least one seed rule per category (e.g., `binary/high_entropy_section.srg`, `cloud/s3_public_bucket.srg`).

---

## 3. Documentation vs Shipped AST

### 3.1 AUTHORING.md vs Compiler Surface

**71. CRITICAL** | `libs/tools/surgec/src/lower/mod.rs:670-673` vs `AUTHORING.md:13`  
`AggregateKind::Distinct` returns a hard error  -  "aggregate `distinct` still lacks a canonical dedup lowering"  -  despite being a valid AST variant. AUTHORING.md claims `compile/` and `lower/` turn SURGE AST into `vyre::Program` payloads.  
**Fix:** Implement lowering for `Distinct` to a vyre deduplication node, or remove the AST variant.

**72. CRITICAL** | `libs/tools/surgec/src/lower/mod.rs:674-678` vs `AUTHORING.md:13`  
`AggregateKind::GroupBy` returns a hard error  -  "aggregate `group_by` still lacks keyed output lowering"  -  despite being a valid AST variant.  
**Fix:** Implement keyed-output lowering for `GroupBy`, or remove the AST variant.

**73. HIGH** | `libs/tools/surgec/src/bin/regenerate-goldens.rs:52-55` vs `AUTHORING.md:41`  
Golden regeneration emits a temporary textual marker blob (`SURGEC-GOLDEN-V1`) instead of real wire artifacts, with an explicit comment: "Temporary textual marker blob until vyre::ir::Program::to_wire is wired through here." This violates AUTHORING.md invariant #4 (every lowering has a regression artifact).  
**Fix:** Wire `vyre::ir::Program::to_wire()` into the regenerate binary and emit canonical wire-format goldens.

**74. HIGH** | `libs/tools/surgec/src/bin/regenerate-goldens.rs:20` vs `AUTHORING.md:41`  
Hard-coded category list `["stdlib", "memory", "web", "malware"]` skips `tls/`, `crypto/`, `kernel/`, `deserialize/`, `auth/`, `chains/`.  
**Fix:** Iterate over all subdirectories of `rules/` dynamically instead of hard-coding categories.

**75. HIGH** | `libs/tools/surgec/src/scan/dispatch.rs:190-201`, `src/scan/distributed.rs:185-188`, `src/scan/collector.rs:325-328`, `src/scan/confidence.rs:127-128,141-142`, `src/scan/auto_suppress.rs:79`, `src/output/explainer.rs:63-67`, `src/output/tfidf.rs:67` vs `AUTHORING.md:71`  
`Finding` and `FileFinding` struct literals are constructed outside `scan::test_builders` in at least eight locations, violating AUTHORING.md invariant: "No partial Finding / FileFinding constructors outside scan::test_builders."  
**Fix:** Replace every direct struct literal with the appropriate builder method from `scan::test_builders`.

### 3.2 SCHEMA.md vs Certificate / Runner Code

**76. MEDIUM** | `libs/performance/matching/vyre/vyre-conform-runner/src/cert.rs:12,22,26,30` vs `rules/op/SCHEMA.md`  
`Certificate` struct field names mismatch the published schema: `version` vs `cert_version`, `backend_id` vs `allowed_backends`, `laws_verified` vs `laws`, `signature_ed25519` vs `signature_blake3`.  
**Fix:** Rename struct fields to match the published SCHEMA.md contract.

**77. MEDIUM** | `libs/performance/matching/vyre/vyre-conform-runner/src/cert.rs:63-65` vs `rules/op/SCHEMA.md:28-30`  
SCHEMA.md mandates byte-identical canonical TOML output, but the runner implements only `to_json()` using `serde_json::to_string_pretty`.  
**Fix:** Implement `to_toml()` using a canonical TOML serializer with ordered fields and trailing newline enforcement.

**78. MEDIUM** | `libs/performance/matching/vyre/vyre-conform-runner/src/cert.rs:45` vs `rules/op/SCHEMA.md` header  
Hard-coded certificate version `"0.6.0"` contradicts SCHEMA.md header v0.5.0 and on-disk TOML stubs.  
**Fix:** Unify on a single version constant shared between runner, schema, and stubs.

**79. MEDIUM** | `libs/performance/matching/vyre/vyre-conform-runner/src/cert.rs:10-33` vs `rules/op/SCHEMA.md:23-24`  
`Certificate` struct has no `extensions` map; serde rejects unknown fields instead of ignoring them per SCHEMA.md extension-table policy.  
**Fix:** Add `extensions: BTreeMap<String, toml::Value>` with `#[serde(default)]` and `flatten`.

**80. MEDIUM** | `libs/performance/matching/vyre/vyre-conform-runner/src/main.rs:265-386`  
The `prove` subcommand emits a `ProveArtifact` JSON blob and never constructs or emits the per-op `Certificate` struct defined in `cert.rs`, and never writes TOML.  
**Fix:** Make `prove` construct and emit canonical TOML `Certificate` files per op.

---

## Summary by Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 24 |
| HIGH     | 32 |
| MEDIUM   | 21 |
| LOW      |  3 |
| **Total**| **80**|
