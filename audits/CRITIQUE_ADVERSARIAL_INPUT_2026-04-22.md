# CRITIQUE_ADVERSARIAL_INPUT_2026-04-22

Scope: `libs/tools/surgec` lowering, decode, rule corpus, and SURGE parsing paths exercised by `vyre`.

Method: hostile-input audit only. I looked for ways to make shipped rules miss attacker-controlled payloads, crash/hang the scanner, or emit wrong findings on crafted source/rule inputs.

Verification note: `cargo test -p surgec --test label_loading load_all_labels -- --nocapture` is currently blocked by an unrelated existing workspace compile failure in `libs/performance/matching/vyre/vyre-foundation/src/lib.rs:97` (`DataTypeSizeBytes` import does not resolve). The findings below are therefore code-audit findings, not post-fix runtime confirmations.

## Findings

1. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:185-205` | Base64 decode only accepts one contiguous alphabet run, so newline-split MIME/base64 payloads bypass every rule that relies on decode recursion.
Crafted input: `b"cGlja2xlLmxv\nYWRzKGRhdGEp"` or `b"ZXZhbA==\r\nKDEp"`.
Why: `extract_base64_regions` stops the run on `\r`/`\n`, then tries to decode each fragment independently. Common attacker base64 with line folding never reaches nested rule inspection.
Fix: normalize RFC 2045 whitespace inside candidate base64 runs before decode, and preserve a source-to-decoded offset map so findings still point back to the original bytes.

2. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:176-181` | `base64_nopad` expansions are emitted by the compiler but the decoder always uses a padded engine, so unpadded nested payloads are invisible.
Crafted input: `b"cGlja2xlLmxvYWRzKGRhdGEp"` with no trailing `=`.
Why: `expand_literal_variants` emits `base64_nopad`, but `extract_base64_regions` only decodes with `GeneralPurpose(..., PAD)`. A nested payload that only exists in unpadded form is missed.
Fix: try both padded and no-pad engines for each base64 alphabet, or select the engine from the candidate shape before decode.

3. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:176-181` | `base64url_nopad` has the same blind spot for URL-safe attacker payloads.
Crafted input: `b"X19wcm90b19f"` or `b"cGlja2xlLmxvYWRzKGRhdGEp"` encoded with `URL_SAFE_NO_PAD`.
Why: the URL-safe alphabet is selected, but the engine still requires padding. Attackers can hide nested sink strings under base64url without `=`.
Fix: support `URL_SAFE_NO_PAD` in decode parity with the compiler’s variant expansion list.

4. HIGH | `libs/tools/surgec/src/scan/decode.rs:385-414` | URL decode only recognizes runs that are entirely `%HH%HH...`; mixed literal-plus-escaped payloads bypass lowering.
Crafted input: `b"pickle%2Eloads(data)"`, `b"__proto__%5Bpolluted%5D"`.
Why: the extractor starts at `%` and keeps decoding only while the next token is also `%HH`. Real payloads commonly mix plain ASCII and escapes, so the decoded token stream is never reconstructed.
Fix: implement a percent-decoder over mixed runs, emitting decoded bytes plus literal passthrough bytes while tracking the original span map.

5. HIGH | `libs/tools/surgec/src/scan/decode.rs:151-169` and `385-414` | `form_urlencoded` variants are emitted, but decode never converts `+` back to space.
Crafted input: `b"curl+http%3A%2F%2Fevil"` or `b"ProcessBuilder+%2Fbin%2Fsh"`.
Why: the compiler explicitly emits `form_urlencoded`, but `extract_url_encoded_regions` ignores `+`, so patterns requiring spaces or argument separation do not reconstruct.
Fix: add a form-urlencoded decode mode that maps `+` to `0x20` and handles mixed literal/escaped segments.

6. HIGH | `libs/tools/surgec/src/scan/decode.rs:132-133` and `407-408` | `MIN_DECODED_SIZE == 16` creates a hard blind spot for short dangerous payloads hidden in decode layers.
Crafted input: `b"ZXZhbCgxKQ=="` (`eval(1)`), `b"X19wcm90b19f"` (`__proto__`), `b"UEsDBA=="` (`PK\x03\x04`).
Why: short decoded fragments are dropped before scanning. Many exploit-relevant atoms are shorter than 16 bytes.
Fix: gate on semantic usefulness instead of a global minimum, or scan all successfully decoded fragments while capping total work via per-file budgets.

7. HIGH | `libs/tools/surgec/src/scan/decode.rs:208-247` and `libs/tools/surgec/src/compile/signals/literal_family.rs:22-34` | The compiler advertises `unicode_escape`, but decode never reconstructs `\uXXXX`.
Crafted input: `b"\\u0070\\u0069\\u0063\\u006b\\u006c\\u0065\\u002e\\u006c\\u006f\\u0061\\u0064\\u0073"`.
Why: rule authors can request `unicode_escape`, but nested content encoded this way is never lowered back into raw bytes for scanning.
Fix: implement `\uXXXX` decoding in `decode.rs`, including validation of surrogate pairs and source span tracking.

8. HIGH | `libs/tools/surgec/src/scan/decode.rs:208-247` and `libs/tools/surgec/src/compile/signals/literal_family.rs:24-25` | `unicode_brace_escape` is declared but never decoded.
Crafted input: `b"\\u{0070}\\u{0069}\\u{0063}\\u{006b}\\u{006c}\\u{0065}"`.
Why: Rust/JS-style brace escapes remain opaque bytes, so nested sink/source markers encoded this way bypass detection.
Fix: add brace-escape decoding with strict range checks and invalid-sequence rejection.

9. HIGH | `libs/tools/surgec/src/scan/decode.rs:208-247` and `libs/tools/surgec/src/compile/signals/literal_family.rs:26-27` | `unicode_surrogate_escape` is declared but never decoded.
Crafted input: `b"\\uD83D\\uDCA5"` or split sink tokens such as `b"\\u0070\\u0069\\u0063\\u006b\\u006c\\u0065"`.
Why: high/low surrogate pairs are never combined, so author-declared variants cannot match nested payloads that use surrogate spelling.
Fix: add surrogate-pair assembly and reject unpaired surrogates at decode time.

10. HIGH | `libs/tools/surgec/src/compile/signals/literal_family.rs:28` and `libs/tools/surgec/src/compile/expand.rs:121-125` | `string_from_char_code` is advertised as supported, but compilation fails at expansion time.
Crafted input: `signal s: literal_family(values: ["pickle.loads"], variants: ["string_from_char_code"])`.
Why: a rule author can ship a “valid” rule that passes signal validation then hard-fails during expansion. That is an easy corpus-level compiler bomb.
Fix: either implement the transform or remove it from `SUPPORTED_VARIANTS`; the validator and expander must be identical by construction.

11. HIGH | `libs/tools/surgec/src/compile/signals/literal_family.rs:29` and `libs/tools/surgec/src/compile/expand.rs:121-125` | `rot13` is declared but not implemented, so a single rule can weaponize compile-time failure.
Crafted input: `signal s: literal_family(values: ["cvpxyr.ybnqf"], variants: ["rot13"])`.
Why: validation accepts the rule, expansion aborts the whole compile.
Fix: generate the transform table during expansion or reject `rot13` during validation until it exists.

12. HIGH | `libs/tools/surgec/src/compile/signals/literal_family.rs:30-32` and `libs/tools/surgec/src/scan/decode.rs:208-247` | HTML entity variants are declared but decode never reconstructs them.
Crafted input: `b"&#x70;&#x69;&#x63;&#x6b;&#x6c;&#x65;"`, `b"&#112;&#105;&#99;&#107;&#108;&#101;"`.
Why: nested payloads written as HTML entities remain invisible even when a rule explicitly claims support for `html_entity_hex` or `html_entity_decimal`.
Fix: implement HTML entity decoding over both hex and decimal forms before scanning.

13. HIGH | `libs/tools/surgec/src/compile/signals/literal_family.rs:33` and `libs/tools/surgec/src/scan/decode.rs:208-247` | `octal_escape` is declared but decode never handles `\123` style bytes.
Crafted input: `b"\\160\\151\\143\\153\\154\\145\\056\\154\\157\\141\\144\\163"`.
Why: classic C/PHP attacker spellings remain opaque.
Fix: add octal escape decoding with width limits and invalid-digit handling.

14. HIGH | `libs/tools/surgec/src/compile/signals/literal_family.rs:34` and `libs/tools/surgec/src/scan/decode.rs:208-247` | `fullwidth` is declared but no lowering path ever normalizes fullwidth homoglyphs.
Crafted input: `b"\xef\xbd\x90\xef\xbd\x89\xef\xbd\x83\xef\xbd\x8b\xef\xbd\x8c\xef\xbd\x85"` (`ｐｉｃｋｌｅ`).
Why: the compiler promises a variant that the scanner cannot reconstruct, so fullwidth-obfuscated payloads evade detection.
Fix: implement NFKC/fullwidth fold as an explicit transform with byte-to-byte provenance.

15. HIGH | `libs/tools/surgec/src/scan/decode.rs:141-156` | Nested decode provenance uses the child-local `offset` when recursing instead of the cumulative `parent_offset`, so nested findings point at the wrong bytes.
Crafted input: `b"prefix " + base64(gzip(base64(b\"pickle.loads(data)\")))`.
Why: recursive calls pass `offset`, not `parent_offset + offset`. A finding in the grandchild layer is attributed to the wrong source window, which can make triage and suppression logic incorrect.
Fix: recurse with the cumulative absolute parent offset and maintain a layered provenance stack rather than a single parent tuple.

16. HIGH | `libs/tools/surgec/src/scan/decode.rs:416-437` | The gzip extractor retries a `GzDecoder` at every `1f 8b` signature, so repeated fake headers produce quadratic-ish decompression attempts.
Crafted input: `b\"\\x1f\\x8b\" * 500000 + b\"junk\"`.
Why: each two-byte magic hit creates a new decoder over the remaining suffix. Corrupt tails still pay setup and partial-read cost.
Fix: validate the gzip header before constructing a decoder, skip forward past invalid headers, and enforce a decode-attempt budget per file.

17. CRITICAL | `libs/tools/surgec/src/scan/decode.rs:166-170` | Decode recursion only supports base64, hex, urlencoded, and gzip. The scope mentions zlib/zip/tar-style nesting, but those formats are invisible.
Crafted input: `b\"eJyrVkrLz1eyUkpKLFKqBQA5WQYJ\"` (zlib), `b\"PK\\x03\\x04...\"` (zip).
Why: an attacker can split a payload across common archive/container layers that the scanner never opens.
Fix: extend decode extraction with zlib/deflate, zip, and tar readers behind explicit budgets and depth accounting.

18. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:572-580` | Per-signal matches are truncated to `MAX_CACHED_POSITIONS`, so a flood of decoy hits can evict the one match that matters.
Crafted input: a file with 256 innocuous `redirect(` or `pickle` strings before the real sink/source pair.
Why: only the first 256 hits survive; later offsets are silently dropped. Hostile files can front-load harmless matches and suppress the security-relevant one.
Fix: rank or window hits by structural relevance, or spill match offsets into a dynamically sized buffer instead of truncating to a fixed first-N cache.

19. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:637-655` | Mixed literal+regex signal families lose regex semantics because the regex path is only chosen when every pattern in the family is regex-backed.
Crafted input: any rule/signal set mixing a regex guard with literal variants, e.g. `["(?i)pickle\\s*\\.\\s*loads", "pickle.loads"]`.
Why: once one literal is present, the family falls into DFA/literal mode and regex patterns are treated as raw bytes. Attackers can vary whitespace/case/comment gaps and bypass the intended regex arm.
Fix: split mixed families into literal and regex subplans and evaluate both, unioning the hit sets before dispatch.

20. HIGH | `libs/tools/surgec/src/scan/collector.rs:248-286` | Every file is read wholly into memory, then every decoded layer is materialized before applicability filtering. Archive bombs and giant binaries become memory bombs.
Crafted input: a multi-GB file or a gzip that expands to the `max_total_bytes` budget across many sibling layers.
Why: scan cost is paid before any rule-specific relevance check. A hostile corpus can OOM the scanner with irrelevant data.
Fix: stream file reads, evaluate applicability before deep decode, and feed layers lazily into rule dispatch.

21. HIGH | `libs/tools/surgec/src/scan/collector.rs:771-813` | Regex-backed signals are compiled per file per pattern, making rule regexes an attacker-controlled CPU tax.
Crafted input: 10k small files against a ruleset with many regex-backed families.
Why: the engine recompiles the same regex for each file instead of caching compiled byte regexes by source.
Fix: cache compiled regexes by pattern bytes and reuse them across files and clauses.

22. CRITICAL | `libs/tools/surgec/rules/labels/redirect_sink_family.toml:44-50` and `libs/tools/surgec/src/lower/mod.rs:1655-1679` | `redirect_sink_family.toml` contains an unescaped backslash sequence, poisoning label loading for every rule that touches label families.
Crafted input: any rule compile that references `@redirect_sink_family`, or any compile path that initializes the global label map.
Why: `LabelSet::load_dir` must parse every label file. `Symfony\Component...` is invalid TOML string syntax here, so the whole label registry can fail to initialize.
Fix: escape backslashes or use single-quoted TOML literal strings, and add a build-time test that parses every label file before merge.

23. CRITICAL | `libs/tools/surgec/rules/labels/template_render_family.toml:40-46` and `libs/tools/surgec/src/lower/mod.rs:1655-1679` | `template_render_family.toml` has the same TOML parse poison via `Twig\Environment::render`.
Crafted input: any compile of `web/template_injection.srg` or any path that loads label masks globally.
Why: one malformed shipped family can brick the whole label-family subsystem.
Fix: same as above, plus add CI that loads all label TOMLs in isolation and as a directory set.

24. HIGH | `libs/tools/surgec/rules/labels/redirect_sink_family.toml:47-48` | `wp_safe_redirect` is listed as an unsafe redirect sink, so the shipped `open_redirect` rule can fire on WordPress’s safe API.
Crafted input: `b"wp_safe_redirect($url)"`.
Why: the family marks a mitigation API as a sink, creating wrong findings that hide real defects under noise.
Fix: remove `wp_safe_redirect` from the sink family and model it as a sanitizer/allowlist boundary instead.

25. HIGH | `libs/tools/surgec/rules/labels/template_render_family.toml:46` | `eval` is classified as a template renderer, so `template_injection` can mislabel generic code execution as SSTI.
Crafted input: `b"eval(user_input)"`.
Why: this is a different bug class with different remediation. The current family collapses them.
Fix: move `eval` into a code-execution sink family and keep template rendering families semantically strict.

26. HIGH | `libs/tools/surgec/rules/labels/external_entity_family.toml:14-15` | `defusedxml.lxml.parse` and `defusedxml.lxml.fromstring` are safe wrappers, but the XXE sink family treats them as vulnerable sinks.
Crafted input: `b"defusedxml.lxml.fromstring(request.data)"`.
Why: the shipped XXE rule will flag a mitigation library as an XXE sink.
Fix: remove safe wrappers from the sink family and introduce an XXE-safe family for sanitizer modeling if needed.

27. MEDIUM | `libs/tools/surgec/rules/labels/external_entity_family.toml:25` | `fast-xml-parser.parse` is treated as an XXE sink even though it is not an entity-resolving libxml-style parser in the same threat model.
Crafted input: `b"fastXmlParser.parse(req.body)"`.
Why: wrong-family labeling inflates false positives and erodes trust.
Fix: tighten the family to parsers that can actually resolve external entities.

28. MEDIUM | `libs/tools/surgec/rules/labels/external_entity_family.toml:68` | `quick_xml::Reader::read_event` is modeled as an XXE sink despite `quick-xml` not behaving like a default external-entity resolver.
Crafted input: `b"quick_xml::Reader::read_event(&mut buf)"`.
Why: this flags benign Rust XML parsing as XXE.
Fix: remove it unless there is a demonstrated vulnerable configuration path; model actual risky XML libraries instead.

29. HIGH | `libs/tools/surgec/rules/labels/url_validation_family.toml:49` | `URI::DEFAULT_PARSER.escape` is classified as URL validation, letting open redirects look sanitized when they are only escaped.
Crafted input: `b"redirect_to(URI::DEFAULT_PARSER.escape(params[:next]))"`.
Why: escaping does not prove host/scheme safety. The rule can be bypassed by passing attacker URLs through a string escaper.
Fix: split “parsing/escaping” from “allowlist validation” and only treat explicit allowlist/check APIs as sanitizers.

30. HIGH | `libs/tools/surgec/rules/labels/url_validation_family.toml:57` | `java.net.URLConnection` is treated as validation even though it initiates outbound access rather than validating targets.
Crafted input: `b"new URLConnection(userUrl); response.sendRedirect(userUrl)"`.
Why: the sanitizer model is semantically wrong and can suppress real open-redirect/SSRF findings.
Fix: remove transport/client APIs from validation families and model them as sinks where appropriate.

31. MEDIUM | `libs/tools/surgec/rules/labels/html_escape_family.toml:35` | The Rust sanitizer entry `" ammonia::clean"` has a leading space, so it will never match and safe code will still be flagged.
Crafted input: `b"ammonia::clean(user_html)"`.
Why: the family contains a typo that turns a mitigation into a false positive.
Fix: trim and canonicalize label entries on load; reject leading/trailing whitespace in CI.

32. HIGH | `libs/tools/surgec/rules/labels/html_escape_family.toml:22` | `DOMPurify.sanitize` is used as a universal sanitizer family, which suppresses non-HTML bug classes it does not actually fix.
Crafted input: `b"logger.info(DOMPurify.sanitize(req.query.q))"`, `b"merge(obj, DOMPurify.sanitize(req.body))"`.
Why: HTML sanitization does not make log data CRLF-safe, prototype-pollution-safe, or regex-safe. This produces false negatives across multiple rules.
Fix: split sanitizer families by vulnerability class instead of reusing `html_escape_family` as a catch-all.

33. MEDIUM | `libs/tools/surgec/rules/labels/html_escape_family.toml:36` | `rocket::response::content::Html` is classified as a sanitizer even though it is a response wrapper.
Crafted input: `b"Html(user_input)"`.
Why: wrapping content in an HTML responder does not sanitize attacker-controlled bytes.
Fix: remove renderer/output wrappers from sanitizer families.

34. HIGH | `libs/tools/surgec/rules/web/redos.srg:23` | `redos.srg` uses `@html_escape_family` as the sanitization boundary for regex DoS.
Crafted input: `b"Regex::new(req.query.pattern); regex.is_match(req.query.input)"` after `html.escape`.
Why: HTML escaping does not prevent catastrophic regex behavior. The rule can wrongly suppress a true ReDoS path.
Fix: add a regex-specific sanitizer/boundary family, or require structural proof of bounded regex/input handling instead of HTML escaping.

35. HIGH | `libs/tools/surgec/rules/web/xxe.srg:21` | `xxe.srg` also treats HTML escaping as an XXE mitigation.
Crafted input: `b"parser.parse(html.escape(request.body))"`.
Why: escaping `<`/`>` for HTML is not a proof that DTD/entity resolution is disabled. The rule can miss true XXE.
Fix: model XXE mitigations explicitly: parser features that disable entity resolution, safe wrappers, or trusted-source proofs.

36. HIGH | `libs/tools/surgec/rules/web/log_injection.srg:21` | `log_injection.srg` treats HTML escaping as CRLF sanitization.
Crafted input: `b"logger.info(html.escape(req.query.q))"` where `q` contains `%0d%0a`.
Why: HTML escaping does not remove or normalize newline/control characters.
Fix: require CRLF/control-character normalization sanitizers, not HTML escaping.

37. HIGH | `libs/tools/surgec/rules/web/prototype_pollution.srg:21` | `prototype_pollution.srg` uses HTML escaping as a mitigation for attacker-controlled object keys.
Crafted input: `b"merge(target, htmlEscape(req.body))"` with key `__proto__`.
Why: HTML escaping does not neutralize dangerous property names.
Fix: require key allowlisting / dangerous-key stripping families for this rule.

38. CRITICAL | `libs/tools/surgec/rules/deserialize/pickle_of_untrusted.srg:8` and `libs/tools/surgec/src/lower/mod.rs:1675-1679` | `pickle_of_untrusted` references `@pickle_load_family`, but there is no shipped label file for it.
Crafted input: `b"pickle.loads(request.data)"`.
Why: the rule cannot lower to a real label mask, so the core Python deserialization RCE sink is effectively blind.
Fix: ship the missing family and add compile-time validation that every referenced family exists before a rule is accepted.

39. HIGH | `libs/tools/surgec/rules/memory/unbounded_recursion.srg:17` and `libs/tools/surgec/rules/labels/receive_family.toml:7-120` | `unbounded_recursion` declares eight languages but `receive_family` only covers C/C++/Rust, so the Python/JS/Go/PHP/Ruby/Java arms are dead.
Crafted input: `b"def f(x): return f(x)\\nf(request.args.get('n'))"`.
Why: the source family is absent for most declared languages, so the taint path never starts.
Fix: validate family coverage per declared language and reject rules that claim languages with no family support.

40. HIGH | `libs/tools/surgec/rules/crypto/insecure_random.srg:18-21` and `libs/tools/surgec/rules/labels/token_generator_family.toml:7-75` | `insecure_random` declares `c`, but `token_generator_family` has no C entries, making the C variant unreachable.
Crafted input: `b"char *tok(){ return make_token(rand()); }"`.
Why: the rule claims C coverage it does not have.
Fix: either add real C token-generator labels or drop `c` from the rule until coverage exists; enforce this in validation.

41. HIGH | `libs/tools/surgec/rules/web/command_injection.srg` and `libs/tools/surgec/rules/labels/http_input_family.toml:6-97` | The web rules declare `c`, but `http_input_family` has no C section, so the C arms are dead on arrival.
Crafted input: `b"system(getenv(\\\"QUERY_STRING\\\"))"` or a C HTTP framework source.
Why: the rule language list outruns the family data.
Fix: enforce per-language family completeness at build time and stop shipping languages with no source/sink coverage.

42. HIGH | `libs/tools/surgec/rules/web/command_injection.srg` and `libs/tools/surgec/rules/labels/shell_escape_family.toml:6-50` | `shell_escape_family` likewise has no C entries, so command-injection mitigation modeling for C is fictional.
Crafted input: `b"execl(\\\"/bin/sh\\\", \\\"sh\\\", \\\"-c\\\", shellEscape(arg), NULL)"`.
Why: even if the source side were fixed, the rule cannot reason about C shell escaping.
Fix: add real C escaping APIs or remove unsupported languages from the rule.

43. MEDIUM | `libs/tools/surgec/rules/memory/use_after_free.srg:25-27` and `libs/tools/surgec/rules/labels/rust_safe_drop_family.toml:8-16` | A C/C++ rule exempts hits using a Rust-only family, which is semantically dead and suggests the rule was copied without language reconciliation.
Crafted input: `b"free(p); use(p);"` in C.
Why: the exemption never fires for the rule’s declared languages, so the rule corpus contains dead semantic branches.
Fix: prune language-incompatible clauses during validation and reject rules that reference families outside their language set.

44. CRITICAL | `libs/tools/surgec/src/lower/mod.rs:97-111` | `lower_rule` ignores `report { ... }` entirely, so binding references that only appear in `report.primary` / `report.related` are never validated.
Crafted input: `b"rule r { report { primary: $ghost, related: [] } }"`.
Why: a malformed or attacker-supplied rule can smuggle nonexistent bindings through compile-time checks, then fail later or emit nonsensical findings.
Fix: validate and lower report bindings with the same scope rules as `let`/`require`.

45. CRITICAL | `libs/tools/surgec/src/lower/mod.rs:145-149` | `Expr::BindingRef` lowers straight to `Expr::var(name)` without proving the binding exists in scope.
Crafted input: `b"rule r { let $a = $ghost report { primary: $a } }"`.
Why: undefined variables survive lowering and become runtime IR references instead of compile errors.
Fix: maintain a lexical scope table in the lowerer and reject any unresolved binding reference immediately.

46. HIGH | `libs/tools/surgec/src/lower/mod.rs:186-193` | `LetIn` absorbs the initializer globally, then lowers the body without a scope frame, so sibling/self references can miscompile.
Crafted input: `b"let $x = $x in $x"` or `b"let $x = $y in let $y = foo() in $x"`.
Why: there is no environment stack or shadowing model. Circular/self references are not detected and sibling visibility is wrong.
Fix: lower `let` with explicit lexical environments, shadowing rules, and cycle detection before absorption.

47. CRITICAL | `libs/tools/surgec/src/lower/mod.rs:371-394` | User predicate lowering has no recursion guard, so self-recursive or mutually recursive predicates can stack overflow the compiler.
Crafted input: `b"predicate p(x) = p(x)\\nrule r { require p(foo()) }"`.
Why: `lower_user_predicate_call` rewrites the body and immediately recurses back into `lower_expr` forever.
Fix: track the active predicate call stack and reject cycles with a compile error that names the loop.

48. HIGH | `libs/surge/src/lib.rs:91-100` | `parse_file` uses `read_to_string`, so a UTF-8 BOM or mixed binary/SURGE file is rejected before the parser can reason about it.
Crafted input: `b"\\xef\\xbb\\xbfsurge = \\\"3\\\"\\nrule r { ... }"` or a >10 MB mostly-text rule file with one non-UTF-8 byte.
Why: attackers can weaponize a single BOM/non-UTF-8 byte to brick rule ingestion.
Fix: read raw bytes, strip BOM explicitly, and surface precise byte-offset decode errors instead of all-or-nothing UTF-8 failure.

49. MEDIUM | `libs/surge/src/lexer.rs:381-386` | Identifiers are ASCII-only, so Unicode lookalikes and mixed-script collisions are rejected at lex time instead of normalized or diagnosed intentionally.
Crafted input: `b"rule r { let $p\\u{0430}th = foo() }"` with Cyrillic `а`.
Why: this is currently an abrupt parse failure rather than a deliberate confusable-identifier defense, and the parser gives no reserved-word/confusable guidance.
Fix: either explicitly normalize and ban confusables with a targeted diagnostic, or fully support Unicode identifiers with NFKC normalization and collision checks.

50. MEDIUM | `libs/surge/src/lexer.rs:306-323` | String escapes treat unknown escapes as literal characters, so `\\uXXXX` in SURGE source does not mean what rule authors expect.
Crafted input: `b"signal s: exact(values: [\\\"\\\\u0070\\\\u0069\\\\u0063\\\\u006b\\\\u006c\\\\u0065\\\"])"`.
Why: the lexer turns `\\u0070` into `u0070`, silently changing rule meaning.
Fix: either implement standard Unicode escapes in string literals or reject unknown escapes instead of passing them through.

51. HIGH | `libs/surge/src/parser/mod.rs:33-40` | The parser stores the entire token stream in a `Vec`, so extreme-size rule files become easy memory-exhaustion attacks.
Crafted input: a 10-50 MB SURGE file with millions of trivial rules or separators.
Why: there is no streaming parse, no token budget, and no structural early cutoff.
Fix: add parser/token budgets and a streaming or chunked parse path for very large rule corpora.

52. HIGH | `libs/tools/surgec/src/lower/mod.rs:1655-1673` | Label-family masks are packed into `u32`, so the 33rd shipped label family becomes a hard compile failure for the whole scanner.
Crafted input: adding one more `rules/labels/*.toml` file or loading a corpus with >32 families.
Why: the design couples label growth to a 32-bit mask with no migration path.
Fix: move to a wider or segmented tag representation before adding more families, and validate capacity at rule-build time instead of at runtime.

53. HIGH | `libs/tools/surgec/src/scan/decode.rs:421-433` | Corrupt gzip members after a valid header can still force bounded but repeated decompression work across many offsets, creating an attacker-controlled hang lane.
Crafted input: `b"\\x1f\\x8b\\x08\\x00" + b"A"*18 + b"\\x1f\\x8b\\x08\\x00" + ...` repeated.
Why: the current loop never memoizes failed offsets or header checks, so each repeated near-header incurs work.
Fix: record failed header offsets and skip them on the next scan, and validate header flags/CRC structure before inflating.

54. HIGH | `libs/tools/surgec/src/scan/decode.rs:272-307` and `354-355` | Delimited hex decoding only accepts a tiny separator set, so attacker spellings with quotes, slashes, or concatenation operators bypass nested reconstruction.
Crafted input: `b"0x70|0x69|0x63|0x6b|0x6c|0x65"`, `b"0x70+0x69+0x63..."`.
Why: only whitespace, comma, semicolon, and colon are accepted separators. Real malware and obfuscators use many more joiners.
Fix: either normalize a broader family of safe structural separators or decode token-wise through a language-aware lexer stage.

55. MEDIUM | `libs/tools/surgec/src/scan/decode.rs:373-380` | Base64 candidate detection accepts `=` anywhere in the run, which lets hostile junk stretches create repeated failed decode attempts.
Crafted input: `b"=============================AAAA"`.
Why: `is_b64_char_for` treats `=` as a generic run character, but valid base64 only allows padding at the tail. Attackers can inflate futile decode attempts.
Fix: pre-validate base64 shape before calling the decoder and reject runs with interior padding.

56. HIGH | `libs/tools/surgec/src/scan/collector.rs:309-347` | Every decode layer is scanned against every dispatch plan even when the layer encoding is semantically irrelevant to the rule, so hostile multi-layer files amplify scan cost multiplicatively.
Crafted input: a file containing many nested base64/hex/gzip fragments plus a large unrelated code body.
Why: the engine lacks a rule-to-layer relevance gate.
Fix: attach decode requirements to compiled plans and skip plans that cannot gain evidence from a given layer.

57. HIGH | `libs/tools/surgec/src/scan/collector.rs:533-539` and `656-662` | File size is narrowed into `u32`, so files above `u32::MAX` hard-fail the whole clause path instead of streaming or sharding.
Crafted input: a >4 GiB sparse or generated file in the scan corpus.
Why: a single large file becomes a denial-of-service against scanning.
Fix: support chunked scanning and 64-bit metadata, or skip oversize files with an explicit finding rather than failing the rule path.

58. MEDIUM | `libs/tools/surgec/src/compile/expand.rs:41-57` | `literal_family` dedupes by transformed bytes only, so two distinct variants that collapse to the same bytes lose provenance and can no longer explain which obfuscation path was matched.
Crafted input: a signal with both `hex` and `hex_upper` over an all-digit payload, or `urlencoded` vs `urlencoded_lower` over alphanumeric-only bytes.
Why: an attacker can exploit collapsed provenance to make findings harder to attribute and debug.
Fix: preserve variant provenance even when transformed bytes are identical, or dedupe after attaching a variant set to the pattern.

59. HIGH | `libs/tools/surgec/rules/web/open_redirect.srg:18-21` | `open_redirect` depends on a poisoned sink family and a semantically weak sanitizer family at the same time, so both false negatives and false positives are currently possible.
Crafted input: `b"return RedirectResponse(URI::DEFAULT_PARSER.escape(params[:next]))"` and `b"wp_safe_redirect($_GET['next'])"`.
Why: one side suppresses real defects as “validated”; the other side flags safe APIs as vulnerable.
Fix: repair `redirect_sink_family`, tighten `url_validation_family` to true validation/allowlist APIs, and add tests for both adversarial examples.

60. CRITICAL | `libs/tools/surgec/src/compile/signals/literal_family.rs:6-35` and `libs/tools/surgec/src/compile/expand.rs:64-130` | The validator/expander contract is broken for fourteen advertised variants, which means rule authors can compile-bomb the engine with inputs that are “supported” according to validation.
Crafted input: any rule using `unicode_escape`, `unicode_escape_upper`, `unicode_brace_escape`, `unicode_brace_escape_upper`, `unicode_surrogate_escape`, `unicode_surrogate_escape_upper`, `string_from_char_code`, `rot13`, `html_entity_hex`, `html_entity_hex_upper`, `html_entity_decimal`, `octal_escape`, or `fullwidth`.
Why: this is not one bug; it is a systemic integrity failure between signal validation and lowering.
Fix: generate both validation and expansion support from one shared enum/table so new variants cannot exist in one path without the other.

## Bottom line

The largest correctness gaps are not cosmetic:

1. Decode recursion does not match the compiler’s advertised obfuscation surface.
2. The shipped rule corpus contains poisoned and semantically wrong label families.
3. The lowerer accepts unresolved and recursively self-referential rule structures that should be compile-time hard errors.
4. Several “supported” languages in shipped rules are fictitious because the required label families do not exist for them.

These are exploitable blind spots, not hygiene issues.
