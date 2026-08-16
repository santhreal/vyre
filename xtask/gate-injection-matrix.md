# Gate injection matrix

Every ported gate has to go red on the same input the script it replaced
failed on. Each row is one edit, the gate that must go red, and the number it
must move to. Apply the edit, run the gate, confirm the number, revert the
edit, confirm the pin again. A gate that stays green under its injection is
not covering the assertion it inherited, whatever its pin says.

The `findings` column is the count with the injection applied, against the
pin recorded for that gate.

| Gate | Injection | Findings |
| --- | --- | --- |
| `backend-extension` | In `vyre-driver-wgpu/src/backend_dispatch.rs`, delete the `BackendPrecedence` inventory submission block. | 0 to 1 |
| `backend-extension` | In `vyre-driver/src/backend/registry/acquire.rs`, rename `registered_backends_by_precedence_slice` at its definition and its call. | 0 to 1 |
| `backend-extension` | Add `let id = "cuda";` to any file under `vyre-driver/src/backend/registry/`. | 0 to 1 |
| `readback-ring` | In `vyre-driver-wgpu/src/engine/record_and_readback/mod.rs`, rename `.arm_ticket(` to `.arm(` at its definition and its call sites. | 0 to 1 |
| `readback-ring` | In `vyre-driver-wgpu/src/lib.rs`, replace `ReadbackRingSet::new()` with `ReadbackRingSet::default()`. | 0 to 1 |
| `program-wire-fields` | Add `pub scratch_hint: u32,` to `Program` in `vyre-foundation/src/ir_inner/model/program/definition.rs`. | 0 to 1 |
| `program-wire-fields` | Delete every mention of `workgroup_size` from `vyre-foundation/src/serial/wire/encode/to_wire/mod.rs` and `vyre-foundation/src/serial/wire/decode/from_wire/mod.rs`. | 0 to 1 |
| `program-wire-fields` | Rename `pub struct Program` to `pub struct ProgramInner`. | gate errors, which is the intended outcome: the declaration is located, not named, so losing it is unmeasurable rather than clean |
| `frozen-contracts` | Add a method to `pub trait ExprVisitor` in `vyre-foundation/src/visit/expr_visitor/mod.rs`. | 1 to 2 |
| `frozen-contracts` | Delete `docs/frozen-traits/MutationClass.txt`. | 1 to 2 |
| `frozen-contracts` | Reindent the body of `pub enum AlgebraicLaw` by four spaces. | stays 1; indentation is not part of the contract |
| `file-size` | Append 200 blank lines to `vyre-foundation/src/optimizer/fact_cache/mod.rs` (measured 570, cap 599). | 75 to 76 |
| `file-size` | Append 60 lines to `vyre-libs/src/decode/inflate.rs` (measured 554, cap 582). | 75 to 76 |
| `file-size` | Add a row to the audit ceilings naming vyre-does-not-exist/src/lib.rs. | 75 to 76 |
| `gpu-loudness` | Add `#[cfg(not(feature = "gpu"))]` above a test in `vyre-driver-wgpu/tests/` with no loud abort within ten lines above or twenty below. | 2 to 3 |
| `gpu-loudness` | Add `if adapter.is_err() { return; }` to a test body. | 2 to 3 |
| `gpu-loudness` | Add `Backend::acquire_or_panic();` five lines below an existing finding site in `conform/vyre-conform/tests/cert_artifact/prove_failure_contracts.rs`. | 2 to 1, which is the allowance working rather than a failure |
| `unification` | Add a second `pub fn child_bodies` to any file under `vyre-foundation/src`. | 0 to 2, because a row over its ceiling reports every site |
| `unification` | Add `BufferAccess::infer(` to a file under `vyre-runtime/src/resident_work_queue`. | 0 to 1 |
| `unification` | Rename the directory `vyre-foundation/src/execution_plan` and update its `mod` declaration. | 0 to 1, reported as a path that does not exist rather than as a clean row |
| `evidence-paths` | In any artifact under `release/evidence`, change one cited path to a filename that does not exist but keeps a tree extension. | 18 to 19 |
| `evidence-paths` | Add `"manifest": "target/debug/build.rs"` to an artifact object, with the build directory gitignored. | 18 to 19, in the gitignored class rather than the missing class |
| `evidence-paths` | Change a cited path to `1.2.0`. | stays 18; a version string is not a citation |
| `docs-coupling` | Delete the `covers` key from the `docs/reference/wire-format.md` page row. | 0 to 1 |
| `docs-coupling` | Change one `covers` entry to a path the tree does not hold. | 0 to 1 |
| `docs-coupling` | In `docs/architecture/parsing.md`, change a cited source path inside a code span to one that does not exist. | 0 to 1 |
| `docs-coupling` | Edit a file `docs/reference/wire-format.md` covers without editing that page. | 0 to 2, one for the page and one for the missing changelog fragment |
| `docs-coupling` | Run it with `--base` naming a ref the checkout does not hold. | 0 to 1, reported as an unreachable base rather than as a gate that could not run |
| `example-capability` | In `examples/libs-template/Cargo.toml.liquid`, change `license` to `{{license}}`. | 0 to 1, naming the placeholder the gate cannot render |
| `example-capability` | In `examples/external_backend_extension/tests/backend_probe.rs`, change the expected dispatch output to `vec![9, 9, 9, 9]`. | 0 to 1, naming the failing test |
| `example-capability` | Delete the `[workspace]` table from `examples/external_backend_extension/Cargo.toml`. | 0 to 3, one for the isolation rule and one for each cargo invocation that then refuses to resolve |
| `example-capability` | Move `examples/external_backend_extension/Cargo.lock` aside so `--locked` cannot resolve. | 0 to 2, one per cargo invocation |
| `example-capability` | Track a file under a new `examples/<name>/` directory and give it no manifest. | 0 to 1 |
| `example-capability` | Track a 400-line Rust file and no lockfile under a new `examples/<name>/` directory carrying a manifest. | 0 to 2, one for the line cap and one for the missing lockfile |
| `invariant-paths` | In `vyre-spec/src/invariants.rs`, change a cited conformance test path to one that does not exist. | 0 to 1 |
| `doc-claims` | In `contracts/doc_claims_manifest.toml`, change one `phrase` to text its document does not contain. | 0 to 1 |
| `doc-claims` | Delete the `test` key from one claim. | 0 to 1, reported as an incomplete row rather than as a missing test |
| `hot-path-owned-dispatch` | In `vyre-driver/src/backend/compiled_pipeline.rs`, make `dispatch` the required method and give `dispatch_borrowed` a default that copies each row with `to_vec`. | 0 to 2, one finding for the requirement and one for the copy it forces |
| `hot-path-inventory` | In `vyre-libs/src/plumbing/registration/operation_catalog.rs`, serve `convergence_contract` by walking `inventory::iter` instead of probing the frozen index. | 0 to 1, quoting the statement that scans |
| `hot-path-nested-rows` | In `vyre-driver/src/backend/vyre_backend.rs`, delete the `dispatch_borrowed_into` declaration. | 0 to 1, naming the returning method that is then the only shape offered |
| `hot-path-nested-rows` | In `vyre-driver/src/backend/vyre_backend.rs`, replace the slot-preserving replacement in `dispatch_borrowed_into` with `*outputs = self.dispatch_borrowed(program, inputs, config)?;`. | 0 to 1, naming the slot the default replaces |
| `hot-path-unbounded-read` | Add `fs::read_to_string(` to a file under `vyre-driver/src` outside the reviewed cache modules. | 1 to 3 |
| `lint-unsafe-budget` | Add `#![allow(unsafe_code)]` to a crate root not in the reviewed budget. | 0 to 1 |
| `lint-unsafe-justification` | Add an `unsafe {` block with no SAFETY comment above it. | 2 to 3 |
| `lint-missing-docs-override` | Add `#![allow(missing_docs)]` to any crate root. | 0 to 1 |
| `proptest-coverage` | Delete two property-test files, taking the count from 182 to 180 against the floor of 181. | 0 to 1 |
| `repo-hygiene` | Add a second `BACKLOG.md` under any crate directory and track it. | 2 to 3 |
| `shader-source` | Add `out.push_str("@compute");` to a file under `vyre-driver-wgpu/src`. | 0 to 1 |
| `layering` | Add `vyre-lints.workspace = true` to `[dependencies]` in `vyre-spec/Cargo.toml`. | 0 to 29, one per member that reaches it, each naming the chain |
| `layering` | Add `wgpu.workspace = true` to `[dependencies]` in `vyre-spec/Cargo.toml`. | 0 to 48, every neutral member that reaches vyre-spec |
| `layering` | Delete the `[[crate]]` entry for `vyre-spec` from `docs/CRATE_OWNERSHIP.toml`. | gate errors: an unregistered member has an empty closure, so the roster is unreviewed rather than clean |
| `layering` | Change `layer` on the `vyre-spec` entry to a name NEUTRAL_LAYERS does not hold. | gate errors: a layer with no neutrality decision would be skipped |
| `layering` | Add `("invented", true)` to `NEUTRAL_LAYERS`. | gate errors: a decision no member uses is an allowance nothing needs |
| `layering` | Add `"not-a-crate"` to `BACKEND_APIS`. | gate errors: a boundary named after a crate the workspace never resolves cannot be crossed |
| `neutral-crates` | Add `naga.workspace = true` to `[dependencies]` in `vyre-primitives/Cargo.toml`. | 1 to 2, and the dotted form is what the shell rule could not see |
| `neutral-crates` | Add `vyre-ir = "0.1"` to `vyre-foundation/fuzz/Cargo.toml`, which the workspace excludes and cargo never loads with the graph. | 1 to 2 |
| `neutral-crates` | Add `"vyre-gone"` to `NEUTRAL_CRATES`. | gate errors: a neutrality rule over a manifest that does not exist reports success forever |

Three of these are negative controls rather than injections: the reindented
frozen enum, the version string cited as a path, and the loud abort added beside a
loudness finding. A gate that moves on those is over-reporting, which mutes it
just as surely as under-reporting.
