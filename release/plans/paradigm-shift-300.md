# Vyre Paradigm-Shift Release Backlog

This is the execution backlog for turning Vyre, Weir, and the active beta Vyre C frontend into a credible public launch. The target is not warning cleanup. The target is a GPU-first substrate that makes dataflow, parsing, fixed-point solving, and large-scale program evaluation materially faster and cleaner than the CPU-centered stack.

## Release invariants

1. CPU paths are allowed only as conformance references, parity harnesses, bootstrapping tools, or where the hardware boundary physically requires CPU orchestration.
2. GPU absence is never silently tolerated on the Santh fleet; probe failures are configuration errors.
3. Every optimization pass owns one duty in one file or module family; orchestration files orchestrate only.
4. Every performance claim has a reproducible benchmark, a baseline, a scale curve, and a failure threshold.
5. Every correctness claim has unit, adversarial, property, fuzz, and cross-substrate tests where the claim applies.
6. Every public API boundary is documented by tests and migration constraints, not by prose alone.
7. Vyre C frontend ships as beta active development unless clang parity is actually demonstrated on a named corpus.
8. Weir is a release pillar, not an afterthought; GPU dataflow is one of the clearest product demonstrations.
9. Megakernel is the default direction for scale; per-dispatch execution is the fallback path to eliminate.
10. The final public launch requires all publish gates, crate metadata, CI, docs, benchmarks, and repository state to be clean.

## 300 tasks

1. Define the release contract in one tracked file: product scope, non-goals, minimum evidence, blocked claims, and publish gates.
2. Split the release gate into `core`, `runtime`, `primitives`, `weir`, `vyrec-beta`, `docs`, and `publishing` sections so failures point to an owner.
3. Add a machine-readable release manifest listing crates, features, required tests, benchmarks, and publish order.
4. Add a gate that fails if any release crate lacks license, repository, description, keywords, categories, README, and crate-level docs.
5. Add a gate that fails if any public crate documents unsupported production claims for `vyrec` or C clang parity.
6. Add a gate that fails if docs claim a benchmark speedup without a benchmark artifact path.
7. Add a gate that fails if docs claim parity without corpus name, corpus hash, backend list, and result hash.
8. Add a gate that fails if any code path prints or logs silent CPU fallback for GPU-required execution.
9. Add a gate that fails if any GPU-required test skips because of “no GPU” without first logging adapter probe details.
10. Add a gate that fails if new optimizer passes are not registered through the pass metadata contract.
11. Make optimizer orchestration files contain scheduling and profile selection only.
12. Move canonicalization pass behavior into canonicalization-owned files only.
13. Move scalar algebra pass behavior into scalar-algebra-owned files only.
14. Move loop pass behavior into loop-owned files only.
15. Move memory pass behavior into memory-owned files only.
16. Move fusion and CSE pass behavior into fusion-CSE-owned files only.
17. Move sync and atomic normalization behavior into sync-owned files only.
18. Move cleanup behavior into cleanup-owned files only.
19. Move dataflow-specific optimization behavior into dataflow-owned files only.
20. Move megakernel-specific optimization behavior into megakernel-owned files only.
21. Add an optimizer profile matrix for `Release`, `Dataflow`, `Megakernel`, and `Conformance`.
22. Add tests proving `Release` excludes dataflow-only and megakernel-only passes unless explicitly selected.
23. Add tests proving `Dataflow` includes only boundary-safe dataflow passes.
24. Add tests proving `Megakernel` includes only passes compatible with persistent execution.
25. Add tests proving ABI-changing passes cannot run in ABI-preserving profiles.
26. Add tests proving backend-aware passes declare required substrate capabilities.
27. Add tests proving pass invalidation metadata is complete for all current passes.
28. Add tests proving pass requirement metadata prevents illegal reordering.
29. Add a pass-cost metadata gate that rejects unknown cost families in release profiles.
30. Add a pass-boundary metadata gate that rejects unknown boundary classes in release profiles.
31. Implement scalar expression reassociation with overflow and float-contract boundaries.
32. Implement integer strength reduction for multiply/divide by constants with exactness guards.
33. Implement bitwise identity folding with poison-free semantics.
34. Implement compare-chain simplification for canonical boolean forms.
35. Implement select/branch simplification for constant predicates.
36. Implement phi-like merge simplification where Vyre IR has equivalent join structure.
37. Implement redundant cast elimination with signedness and width guards.
38. Implement range-aware constant folding for bounded integer domains.
39. Implement algebraic simplification tests over random generated expression DAGs.
40. Implement adversarial tests for algebraic simplification around overflow, NaN, Inf, and signed zero.
41. Implement memory coalescing analysis for contiguous thread-local loads.
42. Implement redundant load elimination inside a single memory epoch.
43. Implement store-forwarding where no aliasing write intervenes.
44. Implement static shared-memory tiling candidate detection.
45. Implement memory-space canonicalization for global, shared, local, and constant buffers.
46. Implement buffer lifetime analysis before lowering.
47. Implement buffer reuse planning for non-overlapping lifetimes.
48. Implement copy-elision for host-to-device invariant buffers.
49. Implement copy-elision for device-to-host intermediate buffers.
50. Add allocation-count benchmarks for representative programs.
51. Implement loop-invariant code motion for pure expressions.
52. Implement loop-invariant buffer descriptor hoisting.
53. Implement loop unroll profitability using backend occupancy estimates.
54. Implement strip-mining profitability using workgroup geometry constraints.
55. Implement loop fusion for adjacent compatible traversal loops.
56. Implement loop fission for memory pressure reduction where fusion hurts occupancy.
57. Implement loop bounds normalization for backend emitters.
58. Implement induction-variable simplification.
59. Implement loop-carried dependency detection for unsafe fusion rejection.
60. Add loop optimizer property tests comparing pre/post execution outputs.
61. Implement global CSE over canonical expression hashes.
62. Implement effect-aware CSE that refuses to merge atomics, barriers, or volatile operations.
63. Implement structural hashing for IR subgraphs with stable deterministic IDs.
64. Implement dominance-aware DCE for unused computations.
65. Implement effect-aware DCE that preserves memory, atomics, barriers, and observable writes.
66. Implement region inlining profitability based on call count, size, and backend limits.
67. Implement region outlining for repeated subgraphs that exceed backend instruction cache pressure.
68. Add CSE benchmarks on generated high-duplication programs.
69. Add DCE adversarial tests with side-effect preservation.
70. Add region inline/outlining tests for boundary correctness.
71. Implement atomic normalization for all supported atomic operations.
72. Implement atomic combine optimization where associative and legal.
73. Implement barrier minimization by proving no cross-thread dependency.
74. Implement redundant barrier elimination with workgroup-scope proof.
75. Implement memory-order canonicalization with backend capability checks.
76. Implement sync-region analysis for persistent kernels.
77. Implement tests that intentionally fail if atomics are folded as pure expressions.
78. Implement tests that intentionally fail if barriers are removed across shared-memory hazards.
79. Add backend-specific sync legality snapshots.
80. Add random schedule perturbation tests for sync-sensitive primitives.
81. Define the Tier 2.5 primitive promotion checklist.
82. Add a primitive registry gate that rejects unclassified primitive modules.
83. Add a primitive boundary gate: primitives cannot import domain libraries.
84. Add a primitive feature gate matrix so downstream users can enable bitset, graph, reduce, scan, DFA, and fixpoint independently.
85. Add primitive docs that state substrate coverage per primitive.
86. Add primitive conformance corpus generation for bitset operations.
87. Add primitive conformance corpus generation for reductions.
88. Add primitive conformance corpus generation for scans.
89. Add primitive conformance corpus generation for CSR graph operations.
90. Add primitive conformance corpus generation for fixed-point operations.
91. Implement GPU-resident bitset OR with changed flag.
92. Implement GPU-resident bitset AND with changed flag.
93. Implement GPU-resident bitset difference with changed flag.
94. Implement GPU-resident bitset equality reduction.
95. Implement GPU-resident sparse bitset frontier expansion.
96. Implement GPU-resident dense bitset frontier expansion.
97. Implement hybrid sparse/dense frontier switching based on density.
98. Implement bitset population count reduction.
99. Implement bitset first-set and next-set primitives.
100. Add adversarial bitset tests for zero words, partial final words, huge bitsets, and unaligned fact counts.
101. Implement GPU-side convergence flag for Weir IFDS solving.
102. Replace per-iteration full frontier readback with a resident changed flag.
103. Download final frontier only after convergence unless tracing is explicitly requested.
104. Add a tracing mode that samples convergence without becoming the hot path.
105. Add resident IFDS resource lifetime tests for success, convergence, and failure cleanup.
106. Add IFDS seed validation tests for out-of-range nodes and facts.
107. Add IFDS graph validation tests for malformed CSR, invalid edge targets, and edge-kind overflow.
108. Add IFDS empty-graph behavior tests.
109. Add IFDS single-node self-loop convergence tests.
110. Add IFDS dense graph convergence tests.
111. Add IFDS sparse graph convergence tests.
112. Add IFDS high-fanout graph benchmarks.
113. Add IFDS deep-chain graph benchmarks.
114. Add IFDS random graph property tests against a CPU reference oracle.
115. Keep the CPU IFDS oracle isolated as parity-only test code.
116. Add Weir benchmark scale curves for nodes, facts, edges, density, and seed count.
117. Add Weir resident vs borrowed dispatch benchmarks.
118. Add Weir cold vs warm prepared-CSR benchmarks.
119. Add Weir max-iteration failure tests with actionable errors.
120. Add Weir dataflow examples that demonstrate multi-query reuse of a prepared graph.
121. Define the megakernel execution contract: resident queue, program handles, resource handles, completion records, and error records.
122. Add a megakernel scheduler module that owns scheduling only.
123. Add a megakernel queue module that owns ring-buffer mechanics only.
124. Add a megakernel resource table module that owns GPU-resident resource lifetimes only.
125. Add a megakernel program table module that owns compiled program handles only.
126. Add a megakernel telemetry module that owns counters and timestamps only.
127. Add a megakernel failure module that owns device-side error encoding only.
128. Add tests proving megakernel modules have no circular dependencies.
129. Add tests proving queue capacity is bounded and backpressure is explicit.
130. Add tests proving resource table exhaustion fails loudly.
131. Implement persistent GPU work queue push from CPU.
132. Implement persistent GPU work queue pop from GPU.
133. Implement device-side completion record writes.
134. Implement batched completion polling without blocking per program.
135. Implement queue wraparound tests.
136. Implement queue full tests.
137. Implement queue empty tests.
138. Implement multi-tenant queue fairness policy.
139. Implement priority aging to avoid starvation.
140. Add queue fairness benchmarks under mixed short and long programs.
141. Implement pipeline cache fingerprinting by adapter, backend, features, IR hash, and emitter version.
142. Add pipeline cache byte-budget enforcement.
143. Add pipeline cache LRU eviction tests.
144. Add pipeline cache collision tests.
145. Add pipeline cache warm-start benchmarks.
146. Add pipeline cache cold-start benchmarks.
147. Add pipeline cache invalidation on emitter version drift.
148. Add pipeline cache invalidation on adapter capability drift.
149. Add tests proving cache misses do not silently compile unsupported programs.
150. Add a benchmark target for 95% steady-state cache hit rate.
151. Implement zero-copy-ish host staging buffers where the backend permits it.
152. Implement pinned host memory strategy for CUDA backend where available.
153. Implement mapped buffer reuse for wgpu where legal.
154. Implement resource pooling by size class.
155. Implement resource pooling by usage flags.
156. Add pool high-watermark telemetry.
157. Add pool byte-budget enforcement.
158. Add pool fragmentation benchmarks.
159. Add pool adversarial tests for allocation/free storms.
160. Add tests proving resources are freed on every error path.
161. Implement backend capability discovery as a first-class typed object.
162. Reject feature use when the adapter capability object does not support it.
163. Add capability snapshots for RTX 5090 local hardware.
164. Add capability snapshots for axiomexec 4090.
165. Add capability snapshots for santhserver GPUs.
166. Add tests that fail if a GPU-required path silently falls back to CPU.
167. Add adapter-probe diagnostics with backend name, vendor, device, driver, and limits.
168. Add backend selection policy tests.
169. Add backend-specific resource limit tests.
170. Add backend-specific workgroup-size legality tests.
171. Implement WGSL emitter parity snapshots.
172. Implement SPIR-V emitter parity snapshots.
173. Implement CUDA/PTX emitter parity snapshots where the backend is present.
174. Add cross-substrate byte-identical conformance tests for Tier 2 ops.
175. Add cross-substrate byte-identical conformance tests for Tier 2.5 primitives.
176. Add cross-substrate byte-identical conformance tests for Weir kernels where semantics are deterministic.
177. Add floating-point exception policy tests.
178. Add integer overflow policy tests.
179. Add NaN canonicalization policy tests.
180. Add signed-zero policy tests.
181. Build a conformance corpus generator for small random programs.
182. Build a conformance corpus generator for branch-heavy programs.
183. Build a conformance corpus generator for memory-heavy programs.
184. Build a conformance corpus generator for atomic-heavy programs.
185. Build a conformance corpus generator for graph/dataflow programs.
186. Add shrinking for conformance counterexamples.
187. Add stable serialization for conformance witnesses.
188. Add deterministic corpus hash reporting.
189. Add result hash reporting per backend.
190. Add a signed or checksummed conformance artifact format.
191. Add fuzzing for IR parser/deserializer boundaries.
192. Add fuzzing for optimizer passes.
193. Add fuzzing for backend emitters.
194. Add fuzzing for wire-format decode.
195. Add fuzzing for resource lifetime APIs.
196. Add OOM injection where allocators can be controlled.
197. Add malformed input tests for every public parse boundary.
198. Add corrupted state tests for serialized program artifacts.
199. Add poisoned/failed dispatch tests for runtime APIs.
200. Add deterministic failure messages with fix guidance for every public error.
201. Audit public APIs for broken migration paths and add compatibility shims where justified.
202. Add API snapshot tests for public traits and structs.
203. Add semver policy docs tied to API snapshot tests.
204. Add deprecation policy that requires replacement path and tests.
205. Remove or finish non-test `todo!`, `unimplemented!`, placeholder, and fake-default implementations.
206. Replace empty-success returns in production code with real behavior or explicit errors.
207. Add a gate that fails on commented-out code in release crates.
208. Add a gate that fails on TODO/FIXME in release crates unless explicitly allowed in beta-only docs.
209. Add a gate that fails on dead public modules not reachable from crate docs or feature flags.
210. Add a gate that fails on duplicate functionality across modules without an explicit shared primitive.
211. Build an optimizer benchmark suite with micro, medium, and generated large programs.
212. Benchmark every pass independently for compile-time cost.
213. Benchmark every pass for runtime impact on representative GPU kernels.
214. Add pass profitability thresholds.
215. Add pass disable switches for investigation without making them production defaults.
216. Add profile-level benchmark budgets.
217. Add regression thresholds for compile-time overhead.
218. Add regression thresholds for GPU runtime.
219. Add regression thresholds for allocation count.
220. Add regression thresholds for host-device transfer bytes.
221. Add dataflow-specific optimization pass for frontier density specialization.
222. Add dataflow-specific optimization pass for fact-domain bit packing.
223. Add dataflow-specific optimization pass for edge-kind specialization.
224. Add dataflow-specific optimization pass for seed-set batching.
225. Add dataflow-specific optimization pass for invariant graph hoisting.
226. Add dataflow-specific optimization pass for convergence flag fusion.
227. Add dataflow-specific optimization pass for sparse/dense transition.
228. Add dataflow-specific optimization pass for transitive closure cutoff detection.
229. Add dataflow-specific optimization pass for SCC condensation when profitable.
230. Add dataflow-specific optimization pass tests against IFDS reference outputs.
231. Add megakernel-specific pass for dispatch batching.
232. Add megakernel-specific pass for resource residency planning.
233. Add megakernel-specific pass for queue packet packing.
234. Add megakernel-specific pass for completion record coalescing.
235. Add megakernel-specific pass for program-handle reuse.
236. Add megakernel-specific pass for static argument layout caching.
237. Add megakernel-specific pass for multi-program fusion when resource boundaries allow it.
238. Add megakernel-specific pass for small-program aggregation.
239. Add megakernel-specific pass for large-program occupancy shaping.
240. Add megakernel-specific optimization tests with scale-dependent benchmarks.
241. Define `vyrec` beta scope in code, docs, tests, and release notes.
242. Move clang-parity claims behind explicit future milestone docs.
243. Add C lexer corpus tests for small real-world headers.
244. Add C preprocessor macro expansion tests for object-like macros.
245. Add C preprocessor macro expansion tests for function-like macros.
246. Add C include resolution tests with bounded filesystem access.
247. Add C conditional compilation tests.
248. Add C token-pasting tests.
249. Add C stringification tests.
250. Add C diagnostic quality tests with actionable source spans.
251. Keep C parser CPU-only pieces labeled beta unless implemented on GPU or impossible to move.
252. Identify which C frontend phases can become GPU-resident token streams.
253. Prototype GPU token classification for C preprocessing.
254. Prototype GPU comment stripping and whitespace classification.
255. Prototype GPU macro candidate detection.
256. Prototype GPU include-guard detection.
257. Prototype GPU bracket/paren balance scan.
258. Prototype GPU line mapping.
259. Benchmark GPU token classification against clang preprocessing baselines.
260. Add a corpus report that states exactly which C features are supported, rejected, or beta.
261. Create a contributor architecture map: one file per subsystem, one duty per file, owner module, extension point.
262. Add module-level docs explaining how to add a new optimizer pass in one example.
263. Add module-level docs explaining how to add a new primitive in one example.
264. Add module-level docs explaining how to add a new backend emitter in one example.
265. Add module-level docs explaining how to add a new Weir dataflow kernel in one example.
266. Add crate docs that describe feature flags and standalone use.
267. Add examples for library mode.
268. Add examples for tool mode where tools exist.
269. Add examples for primitive-only feature mode.
270. Add examples for Weir prepared graph reuse.
271. Add Tier A config support to tool-facing crates.
272. Add Tier B TOML rule/data loading where structured community knowledge exists.
273. Remove hardcoded rule lists that should be Tier B data.
274. Add config precedence tests: compiled defaults, TOML, CLI override.
275. Add external TOML merge tests.
276. Add TOML validation tests with actionable errors.
277. Add malformed TOML adversarial tests.
278. Add examples of community data extension without Rust changes.
279. Add docs separating operational config from knowledge data.
280. Add release gate that rejects hardcoded extendable pattern lists.
281. Add CI jobs for formatting, linting, tests, GPU tests, fuzz smoke, benchmarks smoke, docs, and publish dry-run.
282. Add GPU CI path that fails loudly if adapter probe fails on expected GPU hardware.
283. Add publish dry-run for every release crate.
284. Add README badges only after CI actually exists and is green.
285. Add changelog entries tied to actual code changes and benchmark artifacts.
286. Add license and attribution review for dependencies.
287. Add dependency audit for duplicate crates, abandoned crates, and oversized dependency trees.
288. Add minimal feature build checks.
289. Add all-features build checks.
290. Add no-default-features build checks where crates advertise feature modularity.
291. Run deep personal source review for each release crate.
292. Run full GPU test suite locally on the RTX 5090.
293. Run full GPU test suite on axiomexec 4090.
294. Run full GPU test suite on santhserver GPUs.
295. Run benchmark suite and record hardware, driver, corpus hashes, and results.
296. Fix every open release-blocking finding in the tracked audit file.
297. Confirm `vyrec` is labeled beta/not-production everywhere unless parity evidence exists.
298. Confirm Weir dataflow demo shows scale-dependent megakernel/resident-GPU advantage.
299. Run final `cargo publish --dry-run` for every publishable crate in release order.
300. Execute the public launch: `cargo publish` for approved crates, change the repository visibility to public, then `git push` the release branch and tags.
