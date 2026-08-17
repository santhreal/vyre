//! When the CUDA driver is allowed to host-split a grid-sync program, and when it
//! must refuse instead.
//!
//! The bug these lock out: `<CudaBackend as VyreBackend>::allows_host_grid_sync_split`
//! returns `false`, and its contract says CUDA surfaces missing native
//! grid-barrier lowering as an unsupported feature rather than "silently becoming
//! a slower multi-launch path". The driver's own `dispatch_borrowed` and
//! `dispatch_borrowed_async` did that silent reroute anyway: on a device without
//! cooperative launch they split the program behind the caller's back, which is
//! precisely the outcome the advertised capability promises cannot happen.
//! `vyre-primitives`' persistent fixpoint reads that capability to decide whether
//! it has an escape hatch and documents a silent degrade there as a CORRECTNESS
//! failure, not a performance one, so the promise has to hold in the driver and
//! not only in the registry wrapper.
//!
//! The distinction that has to survive: over-residency splitting is legitimate
//! and load-bearing. The barrier is native, the grid simply does not fit, and the
//! recursive multi-block prefix scan's pass-B grid reaches that path in shipping
//! code. Refusing THAT would break working programs. So the rule is one reason to
//! split (the grid does not fit) and one reason to refuse (there is no native
//! barrier at all), and these tests pin both directions.
//!
//! Six top-level `GridSync` barriers now ship in
//! `vyre-libs/src/parsing/c/parse/structure_statement.rs`, which does not gate on
//! the residency bound the way `exatok` does, so a large enough translation unit
//! reaches the over-residency path from a shipping frontend. That is why the
//! over-residency route is tested behaviorally here and not assumed.

mod harness;

use harness::{
    bytes_u32, cross_block_grid_sync_expected, cross_block_grid_sync_inputs,
    cross_block_grid_sync_program, CROSS_BLOCK_GRID_SYNC_WORKGROUP,
};
use vyre_driver::resolve_launch_workgroup;
use vyre_driver::validation::LaunchGeometryLimits;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::occupancy::cooperative_thread_residency_block_limit;
use vyre_driver_cuda::{cuda_factory, CudaBackend};
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// Grid barriers in [`five_barrier_chain_program`], matching the count shipped in
/// `vyre-libs/src/parsing/c/parse/structure_statement.rs`.
const CHAIN_BARRIERS: u32 = 5;

/// Per-block accumulate iterations in [`five_barrier_chain_program`], the same
/// asymmetry the single-barrier fixture uses: block `b` runs `b * DELAY` dependent
/// read-modify-writes before each barrier, so the block every other block waits on
/// is the slowest one.
const CHAIN_DELAY_PER_BLOCK: u32 = 2;

/// Program with FIVE top-level grid barriers whose answer is wrong unless every
/// one of them blocks.
///
/// Shape, with `b = gid / workgroup`, repeated for each of the five barriers:
///   accumulate: `ring[gid] += 1`, `b * DELAY` times
///   barrier:    `MemoryOrdering::GridSync`
///   republish:  `ring[gid] = ring[n - 1]`  (every lane reads the LAST lane's slot)
/// and a final `out[gid] = ring[n - 1] + input[gid]`.
///
/// The detection mechanism is the single-barrier fixture's, applied at every cut:
/// the slot every block reads belongs to the last lane, whose block does the MOST
/// pre-barrier work, so a barrier that fails to block lets a fast block read that
/// slot mid-accumulation and the output lands BELOW the expected value. Because
/// each stage feeds the next, a failure at any one of the five barriers shows up
/// in the final buffer.
///
/// Built with `Program::wrapped`, so the barriers sit at the top level of the INNER
/// sequence rather than the Program body. That is deliberate: it is the exact
/// structure the C statement-structure kernel uses, and the structure the split
/// lowering has to peel before it can cut.
fn five_barrier_chain_program(n: u32) -> Program {
    assert!(
        n >= 2 * CROSS_BLOCK_GRID_SYNC_WORKGROUP && n % CROSS_BLOCK_GRID_SYNC_WORKGROUP == 0,
        "Fix: the chain fixture needs a whole number of blocks and at least two; got {n} lanes."
    );
    let iterations = Expr::mul(
        Expr::div(Expr::gid_x(), Expr::u32(CROSS_BLOCK_GRID_SYNC_WORKGROUP)),
        Expr::u32(CHAIN_DELAY_PER_BLOCK),
    );
    let mut body: Vec<Node> = Vec::new();
    for stage in 0..CHAIN_BARRIERS {
        // Each loop gets its own name: sibling `let`/loop bindings sharing a name in
        // one region is a validation error, not a silent shadow.
        body.push(Node::loop_for(
            &format!("chain_delay_{stage}"),
            Expr::u32(0),
            iterations.clone(),
            vec![Node::store(
                "ring",
                Expr::gid_x(),
                Expr::add(Expr::load("ring", Expr::gid_x()), Expr::u32(1)),
            )],
        ));
        body.push(Node::barrier_with_ordering(MemoryOrdering::GridSync));
        // Republish the last lane's slot into every lane. Lane `n - 1` writes its
        // own slot the value it already holds, so this is not a cross-lane race
        // even while other blocks read the same slot.
        body.push(Node::store(
            "ring",
            Expr::gid_x(),
            Expr::load("ring", Expr::u32(n - 1)),
        ));
    }
    body.push(Node::store(
        "out",
        Expr::gid_x(),
        Expr::add(
            Expr::load("ring", Expr::u32(n - 1)),
            Expr::load("input", Expr::gid_x()),
        ),
    ));
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(n),
            BufferDecl::read_write("ring", 1, DataType::U32).with_count(n),
            BufferDecl::output("out", 2, DataType::U32).with_count(n),
        ],
        [CROSS_BLOCK_GRID_SYNC_WORKGROUP, 1, 1],
        body,
    )
}

/// `input[gid] == gid` and `ring` seeded identically, re-uploaded per launch
/// because `ring` is read_write and an inherited value would pass without any
/// barrier at all.
fn five_barrier_chain_inputs(n: u32) -> Vec<Vec<u8>> {
    let lanes: Vec<u8> = (0..n).flat_map(|lane| lane.to_le_bytes()).collect();
    vec![lanes.clone(), lanes]
}

/// The only correct `out` for [`five_barrier_chain_program`].
///
/// Each stage lifts the last lane's slot by `(blocks - 1) * DELAY`: stage 0 takes
/// it from `n - 1` to `(n - 1) + (blocks - 1) * DELAY`, and each later stage
/// republishes that value to every lane and lifts it again by the same amount. So
/// after five stages the shared value is `(n - 1) + 5 * (blocks - 1) * DELAY`, and
/// each lane stores that plus its own `input[gid] == gid`.
fn five_barrier_chain_expected(n: u32) -> Vec<u32> {
    let blocks = n / CROSS_BLOCK_GRID_SYNC_WORKGROUP;
    let shared = (n - 1) + CHAIN_BARRIERS * (blocks - 1) * CHAIN_DELAY_PER_BLOCK;
    (0..n).map(|gid| shared + gid).collect()
}

fn backend() -> CudaBackend {
    CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.")
}


/// Launch limits for this device, so a test can resolve the workgroup the driver
/// will actually plan.
fn launch_limits(backend: &CudaBackend) -> LaunchGeometryLimits {
    LaunchGeometryLimits {
        backend: "CUDA",
        max_threads_per_block: backend.max_threads_per_block(),
        max_block_dim: backend.max_block_dim(),
        max_grid_dim: backend.max_grid_dim(),
        max_threads_per_sm: backend.max_threads_per_sm(),
    }
}

/// The cooperative lane ceiling FOR THIS PROGRAM, derived from the effective
/// workgroup rather than the declared one.
///
/// This distinction is not pedantry, it is the whole bound. A program declaring
/// `[256, 1, 1]` does NOT necessarily launch 256 wide: `Mode::production_default()`
/// is `NaturalGradient`, so with `VYRE_AUTOTUNER` unset the tuner may pick a
/// cold-start workgroup, and for these fixtures it picks `[1024, 1, 1]`. The
/// residency bound is computed on that effective width, and 1024 is the worst case
/// on this device: `max_threads_per_sm / workgroup` is integer division, 1536/1024
/// is 1, so one block per SM fits and 512 of every SM's 1536 thread slots go
/// unused. The ceiling is therefore 170 blocks of 1024 (174,080 lanes) for a
/// tunable program by default, and 1020 blocks of 256 (261,120 lanes) when the
/// declared width survives.
///
/// Per PROGRAM, not per device, and that is the second half of the lesson.
/// `is_natural_gradient_launch_tunable` rejects a program that sets
/// `non_composable_with_self`, uses `LocalId`/`WorkgroupId`, or wants workgroup
/// scratch, so two programs on one device can have different effective widths and
/// therefore different ceilings. Resolving the width once and reusing it across
/// fixtures would reintroduce the same class of wrong answer in the other
/// direction: `parsing::c`'s statement-structure kernel is exempt this way and
/// keeps its declared 256, so its ceiling is 261,120 lanes while this fixture's is
/// 174,080.
///
/// A test that computed the bound from the DECLARED workgroup would assert the
/// wrong boundary, silently pass whenever both configurations agree on the verdict,
/// and fail confusingly when they do not. That happened: the boundary test caught
/// it by reporting `fits=false` at 681 blocks of 256 instead of 1021.
fn cooperative_lane_ceiling(backend: &CudaBackend, program: &Program) -> Option<u32> {
    let declared_lanes = program
        .buffers()
        .iter()
        .map(vyre_foundation::ir::BufferDecl::count)
        .max()?;
    let effective = resolve_launch_workgroup(
        program,
        &DispatchConfig::default(),
        launch_limits(backend),
        declared_lanes,
    );
    let resident_blocks = cooperative_thread_residency_block_limit(&backend.caps, effective[0]);
    if resident_blocks == 0 {
        return None;
    }
    u32::try_from(resident_blocks)
        .ok()?
        .checked_mul(effective[0])
}

/// Lanes for a grid that provably does NOT fit a cooperative launch of the program
/// `build` produces, derived from the driver's own bound rather than a number typed
/// into a test.
///
/// The ceiling depends on the effective workgroup, which depends on the program, and
/// the program depends on its lane count, so this iterates: pick a candidate lane
/// count, rebuild the program at that width, and re-derive the ceiling until the
/// candidate is genuinely above the ceiling of the program actually being launched.
fn over_residency_lanes_for(backend: &CudaBackend, build: impl Fn(u32) -> Program) -> Option<u32> {
    let mut lanes = FITTING_LANES;
    for _ in 0..8 {
        let ceiling = cooperative_lane_ceiling(backend, &build(lanes))?;
        let candidate = ceiling
            .checked_div(CROSS_BLOCK_GRID_SYNC_WORKGROUP)?
            .checked_add(4)?
            .checked_mul(CROSS_BLOCK_GRID_SYNC_WORKGROUP)?;
        if candidate > ceiling && cooperative_lane_ceiling(backend, &build(candidate))? < candidate
        {
            return Some(candidate);
        }
        lanes = candidate;
    }
    None
}

/// Lanes above the ceiling for the single-barrier cross-block fixture.
fn over_residency_lanes(backend: &CudaBackend) -> Option<u32> {
    over_residency_lanes_for(backend, cross_block_grid_sync_program)
}

/// Lanes for a grid that comfortably fits: four blocks.
const FITTING_LANES: u32 = 4 * CROSS_BLOCK_GRID_SYNC_WORKGROUP;

/// A grid-sync program too wide for a cooperative launch must still run, and run
/// CORRECTLY.
///
/// This is the load-bearing half of the policy. Making missing-native-barrier a
/// refusal is only safe if the over-residency route is untouched: the recursive
/// multi-block prefix scan's pass-B grid exceeds cooperative residency in
/// shipping code, and `parsing::c`'s statement-structure kernel can reach the
/// same path with a large translation unit. If this regresses to a refusal, those
/// programs stop working entirely rather than getting slower.
///
/// The fixture's answer depends on the barrier actually blocking, so a split that
/// dropped the barrier semantics would show up as wrong values here, not as a
/// pass.
#[test]
fn grid_sync_program_wider_than_cooperative_residency_still_dispatches_correctly() {
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let Some(lanes) = over_residency_lanes(&backend) else {
        panic!(
            "Fix: hardware reports grid-sync support, so the cooperative residency bound must be \
             positive and an over-residency lane count must be derivable."
        );
    };
    let ceiling = cooperative_lane_ceiling(&backend, &cross_block_grid_sync_program(lanes))
        .expect("Fix: the cooperative lane ceiling must be derivable on grid-sync hardware.");
    assert!(
        lanes > ceiling,
        "Fix: this test is only meaningful when the launch exceeds the cooperative lane ceiling; \
         got {lanes} lanes against a ceiling of {ceiling}. Compare LANES, not declared-width \
         blocks: the effective workgroup is what the bound is computed on."
    );

    let program = cross_block_grid_sync_program(lanes);
    let inputs = cross_block_grid_sync_inputs(lanes);
    let outputs = backend
        .dispatch(&program, &inputs, &DispatchConfig::default())
        .expect(
            "Fix: a grid-sync program whose grid exceeds cooperative residency must route to the \
             host-orchestrated split and succeed. A refusal here means the missing-native-barrier \
             refusal was applied to the over-residency case too, which breaks the multi-block \
             prefix scan and the C statement-structure kernel.",
        );
    assert_eq!(
        bytes_u32(outputs.last().expect("the fixture declares an output")),
        cross_block_grid_sync_expected(lanes),
        "Fix: the host-split route must preserve whole-grid barrier semantics; wrong values here \
         mean the split segments do not actually order the pre-barrier writes."
    );
}

/// The same program at a FITTING grid must take the native cooperative route and
/// produce the identical answer.
///
/// Two routes, one answer. If they ever disagree, one of them is wrong and the
/// residency bound silently decides which answer a caller gets, which is the
/// worst possible shape for a correctness bug.
#[test]
fn native_and_over_residency_routes_agree_on_the_answer() {
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let fitting_program = cross_block_grid_sync_program(FITTING_LANES);
    let fitting_inputs = cross_block_grid_sync_inputs(FITTING_LANES);
    let native = backend
        .dispatch(
            &fitting_program,
            &fitting_inputs,
            &DispatchConfig::default(),
        )
        .expect("Fix: a fitting grid-sync program must dispatch natively.");
    assert_eq!(
        bytes_u32(native.last().expect("the fixture declares an output")),
        cross_block_grid_sync_expected(FITTING_LANES),
        "Fix: the native cooperative route must produce the grid-synchronized answer."
    );

    let Some(lanes) = over_residency_lanes(&backend) else {
        panic!("Fix: cooperative residency bound must be positive on grid-sync hardware.");
    };
    let split_program = cross_block_grid_sync_program(lanes);
    let split_inputs = cross_block_grid_sync_inputs(lanes);
    let split = backend
        .dispatch(&split_program, &split_inputs, &DispatchConfig::default())
        .expect("Fix: an over-residency grid-sync program must dispatch via the split route.");

    // Both routes obey the same closed-form answer, so agreement is asserted
    // against that form at each width rather than against each other's bytes.
    assert_eq!(
        bytes_u32(split.last().expect("the fixture declares an output")),
        cross_block_grid_sync_expected(lanes),
        "Fix: the two routes must not produce different answers for the same program shape; the \
         residency bound must decide only HOW the barrier is realized, never WHAT it computes."
    );
}

/// The advertised capability must match what the fits-check actually reports.
///
/// `cooperative_grid_sync_fits` is the honest per-dispatch signal, and callers use
/// it to choose a route before paying for an allocate and upload. If it answered
/// `true` for an over-residency grid, an orchestrator would pick the native route
/// and eat `CooperativeResidencyExceeded` after the upload. If it answered `false`
/// for a fitting grid, every cooperative launch would be abandoned for the slow
/// path. Both booleans are asserted as real observed values.
#[test]
fn cooperative_fits_check_reports_true_for_a_fitting_grid_and_false_above_residency() {
    let backend = cuda_factory()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    let direct = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    if !direct.hardware_supports_grid_sync() {
        return;
    }

    let fitting_program = cross_block_grid_sync_program(FITTING_LANES);
    let fitting_inputs = cross_block_grid_sync_inputs(FITTING_LANES);
    let fitting_borrowed: Vec<&[u8]> = fitting_inputs.iter().map(Vec::as_slice).collect();
    assert!(
        backend
            .cooperative_grid_sync_fits(
                &fitting_program,
                &fitting_borrowed,
                &DispatchConfig::default()
            )
            .expect("Fix: the fits check must compute launch geometry for a valid program."),
        "Fix: a four-block grid-sync program fits cooperative residency on this device and the \
         fits check must say so, or the native route is never chosen."
    );

    let Some(lanes) = over_residency_lanes(&direct) else {
        panic!("Fix: cooperative residency bound must be positive on grid-sync hardware.");
    };
    let wide_program = cross_block_grid_sync_program(lanes);
    let wide_inputs = cross_block_grid_sync_inputs(lanes);
    let wide_borrowed: Vec<&[u8]> = wide_inputs.iter().map(Vec::as_slice).collect();
    assert!(
        !backend
            .cooperative_grid_sync_fits(&wide_program, &wide_borrowed, &DispatchConfig::default())
            .expect("Fix: the fits check must compute launch geometry for a valid program."),
        "Fix: a grid above the cooperative residency bound does NOT fit and the fits check must \
         report false, or an orchestrator picks the native route and fails after uploading."
    );
}

/// The preflight must predict the route the driver actually takes, at the exact
/// boundary lane count.
///
/// The defect this locks out: `cooperative_grid_sync_fits` carried its OWN copy of
/// the residency arithmetic, separate from the copy the dispatch route decision
/// used, and it had no production caller at all. A predicate that nobody reads
/// cannot be wrong, so nobody noticed the copies already disagreed on a workgroup
/// whose thread count does not fit `u32`. Now the route decision calls the
/// predicate, so this test checks the property that sharing is supposed to buy:
/// preflight `true` means the native cooperative route runs, preflight `false`
/// means the split route runs, at the boundary and one declared-width block either
/// side of it.
///
/// If these ever diverge, an orchestrator routes on the preflight and then fails
/// anyway, which is worse than having no preflight, and the failure appears one
/// upload too late to be cheap.
///
/// The boundary comes from [`cooperative_lane_ceiling`], which resolves the
/// EFFECTIVE workgroup. Writing this test against the declared workgroup is what
/// exposed the autotuner's override in the first place: it asserted a boundary at
/// 1021 blocks of 256 and the real one was 681, because the tuner launches this
/// program 1024 wide.
#[test]
fn preflight_verdict_matches_the_route_taken_at_the_exact_residency_boundary() {
    let registry = cuda_factory()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let Some(ceiling_lanes) =
        cooperative_lane_ceiling(&backend, &cross_block_grid_sync_program(FITTING_LANES))
    else {
        panic!("Fix: cooperative residency bound must be positive on grid-sync hardware.");
    };
    // The fixture needs a whole number of declared-width blocks, so step by one
    // such block on either side of the ceiling.
    assert_eq!(
        ceiling_lanes % CROSS_BLOCK_GRID_SYNC_WORKGROUP,
        0,
        "Fix: the cooperative lane ceiling ({ceiling_lanes}) must be a whole number of \
         declared-width blocks for this fixture to straddle it exactly."
    );

    for (lanes, must_fit) in [
        (ceiling_lanes, true),
        (ceiling_lanes + CROSS_BLOCK_GRID_SYNC_WORKGROUP, false),
    ] {
        let program = cross_block_grid_sync_program(lanes);
        let inputs = cross_block_grid_sync_inputs(lanes);
        let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();

        let fits = registry
            .cooperative_grid_sync_fits(&program, &borrowed, &DispatchConfig::default())
            .expect("Fix: the preflight must compute launch geometry for a valid program.");
        assert_eq!(
            fits, must_fit,
            "Fix: at {lanes} lanes against a cooperative ceiling of {ceiling_lanes} lanes, the \
             preflight must report fits={must_fit}. A boundary off by one block means the \
             preflight and the route decision disagree for exactly one grid width, which is the \
             hardest possible version of this bug to find."
        );

        let outputs = backend
            .dispatch(&program, &inputs, &DispatchConfig::default())
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: {lanes} lanes (preflight fits={fits}) must dispatch successfully by \
                     whichever route the preflight predicts: {error}"
                )
            });
        assert_eq!(
            bytes_u32(outputs.last().expect("the fixture declares an output")),
            cross_block_grid_sync_expected(lanes),
            "Fix: at {lanes} lanes the chosen route must still produce the grid-synchronized \
             answer; a wrong value means the route the preflight predicted does not honor the \
             barrier."
        );
    }
}

/// The cooperative ceiling must be computed on the EFFECTIVE workgroup, and the
/// autotuner's default choice must be visible rather than assumed.
///
/// This is the finding that corrected a published crossing point. A program
/// declaring `[256, 1, 1]` is launched 1024 wide by default, because
/// `Mode::production_default()` is `NaturalGradient` and the tuner picks a
/// cold-start workgroup when `VYRE_AUTOTUNER` is unset. Since
/// `max_threads_per_sm / workgroup` truncates (1536/1024 = 1), that choice cuts the
/// cooperative thread ceiling from 261,120 lanes to 174,080, a third of the device's
/// cooperative capacity, and it moves the route-transition point for every consumer
/// that reasons about it from the declared width.
///
/// Asserted as exact values so a change in the tuner's default, or in how the
/// residency bound is derived, shows up here with both numbers rather than as a
/// consumer's documentation quietly going stale.
#[test]
fn cooperative_ceiling_follows_the_effective_workgroup_not_the_declared_one() {
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let program = cross_block_grid_sync_program(FITTING_LANES);
    assert_eq!(
        program.workgroup_size(),
        [CROSS_BLOCK_GRID_SYNC_WORKGROUP, 1, 1],
        "Fix: the fixture must declare a 256-wide workgroup for this comparison to mean anything."
    );

    let effective = resolve_launch_workgroup(
        &program,
        &DispatchConfig::default(),
        launch_limits(&backend),
        FITTING_LANES,
    );
    let declared_bound =
        cooperative_thread_residency_block_limit(&backend.caps, CROSS_BLOCK_GRID_SYNC_WORKGROUP);
    let effective_bound = cooperative_thread_residency_block_limit(&backend.caps, effective[0]);

    // Both bounds are real values from the device caps: 170 SMs, 1536 threads/SM.
    assert_eq!(
        declared_bound * u64::from(CROSS_BLOCK_GRID_SYNC_WORKGROUP),
        u64::from(backend.caps.max_threads_per_sm_u32())
            * u64::from(backend.caps.multi_processor_count_u32()),
        "Fix: at 256 wide the workgroup divides 1536 exactly, so the block bound must account for \
         every thread slot on the device."
    );
    assert!(
        effective_bound * u64::from(effective[0])
            <= declared_bound * u64::from(CROSS_BLOCK_GRID_SYNC_WORKGROUP),
        "Fix: the effective workgroup can only equal or waste thread slots relative to a width \
         that divides max_threads_per_sm evenly; a larger product means the bound is miscomputed."
    );

    let ceiling = cooperative_lane_ceiling(&backend, &program)
        .expect("Fix: the cooperative lane ceiling must be derivable on grid-sync hardware.");
    assert_eq!(
        ceiling,
        u32::try_from(effective_bound).expect("bound fits u32") * effective[0],
        "Fix: the lane ceiling must be the effective block bound times the effective workgroup."
    );
    // The ceiling must obey the residency law for WHICHEVER width is in effect, with
    // exact numbers on both branches. Deliberately not pinned to the tuner's current
    // choice: LaunchWidthTuner is changing cold-start selection to stop wasting thread
    // slots, and a test that hardcoded today's 1024 would report that correct fix as a
    // regression. What must never change is the arithmetic and the device total.
    let device_threads = u64::from(backend.caps.max_threads_per_sm_u32())
        * u64::from(backend.caps.multi_processor_count_u32());
    assert_eq!(
        device_threads, 261_120,
        "Fix: 170 SMs at 1536 threads each is 261,120 thread slots; a different total means the \
         probed device caps changed and every ceiling below moves with them."
    );
    if backend.caps.max_threads_per_sm_u32() % effective[0] == 0 {
        // A width that divides the per-SM thread budget evenly wastes nothing.
        assert_eq!(
            u64::from(ceiling),
            device_threads,
            "Fix: an evenly dividing width ({}) must reach every thread slot: {device_threads} \
             lanes. A lower ceiling means the block bound is miscomputed.",
            effective[0]
        );
    } else {
        // Truncation waste, the case that cost a third of the device at width 1024.
        let blocks_per_sm = backend.caps.max_threads_per_sm_u32() / effective[0];
        assert_eq!(
            u64::from(ceiling),
            u64::from(blocks_per_sm)
                * u64::from(effective[0])
                * u64::from(backend.caps.multi_processor_count_u32()),
            "Fix: with width {} the per-SM budget truncates to {blocks_per_sm} block(s), so the \
             ceiling is that times the width times the SM count.",
            effective[0]
        );
        assert!(
            u64::from(ceiling) < device_threads,
            "Fix: a width that does not divide the per-SM thread budget MUST leave slots idle; \
             equality here means the truncation branch was entered wrongly."
        );
    }
}

/// The preflight must answer `false`, not error, for a program with no grid-sync
/// barrier.
///
/// The trait documents `Ok(false)` for that case specifically: there is nothing to
/// launch cooperatively, and an error would force every orchestrator to special
/// case a question it is allowed to ask about any program. Now that the route
/// decision shares this function, an error here would also turn a plain dispatch
/// into a failure.
#[test]
fn preflight_reports_false_without_erroring_for_a_program_with_no_grid_sync_barrier() {
    let registry = cuda_factory()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    // A barrier-free program, defined here so the preflight is asked about a
    // program that provably carries no GridSync node.
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(8),
            BufferDecl::output("out", 1, DataType::U32).with_count(8),
        ],
        [128, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(1)),
        )],
    );
    let inputs: Vec<Vec<u8>> = vec![(1..=8_u32).flat_map(u32::to_le_bytes).collect()];
    let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    assert!(
        !registry
            .cooperative_grid_sync_fits(&program, &borrowed, &DispatchConfig::default())
            .expect(
                "Fix: a program with no grid-sync barrier must answer the preflight with \
                 Ok(false), not an error."
            ),
        "Fix: with no grid-sync barrier there is nothing to launch cooperatively, so the preflight \
         reports false."
    );
}

/// `allows_host_grid_sync_split` stays false, and that is not contradicted by the
/// over-residency route.
///
/// The two are about different things and the distinction is what makes the policy
/// coherent: this flag governs whether the shared REGISTRY WRAPPER may emulate a
/// grid barrier the backend does not have (`should_split_grid_sync` also requires
/// `!supports_grid_sync()`), while the over-residency split happens with a fully
/// native barrier available and a grid that does not fit. This test asserts both
/// facts in one place so a future reader does not "fix" the flag to true after
/// noticing the driver splits.
#[test]
fn registry_split_permission_stays_false_while_over_residency_splitting_works() {
    let backend = cuda_factory()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    assert!(
        !backend.allows_host_grid_sync_split(),
        "Fix: CUDA must not permit the registry wrapper to emulate a missing grid barrier by \
         splitting; that path must be a loud refusal."
    );
    let direct = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    if !direct.hardware_supports_grid_sync() {
        return;
    }
    assert!(
        backend.supports_grid_sync(),
        "Fix: on this device the barrier IS native, which is exactly why the registry wrapper's \
         emulation permission being false costs nothing and the over-residency split is a \
         different mechanism."
    );
}

/// FIVE top-level grid barriers must split correctly above cooperative residency,
/// not just one.
///
/// This exists because a downstream consumer was about to document "one token past
/// the ceiling is a correct kernel split" for a five-barrier kernel, citing
/// evidence gathered from a ONE-barrier fixture. One cut and five cuts are not the
/// same claim: the split has to peel the wrapper region, cut at every barrier, and
/// carry the read_write buffer state across six separate launches in order. A
/// lowering that handled the first cut and dropped the rest, or that reordered the
/// segments, would pass the single-barrier test and silently corrupt this one.
///
/// `parsing::c`'s statement-structure kernel launches at workgroup 256 with two
/// lanes per token, so it crosses the 1020-block cooperative ceiling at 130,560
/// tokens and takes this route in shipping code. That is why this is verified on
/// hardware rather than reasoned about from the lowering source.
#[test]
fn five_barrier_program_splits_correctly_above_cooperative_residency() {
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let Some(lanes) = over_residency_lanes_for(&backend, five_barrier_chain_program) else {
        panic!("Fix: cooperative residency bound must be positive on grid-sync hardware.");
    };
    let ceiling = cooperative_lane_ceiling(&backend, &five_barrier_chain_program(lanes))
        .expect("Fix: the cooperative lane ceiling must be derivable on grid-sync hardware.");
    assert!(
        lanes > ceiling,
        "Fix: this test is only meaningful above the cooperative lane ceiling; got {lanes} lanes \
         against a ceiling of {ceiling}."
    );

    let program = five_barrier_chain_program(lanes);
    let inputs = five_barrier_chain_inputs(lanes);
    let outputs = backend
        .dispatch(&program, &inputs, &DispatchConfig::default())
        .expect(
            "Fix: a five-barrier grid-sync program above cooperative residency must route to the \
             host-orchestrated split and succeed. A failure here means multi-cut splitting is \
             broken and the C statement-structure kernel has a real input ceiling after all.",
        );
    let actual = bytes_u32(outputs.last().expect("the fixture declares an output"));
    let expected = five_barrier_chain_expected(lanes);
    assert_eq!(
        actual, expected,
        "Fix: the split route must honor ALL FIVE barriers. Values below expectation mean a cut \
         was dropped or reordered and a stage read the last lane's slot mid-accumulation."
    );
}

/// The same five-barrier program at a FITTING grid must give the identical answer
/// through the native cooperative route.
///
/// Five barriers in one cooperative launch is also the case where the per-launch
/// `_vyre_grid_barrier` reset has to be right: the counter is shared by all five
/// barrier sites, whose release targets are `(index + 1) * gridSize`, so a stale
/// counter releases the later barriers first. This is the multi-barrier companion
/// to the arrival audit's ceiling of `barriers * blocks`.
#[test]
fn five_barrier_program_is_correct_on_the_native_cooperative_route_too() {
    let backend = backend();
    if !backend.hardware_supports_grid_sync() {
        return;
    }
    let program = five_barrier_chain_program(FITTING_LANES);
    let expected = five_barrier_chain_expected(FITTING_LANES);

    // Twice, on one loaded module: launch 2 is where a stale counter shows up, and
    // with five barrier sites a missed reset releases four of them immediately.
    for launch in 1..=2_u32 {
        let inputs = five_barrier_chain_inputs(FITTING_LANES);
        let outputs = backend
            .dispatch(&program, &inputs, &DispatchConfig::default())
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: native launch {launch} of the five-barrier program must succeed: {error}"
                )
            });
        assert_eq!(
            bytes_u32(outputs.last().expect("the fixture declares an output")),
            expected,
            "Fix: native launch {launch} must honor all five barriers; wrong values on launch 2 \
             specifically mean the module-scope counter was not reset between launches."
        );
    }
}
