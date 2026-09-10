//! Declared semantics of the visual operations the cpu-ref oracle checks.
//!
//! WHY: three registered visual operations disagreed with their conformance
//! fixtures. `dirty_region_patch` read past the end of its declared `patch`
//! buffer, and the `rgba_to_grayscale` and `text_run` fixtures contradicted the
//! luma formula and the channel order their operations declare. Every case here
//! derives the expected result from the declaration, either the formula in the
//! operation's doc comment or the packing stated in the `visual` module
//! documentation, so a fixture recorded from a different implementation fails
//! here instead of being adopted as the contract.
//!
//! Not covered: GPU emission. Every case runs on the pure-Rust reference
//! interpreter, which reports an out-of-bounds access instead of faulting on it,
//! so these cases prove the access pattern rather than the emitted code.

use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;
use vyre_libs_visual::visual::{dirty_region_patch_rgba, rgba_to_grayscale, text_run_blend};
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

const GRAYSCALE_OP: &str = "vyre-libs::visual::rgba_to_grayscale";
const TEXT_RUN_OP: &str = "vyre-libs::visual::text_run";
const DIRTY_REGION_OP: &str = "vyre-libs::visual::dirty_region_patch";

/// Background pixel used by the patch cases: opaque black plus the atlas index,
/// so a pixel that came from the wrong source is identifiable.
const ATLAS_TAG: u32 = 0xFF00_0000;
/// Patch pixel tag, distinct from [`ATLAS_TAG`] in the green channel.
const PATCH_TAG: u32 = 0xFF01_0000;

fn registration(id: &str) -> &'static OperationRegistration {
    inventory::iter::<OperationRegistration>
        .into_iter()
        .find(|reg| reg.id == id)
        .unwrap_or_else(|| panic!("`{id}` must be a registered operation"))
}

fn fixture_cases(reg: &OperationRegistration) -> Vec<Vec<Vec<u8>>> {
    reg.test_inputs
        .unwrap_or_else(|| panic!("`{}` must declare fixture inputs", reg.id))()
}

fn fixture_expected(reg: &OperationRegistration) -> Vec<Vec<Vec<u8>>> {
    reg.expected_output
        .unwrap_or_else(|| panic!("`{}` must declare fixture outputs", reg.id))()
}

fn fixture_program(reg: &OperationRegistration) -> Program {
    reg.build
        .unwrap_or_else(|| panic!("`{}` must declare a program builder", reg.id))()
}

/// Host inputs for one fixture case, in the interpreter's input ABI order.
///
/// A fixture case carries one buffer per program declaration, output buffers
/// included, so the host-staged ones are selected by the single definition of
/// that ABI rather than by position. A case that already carries only the inputs
/// is passed through.
fn fixture_inputs(program: &Program, case: &[Vec<u8>]) -> Vec<Value> {
    let host_input_count = program
        .buffers()
        .iter()
        .filter(|decl| decl.consumes_host_input())
        .count();
    if case.len() == host_input_count {
        return case.iter().map(|bytes| Value::from(bytes.as_slice())).collect();
    }
    assert_eq!(
        case.len(),
        program.buffers().len(),
        "a fixture case supplies either one buffer per program declaration or one per host input"
    );
    program
        .buffers()
        .iter()
        .zip(case)
        .filter(|(decl, _)| decl.consumes_host_input())
        .map(|(_, bytes)| Value::from(bytes.as_slice()))
        .collect()
}

/// Output bytes plus the count of accesses that left a declared buffer.
fn run(program: &Program, inputs: &[Value]) -> (Vec<Vec<u8>>, u64) {
    let (outputs, report) = vyre_reference::reference_eval_oob_report(program, inputs)
        .expect("reference evaluation must succeed");
    (
        outputs.iter().map(Value::to_bytes).collect(),
        report.total(),
    )
}

// ================================================================
// rgba_to_grayscale: the declared Rec. 601 fixed-point luma
// ================================================================

/// `Y = (77 * R + 150 * G + 29 * B + 128) >> 8`, the formula `rgba_to_grayscale`
/// declares, evaluated on the host.
///
/// The weights sum to 256 and the `+ 128` rounds to nearest, so red gives 77 and
/// green gives 149. A float Rec. 601 implementation gives 76 and 150 instead;
/// this states the integer formula so the two cannot be confused.
fn declared_luma(pixel: u32) -> u32 {
    let r = pixel & 0xFF;
    let g = (pixel >> 8) & 0xFF;
    let b = (pixel >> 16) & 0xFF;
    ((77 * r + 150 * g + 29 * b + 128) >> 8).min(255)
}

/// The declared output pixel: luma in R, G and B, alpha carried through.
fn declared_gray_pixel(pixel: u32) -> u32 {
    let y = declared_luma(pixel);
    y | (y << 8) | (y << 16) | ((pixel >> 24) << 24)
}

#[test]
fn registered_grayscale_fixture_states_the_declared_luma() {
    let reg = registration(GRAYSCALE_OP);
    let cases = fixture_cases(reg);
    let expected = fixture_expected(reg);
    assert_eq!(cases.len(), expected.len(), "one oracle per fixture case");

    for (index, (case, expected_buffers)) in cases.iter().zip(&expected).enumerate() {
        let declared: Vec<u32> = decode_u32_le_bytes_all(&case[0])
            .into_iter()
            .map(declared_gray_pixel)
            .collect();
        let recorded = decode_u32_le_bytes_all(&expected_buffers[0]);
        assert_eq!(
            recorded, declared,
            "case {index}: the expected_output fixture must state \
             `Y = (77 * R + 150 * G + 29 * B + 128) >> 8` per pixel"
        );
    }
}

#[test]
fn grayscale_program_computes_the_declared_luma() {
    let pixels = [
        0xFF00_00FFu32,
        0xFF00_FF00,
        0xFFFF_0000,
        0xFF80_8080,
        0xFFFF_FFFF,
        0x0000_0000,
        0x8003_0201,
    ];
    let program = rgba_to_grayscale("in", "out", u32::try_from(pixels.len()).unwrap());
    let inputs = vec![Value::from(pack_u32_slice(&pixels))];

    let (outputs, out_of_bounds) = run(&program, &inputs);
    assert_eq!(
        out_of_bounds, 0,
        "rgba_to_grayscale must access only its declared buffers"
    );
    let declared: Vec<u32> = pixels.iter().copied().map(declared_gray_pixel).collect();
    assert_eq!(
        decode_u32_le_bytes_all(&outputs[0]),
        declared,
        "the composition must round the declared luma to nearest, not truncate it"
    );
}

// ================================================================
// text_run: the declared channel order
// ================================================================

#[test]
fn registered_text_run_fixture_states_the_glyph_color_in_declared_channel_order() {
    let reg = registration(TEXT_RUN_OP);
    let cases = fixture_cases(reg);
    let expected = fixture_expected(reg);
    assert_eq!(cases.len(), expected.len(), "one oracle per fixture case");

    for (index, (case, expected_buffers)) in cases.iter().zip(&expected).enumerate() {
        let glyphs = decode_u32_le_bytes_all(&case[0]);
        let atlas = decode_u32_le_bytes_all(&case[1]);
        let background = decode_u32_le_bytes_all(&case[2]);
        // The case places one 1x1 glyph at (0,0), sampled from atlas (0,0).
        assert_eq!(
            glyphs[..6],
            [0, 0, 1, 1, 0, 0],
            "case {index}: this assertion reads a 1x1 glyph at the origin"
        );
        assert_eq!(
            atlas[0] >> 24,
            255,
            "case {index}: coverage at atlas (0,0) must be full"
        );
        assert_eq!(
            background[0], 0xFF00_0000,
            "case {index}: the background under the glyph must be opaque black"
        );

        // Full coverage over an opaque background leaves the glyph color
        // unchanged, so the first expected pixel is the glyph color word. Both
        // are read as `visual` declares: bits [7:0] R, [15:8] G, [23:16] B,
        // [31:24] A.
        let recorded = decode_u32_le_bytes_all(&expected_buffers[0]);
        assert_eq!(
            recorded[0], glyphs[6],
            "case {index}: at full coverage the output pixel is the glyph color, \
             so the color word and the expected pixel must use one channel order"
        );
    }
}

#[test]
fn text_run_writes_each_glyph_color_channel_to_its_declared_byte() {
    // R = 0x11, G = 0x22, B = 0x33, A = 0xFF under the declared packing.
    let color = 0xFF33_2211u32;
    let program = text_run_blend("glyphs", 1, "atlas", 2, 2, "bg", "out", 2, 2);
    let glyphs = [0u32, 0, 1, 1, 0, 0, color];
    let atlas = [0xFF00_0000u32, 0, 0, 0];
    let background = [0xFF00_0000u32; 4];
    let inputs = vec![
        Value::from(pack_u32_slice(&glyphs)),
        Value::from(pack_u32_slice(&atlas)),
        Value::from(pack_u32_slice(&background)),
    ];

    let (outputs, out_of_bounds) = run(&program, &inputs);
    assert_eq!(
        out_of_bounds, 0,
        "text_run must access only its declared buffers"
    );
    assert_eq!(
        &outputs[0][..4],
        &[0x11, 0x22, 0x33, 0xFF],
        "byte 0 carries R, byte 1 G, byte 2 B, byte 3 A"
    );
    assert_eq!(
        decode_u32_le_bytes_all(&outputs[0])[1..],
        [0xFF00_0000, 0xFF00_0000, 0xFF00_0000],
        "a pixel outside the glyph box keeps the background"
    );
}

// ================================================================
// dirty_region_patch: every access inside the declared buffers
// ================================================================

/// The patched surface, computed on the host from the operation's definition:
/// a pixel inside the destination rectangle comes from `patch` at
/// `(ax - dest_x) + (ay - dest_y) * patch_w`, every other pixel from `atlas`.
fn declared_patch_result(
    atlas_w: u32,
    atlas_h: u32,
    patch_w: u32,
    patch_h: u32,
    dest_x: u32,
    dest_y: u32,
) -> Vec<u32> {
    let end_x = atlas_w.min(dest_x.saturating_add(patch_w));
    let end_y = atlas_h.min(dest_y.saturating_add(patch_h));
    (0..atlas_w * atlas_h)
        .map(|i| {
            let ax = i % atlas_w;
            let ay = i / atlas_w;
            if ax >= dest_x && ax < end_x && ay >= dest_y && ay < end_y {
                PATCH_TAG | ((ax - dest_x) + (ay - dest_y) * patch_w)
            } else {
                ATLAS_TAG | i
            }
        })
        .collect()
}

/// Run one patch geometry and return the surface plus the out-of-bounds count.
fn patch_run(
    atlas_w: u32,
    atlas_h: u32,
    patch_w: u32,
    patch_h: u32,
    dest_x: u32,
    dest_y: u32,
) -> (Vec<u32>, u64) {
    let program = dirty_region_patch_rgba(
        "atlas", "patch", atlas_w, atlas_h, patch_w, patch_h, dest_x, dest_y, "out",
    );
    let atlas: Vec<u32> = (0..atlas_w * atlas_h).map(|i| ATLAS_TAG | i).collect();
    let patch: Vec<u32> = (0..patch_w * patch_h).map(|i| PATCH_TAG | i).collect();
    let inputs = vec![
        Value::from(pack_u32_slice(&atlas)),
        Value::from(pack_u32_slice(&patch)),
    ];

    let (outputs, out_of_bounds) = run(&program, &inputs);
    (decode_u32_le_bytes_all(&outputs[0]), out_of_bounds)
}

#[test]
fn registered_dirty_region_patch_case_accesses_only_declared_buffers() {
    let reg = registration(DIRTY_REGION_OP);
    let program = fixture_program(reg);
    let cases = fixture_cases(reg);
    let expected = fixture_expected(reg);
    assert_eq!(cases.len(), expected.len(), "one oracle per fixture case");

    for (index, (case, expected_buffers)) in cases.iter().zip(&expected).enumerate() {
        let inputs = fixture_inputs(&program, case);
        let (outputs, out_of_bounds) = run(&program, &inputs);
        assert_eq!(
            out_of_bounds, 0,
            "case {index}: dirty_region_patch must access only its declared buffers"
        );
        assert_eq!(
            outputs[0], expected_buffers[0],
            "case {index}: the patched surface must match the declared oracle"
        );
    }
}

/// The largest in-bounds destination rectangle: the patch covers the whole
/// atlas, so every pixel is patched and the last patch index read is
/// `patch_w * patch_h - 1`.
#[test]
fn dirty_region_patch_largest_region_stays_in_bounds() {
    let (surface, out_of_bounds) = patch_run(4, 4, 4, 4, 0, 0);
    assert_eq!(
        out_of_bounds, 0,
        "a destination rectangle covering the whole atlas must read no pixel outside `patch`"
    );
    assert_eq!(surface, declared_patch_result(4, 4, 4, 4, 0, 0));
    assert_eq!(
        surface[15],
        PATCH_TAG | 15,
        "the last surface pixel must come from the last patch element"
    );
}

/// The smallest destination rectangle, at every position on the surface. A 1x1
/// region leaves fifteen of sixteen pixels outside it, and each of those
/// computes a patch index that is out of range before it is discarded.
#[test]
fn dirty_region_patch_smallest_region_stays_in_bounds_at_every_destination() {
    for dest_y in 0..4 {
        for dest_x in 0..4 {
            let (surface, out_of_bounds) = patch_run(4, 4, 1, 1, dest_x, dest_y);
            assert_eq!(
                out_of_bounds, 0,
                "a 1x1 region at ({dest_x}, {dest_y}) must read no pixel outside `patch`"
            );
            assert_eq!(
                surface,
                declared_patch_result(4, 4, 1, 1, dest_x, dest_y),
                "a 1x1 region at ({dest_x}, {dest_y}) must patch exactly that pixel"
            );
        }
    }
}

/// A destination rectangle reaching past the surface is clipped to it, and the
/// clipped rectangle still indexes `patch` inside its declared count.
#[test]
fn dirty_region_patch_region_overhanging_the_atlas_stays_in_bounds() {
    let (surface, out_of_bounds) = patch_run(4, 4, 3, 3, 2, 2);
    assert_eq!(
        out_of_bounds, 0,
        "a region reaching past the atlas must read no pixel outside `patch`"
    );
    assert_eq!(surface, declared_patch_result(4, 4, 3, 3, 2, 2));
    assert_eq!(
        surface[15],
        PATCH_TAG | 4,
        "surface (3,3) must read patch (1,1), the last cell inside the clipped rectangle"
    );
}

/// A destination rectangle entirely off the surface patches nothing and reads
/// nothing from `patch`.
#[test]
fn dirty_region_patch_region_outside_the_atlas_copies_the_atlas() {
    let (surface, out_of_bounds) = patch_run(4, 4, 2, 2, 4, 4);
    assert_eq!(
        out_of_bounds, 0,
        "a region off the surface must read no pixel outside `patch`"
    );
    let atlas: Vec<u32> = (0..16).map(|i| ATLAS_TAG | i).collect();
    assert_eq!(surface, atlas, "nothing inside the atlas is patched");
}

/// A destination origin near the u32 limit builds, patches nothing and reads
/// nothing from `patch`.
///
/// The exclusive region bound is `dest + extent`, which wraps for an origin this
/// large. A wrapped bound is below every `ax`, so the region test would admit
/// every pixel and the patch index would run past the declared count.
#[test]
fn dirty_region_patch_destination_near_the_u32_limit_patches_nothing() {
    let program = dirty_region_patch_rgba("atlas", "patch", 4, 4, 2, 2, u32::MAX, u32::MAX, "out");
    let atlas: Vec<u32> = (0..16).map(|i| ATLAS_TAG | i).collect();
    let patch: Vec<u32> = (0..4).map(|i| PATCH_TAG | i).collect();
    let inputs = vec![
        Value::from(pack_u32_slice(&atlas)),
        Value::from(pack_u32_slice(&patch)),
    ];

    let (outputs, out_of_bounds) = run(&program, &inputs);
    assert_eq!(
        out_of_bounds, 0,
        "a destination origin at the u32 limit must read no pixel outside `patch`"
    );
    assert_eq!(
        decode_u32_le_bytes_all(&outputs[0]),
        atlas,
        "a region that cannot intersect the atlas patches nothing"
    );
}

// ================================================================
// The class, across every registered visual operation
// ================================================================

/// Every registered visual operation keeps every access inside its declared
/// buffers on its own conformance fixture.
///
/// WHY: a `select` evaluates both arms, on the reference interpreter and on
/// hardware alike, so a load written into the discarded arm still issues.
/// `dirty_region_patch` indexed `patch` at `ax - dest_x`, which wraps to a huge
/// u32 for a pixel left of the destination rectangle, and the read left the
/// buffer for three of four pixels. The variant space is the registry rather
/// than a written list, so a visual operation added with the same shape turns
/// this case red instead of passing unnoticed.
///
/// Not covered: an operation whose fixture never reaches the ungated arm. This
/// proves the fixture's access pattern, not every reachable index.
#[test]
fn every_registered_visual_program_accesses_only_declared_buffers() {
    let mut scanned = Vec::new();
    let mut failures = Vec::new();

    for reg in inventory::iter::<OperationRegistration> {
        if !reg.id.starts_with("vyre-libs::visual::") {
            continue;
        }
        let (Some(build), Some(test_inputs)) = (reg.build, reg.test_inputs) else {
            failures.push(format!(
                "{}: exposes no program builder or no fixture, so its access pattern cannot be \
                 scanned. Give it both, or move it out of the visual namespace.",
                reg.id
            ));
            continue;
        };
        let program = build();
        scanned.push(reg.id);
        for (index, case) in test_inputs().iter().enumerate() {
            let inputs = fixture_inputs(&program, case);
            match vyre_reference::reference_eval_oob_report(&program, &inputs) {
                Ok((_, report)) if report.total() == 0 => {}
                Ok((_, report)) => failures.push(format!(
                    "{} case {index}: {} load(s), {} store(s) and {} atomic(s) left a declared \
                     buffer. Fix: gate the access with an explicit bound.",
                    reg.id, report.oob_loads, report.oob_stores, report.oob_atomics
                )),
                Err(error) => {
                    failures.push(format!("{} case {index}: evaluation failed: {error}", reg.id));
                }
            }
        }
    }

    for id in [GRAYSCALE_OP, TEXT_RUN_OP, DIRTY_REGION_OP] {
        assert!(
            scanned.contains(&id),
            "`{id}` was not scanned, so this case no longer reaches the registry it claims to \
             enumerate. Fix: keep the operation registered under the visual namespace with a \
             program builder and a fixture."
        );
    }
    assert!(
        failures.is_empty(),
        "{} visual operation case(s) left their declared buffers:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
