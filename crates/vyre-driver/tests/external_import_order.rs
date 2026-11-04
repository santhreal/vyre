//! Proves external-import authentication answers in one order for every backend.
//!
//! WHY: a descriptor can be wrong twice. Before the neutral importer owned the
//! order, each concrete driver chose its own: two drivers named the
//! unsupported combination first and one named the pitch first, so the same
//! doubly-wrong descriptor produced two different errors depending on which
//! device the caller reached for. The order is now stated once, in
//! `vyre_driver::external_import`, and this pins it: the combination the
//! backend cannot import is named before the pitch, because a caller told only
//! about the pitch aligns it and is refused a second time for the reason that
//! was true all along.
//!
//! What this does not judge: which combinations a backend refuses. That is a
//! hardware fact and belongs to the concrete driver's policy.

use vyre_driver::external_import::{
    ExternalImportDescriptor, ExternalImportHandle, ExternalImportPolicy, ExternalResourceImporter,
};
use vyre_driver::{
    ColorInterpretation, ExternalMemoryKind, ImageDimensions, ImageFormat, ResourceAbiError,
    ResourcePermittedUsages, TimelineSyncProtocol,
};

/// A handle that stands for nothing but its own memory class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProbeHandle(u64);

impl ExternalImportHandle for ProbeHandle {
    fn memory_kind(&self) -> ExternalMemoryKind {
        ExternalMemoryKind::DmaBuf
    }

    fn provenance_tag(&self) -> u64 {
        self.0
    }
}

/// A policy that refuses exactly one format, so the refusal is unambiguous.
#[derive(Clone, Copy, Debug)]
struct ProbePolicy;

impl ExternalImportPolicy for ProbePolicy {
    type Handle = ProbeHandle;

    const OWNER: &'static str = "external import order probe";

    const COLOR: ColorInterpretation = ColorInterpretation::Srgb;

    fn refuse_unsupported(
        descriptor: &ExternalImportDescriptor<Self::Handle>,
    ) -> Result<(), ResourceAbiError> {
        if descriptor.format.is_depth_stencil() {
            return Err(descriptor.unsupported_combination());
        }
        Ok(())
    }
}

/// A descriptor wrong in one named way, at a pitch that is 256-byte aligned.
fn descriptor(format: ImageFormat, row_pitch_bytes: u32) -> ExternalImportDescriptor<ProbeHandle> {
    ExternalImportDescriptor {
        resource_id: 7,
        format,
        dimensions: ImageDimensions::d2(64, 64),
        row_pitch_bytes,
        handle: ProbeHandle(0x1234),
        permitted_usages: ResourcePermittedUsages::SAMPLED,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    }
}

#[test]
fn an_unsupported_combination_is_named_before_an_unaligned_pitch() {
    let importer = ExternalResourceImporter::<ProbePolicy>::new(1);
    // Depth32Float over 64 pixels needs 256 bytes; 100 is short and unaligned,
    // so this descriptor is wrong in both ways at once.
    let refusal = importer
        .authenticate_import(&descriptor(ImageFormat::Depth32Float, 100))
        .expect_err("a doubly wrong descriptor is refused");

    assert_eq!(
        refusal,
        ResourceAbiError::UnsupportedZeroCopyNegotiation {
            resource_id: 7,
            format: ImageFormat::Depth32Float,
            memory_kind: ExternalMemoryKind::DmaBuf,
        }
    );
}

#[test]
fn an_unaligned_pitch_is_named_when_the_combination_is_admissible() {
    let importer = ExternalResourceImporter::<ProbePolicy>::new(1);
    let refusal = importer
        .authenticate_import(&descriptor(ImageFormat::Rgba8Unorm, 100))
        .expect_err("an unaligned pitch is refused");

    assert_eq!(
        refusal,
        ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id: 7,
            provided_pitch: 100,
            required_pitch: 256,
        }
    );
}

#[test]
fn a_short_pitch_states_the_next_aligned_pitch_and_not_the_one_it_was_given() {
    let importer = ExternalResourceImporter::<ProbePolicy>::new(1);
    let mut wide = descriptor(ImageFormat::Rgba8Unorm, 512);
    // 300 pixels of Rgba8Unorm need 1200 bytes, which rounds to 1280.
    wide.dimensions = ImageDimensions::d2(300, 4);
    let refusal = importer
        .authenticate_import(&wide)
        .expect_err("a pitch shorter than one row is refused");

    assert_eq!(
        refusal,
        ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id: 7,
            provided_pitch: 512,
            required_pitch: 1280,
        }
    );
}

#[test]
fn degenerate_dimensions_are_named_before_the_combination() {
    let importer = ExternalResourceImporter::<ProbePolicy>::new(1);
    let mut degenerate = descriptor(ImageFormat::Depth32Float, 256);
    degenerate.dimensions = ImageDimensions::d2(0, 64);
    let refusal = importer
        .authenticate_import(&degenerate)
        .expect_err("a degenerate surface is refused");

    assert_eq!(
        refusal,
        ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id: 7,
            provided_pitch: 256,
            required_pitch: 256,
        }
    );
}
