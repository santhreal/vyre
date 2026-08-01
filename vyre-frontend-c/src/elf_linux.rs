//! Linux-native **ELF64 relocatable** objects so `cc` / `ld` accept `vyre-frontend-c` outputs.
//!
//! Each translation unit is emitted as ET_REL with a one-instruction `.text`
//! carrier and a custom section holding the full `VYRECOB2` blob. Link mode uses a tiny `_start`
//! object (`exit(0)` syscall) plus `-nostdlib`.

use std::path::Path;

use object::write::{Object, StandardSection, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
};

use crate::hash::blake3_128;

fn section_name_for_tu(source: &Path) -> Vec<u8> {
    let tag = blake3_128(source.as_os_str().as_encoded_bytes());
    let mut name = String::from(".vyrecob2.");
    for byte in tag {
        use std::fmt::Write as _;

        let _ = write!(&mut name, "{byte:02x}");
    }
    name.into_bytes()
}

/// x86_64 ET_REL: `.text` = `ret`, custom section = `vyrecob2` payload,
/// local carrier symbol.
pub fn emit_translation_unit_relocatable(
    vyrecob2: &[u8],
    source_path: &Path,
) -> Result<Vec<u8>, String> {
    match std::env::consts::ARCH {
        "x86_64" => emit_tu_x86_64(vyrecob2, source_path),
        "aarch64" => emit_tu_aarch64(vyrecob2, source_path),
        other => Err(format!(
            "vyre-frontend-c: ELF emission is unsupported for host arch `{other}` (supported: x86_64, aarch64)"
        )),
    }
}

fn emit_text_relocatable(
    architecture: Architecture,
    code: &[u8],
    alignment: u64,
    symbol_name: &[u8],
    scope: SymbolScope,
    custom_section: Option<(Vec<u8>, &[u8])>,
) -> Result<Vec<u8>, String> {
    let mut obj = Object::new(BinaryFormat::Elf, architecture, Endianness::Little);
    let text = obj.section_id(StandardSection::Text);
    let offset = obj.append_section_data(text, code, alignment);
    obj.add_symbol(Symbol {
        name: symbol_name.to_vec(),
        value: offset,
        size: code.len() as u64,
        kind: SymbolKind::Text,
        scope,
        weak: false,
        section: SymbolSection::Section(text),
        flags: SymbolFlags::None,
    });
    if let Some((name, contents)) = custom_section {
        let section = obj.add_section(Vec::new(), name, SectionKind::Data);
        obj.append_section_data(section, contents, 1);
    }
    obj.write().map_err(|error| error.to_string())
}

fn emit_tu_x86_64(vyrecob2: &[u8], source_path: &Path) -> Result<Vec<u8>, String> {
    emit_text_relocatable(
        Architecture::X86_64,
        &[0xC3],
        1,
        b"vyre_tu_entry",
        SymbolScope::Compilation,
        Some((section_name_for_tu(source_path), vyrecob2)),
    )
}

fn emit_tu_aarch64(vyrecob2: &[u8], source_path: &Path) -> Result<Vec<u8>, String> {
    emit_text_relocatable(
        Architecture::Aarch64,
        &[0xC0, 0x03, 0x5F, 0xD6],
        4,
        b"vyre_tu_entry",
        SymbolScope::Compilation,
        Some((section_name_for_tu(source_path), vyrecob2)),
    )
}

/// Minimal relocatable object defining global `_start` as `exit(0)` for the host arch.
pub fn emit_link_startup_relocatable() -> Result<Vec<u8>, String> {
    match std::env::consts::ARCH {
        "x86_64" => emit_start_x86_64(),
        "aarch64" => emit_start_aarch64(),
        other => Err(format!(
            "vyre-frontend-c: link startup object is unsupported for `{other}` (supported: x86_64, aarch64)"
        )),
    }
}

/// Linux x86_64: `mov $60,%rax; xor %edi,%edi; syscall` (exit(0)).
fn emit_start_x86_64() -> Result<Vec<u8>, String> {
    emit_text_relocatable(
        Architecture::X86_64,
        &[
            0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00, 0x48, 0x31, 0xff, 0x0f, 0x05,
        ],
        1,
        b"_start",
        SymbolScope::Linkage,
        None,
    )
}

/// Linux aarch64: `mov x8, #93; mov x0, #0; svc #0` (exit(0)).
fn emit_start_aarch64() -> Result<Vec<u8>, String> {
    emit_text_relocatable(
        Architecture::Aarch64,
        &[
            0x48, 0x12, 0x80, 0xd2, // mov x8, #93
            0x00, 0x00, 0x80, 0xd2, // mov x0, #0
            0x01, 0x00, 0x00, 0xd4, // svc #0
        ],
        4,
        b"_start",
        SymbolScope::Linkage,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tu_object_has_elf_magic() {
        let bytes = emit_translation_unit_relocatable(b"VYREC02\0", Path::new("x.c")).unwrap();
        assert_eq!(&bytes[0..4], b"\x7fELF");
    }

    #[test]
    fn startup_object_has_elf_magic() {
        let bytes = emit_link_startup_relocatable().unwrap();
        assert_eq!(&bytes[0..4], b"\x7fELF");
    }

    #[test]
    fn tu_section_name_uses_128_bit_path_tag() {
        let name = section_name_for_tu(Path::new("src/main.c"));
        let name = std::str::from_utf8(&name).expect("Fix: section name must be ASCII");
        assert!(name.starts_with(".vyrecob2."));
        assert_eq!(name.len(), ".vyrecob2.".len() + 32);
    }
}
