use super::super::lru_tick_cache::LruTickCache;
use super::{
    default_buffers, finite_body_with_io, persistent_body_with_io, prepare_megakernel_program,
    wrap_megakernel_program, wrap_persistent_megakernel_program,
};
use std::cell::RefCell;
use std::sync::Arc;
use vyre_foundation::ir::Program;

const EMPTY_TEMPLATE_CACHE_CAP: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EmptyTemplateKey {
    workgroup_size_x: u32,
    slot_count: u32,
    include_io_polling: bool,
    finite_once: bool,
    control_report_only: bool,
}

type EmptyTemplateCache = LruTickCache<EmptyTemplateKey, Arc<Program>>;

fn empty_template_cache() -> EmptyTemplateCache {
    EmptyTemplateCache::with_capacity(EMPTY_TEMPLATE_CACHE_CAP)
}

thread_local! {
    static EMPTY_TEMPLATE_CACHE: RefCell<EmptyTemplateCache> =
        RefCell::new(empty_template_cache());
}

pub(super) fn cached_empty_sharded_program(
    workgroup_size_x: u32,
    slot_count: u32,
    include_io_polling: bool,
) -> Program {
    cached_empty_sharded_program_shared(workgroup_size_x, slot_count, include_io_polling)
        .as_ref()
        .clone()
}

pub(super) fn cached_empty_sharded_program_shared(
    workgroup_size_x: u32,
    slot_count: u32,
    include_io_polling: bool,
) -> Arc<Program> {
    let key = EmptyTemplateKey {
        workgroup_size_x,
        slot_count,
        include_io_polling,
        finite_once: false,
        control_report_only: false,
    };
    cached_template(key, || {
        wrap_persistent_megakernel_program(
            workgroup_size_x,
            slot_count,
            persistent_body_with_io(workgroup_size_x, &[], include_io_polling),
        )
    })
}

pub(super) fn cached_empty_sharded_once_program(workgroup_size_x: u32, slot_count: u32) -> Program {
    cached_empty_sharded_once_program_shared(workgroup_size_x, slot_count)
        .as_ref()
        .clone()
}

pub(super) fn cached_empty_sharded_once_program_shared(
    workgroup_size_x: u32,
    slot_count: u32,
) -> Arc<Program> {
    let key = EmptyTemplateKey {
        workgroup_size_x,
        slot_count,
        include_io_polling: false,
        finite_once: true,
        control_report_only: false,
    };
    cached_template(key, || {
        wrap_megakernel_program(
            workgroup_size_x,
            slot_count,
            finite_body_with_io(workgroup_size_x, &[], false),
        )
    })
}

pub(super) fn cached_empty_sharded_once_control_report_program_shared(
    workgroup_size_x: u32,
    slot_count: u32,
) -> Arc<Program> {
    let key = EmptyTemplateKey {
        workgroup_size_x,
        slot_count,
        include_io_polling: false,
        finite_once: true,
        control_report_only: true,
    };
    cached_template(key, || {
        let mut buffers = default_buffers(slot_count);
        for buffer in buffers.iter_mut().skip(1) {
            buffer.output_byte_range = Some(0..0);
        }
        prepare_megakernel_program(Program::wrapped(
            buffers,
            [workgroup_size_x, 1, 1],
            finite_body_with_io(workgroup_size_x, &[], false),
        ))
    })
}

/// Return the cached template for `key`, building and installing it on a miss.
fn cached_template(key: EmptyTemplateKey, build: impl FnOnce() -> Program) -> Arc<Program> {
    if let Some(program) =
        EMPTY_TEMPLATE_CACHE.with(|cache| cache.borrow_mut().get(&key).map(Arc::clone))
    {
        return program;
    }

    let program = Arc::new(build());
    EMPTY_TEMPLATE_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, Arc::clone(&program));
    });
    program
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_template_cache_refreshes_hot_template_on_hit() {
        EMPTY_TEMPLATE_CACHE.with(|cache| cache.borrow_mut().clear());
        let hot = cached_empty_sharded_program_shared(1, 1, false);
        for slot_count in 2..=EMPTY_TEMPLATE_CACHE_CAP as u32 {
            let _ = cached_empty_sharded_program_shared(1, slot_count, false);
        }
        let hot_after_hit = cached_empty_sharded_program_shared(1, 1, false);
        assert!(Arc::ptr_eq(&hot, &hot_after_hit));
        let _ =
            cached_empty_sharded_program_shared(1, (EMPTY_TEMPLATE_CACHE_CAP + 1) as u32, false);
        let hot_after_eviction = cached_empty_sharded_program_shared(1, 1, false);
        assert!(Arc::ptr_eq(&hot, &hot_after_eviction));
    }

    #[test]
    fn empty_control_report_template_is_cached_by_arc() {
        EMPTY_TEMPLATE_CACHE.with(|cache| cache.borrow_mut().clear());
        let first = cached_empty_sharded_once_control_report_program_shared(64, 128);
        let second = cached_empty_sharded_once_control_report_program_shared(64, 128);

        assert!(Arc::ptr_eq(&first, &second));
    }
}
