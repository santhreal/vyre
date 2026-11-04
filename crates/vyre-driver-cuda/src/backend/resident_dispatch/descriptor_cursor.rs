//! Walking the descriptor-ordered resource lists of a resident dispatch: how
//! many handles a plan needs, and the cursor that takes the next handle or
//! bound resource in that order.

use vyre_driver::BackendError;

use crate::backend::plan::CudaDispatchPlan;
use crate::backend::resident::{CudaDispatchBinding, CudaResidentBuffer};

pub(crate) fn resident_required_handles(
    prepared: &CudaDispatchPlan,
) -> Result<usize, BackendError> {
    prepared
        .bindings
        .bindings
        .len()
        .checked_sub(prepared.bindings.shared_indices.len())
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident binding plan has {} binding(s) but {} shared binding index(es). Rebuild the dispatch plan before launching.",
                prepared.bindings.bindings.len(),
                prepared.bindings.shared_indices.len()
            ),
        })
}

macro_rules! define_next_descriptor_resource {
    ($name:ident $(<$lifetime:lifetime>)?, $resource:ty, $items:ident, $cursor:ident, $resource_name:literal, $rebuild:literal) => {
        #[doc = concat!("Take the next ", $resource_name, " in descriptor order.")]
        pub(crate) fn $name $(<$lifetime>)? (
            $items: &[$resource],
            $cursor: &mut usize,
            context: &'static str,
        ) -> Result<$resource, BackendError> {
            let index = *$cursor;
            let Some(resource) = $items.get(index).copied() else {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA {context} ran out of {} at descriptor slot {index} after receiving {} item(s). Validate the resource count against the binding plan before launch.",
                        $resource_name,
                        $items.len()
                    ),
                });
            };
            *$cursor = $cursor
                .checked_add(1)
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA {context} {} cursor overflowed at descriptor slot {index}. {}",
                        $resource_name,
                        $rebuild
                    ),
                })?;
            Ok(resource)
        }
    };
}

define_next_descriptor_resource!(
    next_resident_handle,
    CudaResidentBuffer,
    handles,
    next_handle,
    "resident buffer handles",
    "Rebuild the resident binding plan before launch."
);

define_next_descriptor_resource!(
    next_dispatch_binding<'a>,
    CudaDispatchBinding<'a>,
    bindings,
    next_binding,
    "bound resources",
    "Rebuild the resident binding plan before launch."
);
