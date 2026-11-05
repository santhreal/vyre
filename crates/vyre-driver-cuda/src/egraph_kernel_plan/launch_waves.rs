//! Turning an item count into bounded launch waves, plus the checked
//! arithmetic every wave count and plan size goes through.

use crate::numeric::CUDA_NUMERIC;

use super::{
    CudaEGraphKernelLaunchConfig, CudaEGraphKernelPass, CudaEGraphKernelPlanError,
    CudaEGraphKernelWave, CudaEGraphSignaturePairWave,
};

pub(super) fn append_pass_waves(
    waves: &mut Vec<CudaEGraphKernelWave>,
    total_items: &mut u64,
    total_blocks: &mut u64,
    pass: CudaEGraphKernelPass,
    item_count: u64,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<(), CudaEGraphKernelPlanError> {
    if item_count == 0 {
        return Ok(());
    }
    let items_per_wave = u64::from(config.threads_per_block)
        .checked_mul(u64::from(config.max_blocks_per_launch))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "items per launch wave",
        })?;
    let mut first_item = 0_u64;
    while first_item < item_count {
        let remaining = item_count - first_item;
        let wave_items = remaining.min(items_per_wave);
        let blocks = ceil_div_u64(wave_items, u64::from(config.threads_per_block))?;
        let blocks =
            u32::try_from(blocks).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
                field: "blocks per launch wave",
            })?;
        waves.push(CudaEGraphKernelWave {
            pass,
            first_item,
            item_count: wave_items,
            blocks,
            threads_per_block: config.threads_per_block,
        });
        *total_items = total_items.checked_add(wave_items).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "total logical items",
            },
        )?;
        *total_blocks = total_blocks.checked_add(u64::from(blocks)).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "total blocks",
            },
        )?;
        first_item =
            first_item
                .checked_add(wave_items)
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "next wave first item",
                })?;
    }
    Ok(())
}

pub(super) fn append_signature_pair_waves(
    pair_waves: &mut Vec<CudaEGraphSignaturePairWave>,
    total_blocks: &mut u64,
    bucket_index: u32,
    pair_count: u64,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<(), CudaEGraphKernelPlanError> {
    let items_per_wave = u64::from(config.threads_per_block)
        .checked_mul(u64::from(config.max_blocks_per_launch))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "items per signature pair launch wave",
        })?;
    let mut first_pair = 0_u64;
    while first_pair < pair_count {
        let remaining = pair_count - first_pair;
        let wave_pairs = remaining.min(items_per_wave);
        let blocks = ceil_div_u64(wave_pairs, u64::from(config.threads_per_block))?;
        let blocks =
            u32::try_from(blocks).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
                field: "blocks per signature pair launch wave",
            })?;
        pair_waves.push(CudaEGraphSignaturePairWave {
            bucket_index,
            first_pair,
            pair_count: wave_pairs,
            blocks,
            threads_per_block: config.threads_per_block,
        });
        *total_blocks = total_blocks.checked_add(u64::from(blocks)).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "signature pair total blocks",
            },
        )?;
        first_pair =
            first_pair
                .checked_add(wave_pairs)
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "next signature pair first item",
                })?;
    }
    Ok(())
}

pub(super) fn wave_count_for(
    item_count: u64,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<u64, CudaEGraphKernelPlanError> {
    if item_count == 0 {
        return Ok(0);
    }
    let items_per_wave = u64::from(config.threads_per_block)
        .checked_mul(u64::from(config.max_blocks_per_launch))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "items per launch wave",
        })?;
    ceil_div_u64(item_count, items_per_wave)
}

pub(super) fn ceil_div_u64(
    numerator: u64,
    denominator: u64,
) -> Result<u64, CudaEGraphKernelPlanError> {
    if denominator == 0 {
        return Err(CudaEGraphKernelPlanError::CountOverflow {
            field: "ceil division denominator",
        });
    }
    if numerator == 0 {
        return Ok(0);
    }
    numerator
        .checked_add(denominator - 1)
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "ceil division numerator",
        })
        .map(|value| value / denominator)
}

pub(crate) fn usize_to_u64(
    value: usize,
    field: &'static str,
) -> Result<u64, CudaEGraphKernelPlanError> {
    CUDA_NUMERIC
        .usize_to_u64(value, field)
        .map_err(|_| CudaEGraphKernelPlanError::CountOverflow { field })
}
