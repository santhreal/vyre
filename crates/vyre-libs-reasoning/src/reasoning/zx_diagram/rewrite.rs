//! ZX-diagram rewrite rules: spider fusion, identity removal, color change,
//! and the simplification chain that runs the first two to fixpoint.

#[cfg(test)]
pub(crate) use vyre_reference::composition_witness::{ZxColor, ZxDiagram, ZxSpider};

/// Apply spider fusion (S1): merge every adjacent pair of same-color spiders.
#[must_use]
#[cfg(test)]
pub(crate) fn spider_fusion(diagram: ZxDiagram) -> ZxDiagram {
    vyre_reference::composition_witness::zx_spider_fusion_witness(diagram)
}

/// Apply identity removal (S2): drop phase-0 spiders of degree 2 between same-color neighbors.
#[must_use]
#[cfg(test)]
pub(crate) fn identity_removal(diagram: ZxDiagram) -> ZxDiagram {
    vyre_reference::composition_witness::zx_identity_removal_witness(diagram)
}

/// Apply color-change (H) at vertex `v`: flip the color, leave the phase.
#[cfg(test)]
pub(crate) fn color_change(diagram: &mut ZxDiagram, v: u32) {
    vyre_reference::composition_witness::zx_color_change_witness(diagram, v);
}

/// The joint fixpoint of spider fusion and identity removal.
#[must_use]
#[cfg(test)]
pub(crate) fn simplified_diagram(diagram: ZxDiagram) -> ZxDiagram {
    vyre_reference::composition_witness::zx_simplified_diagram_witness(diagram)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(phase: u32) -> ZxSpider {
        ZxSpider {
            color: ZxColor::Z,
            phase_num: phase,
        }
    }
    fn x(phase: u32) -> ZxSpider {
        ZxSpider {
            color: ZxColor::X,
            phase_num: phase,
        }
    }

    #[test]
    fn fusion_merges_two_z_spiders() {
        let d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(1), z(3)],
            edges: vec![(0, 1)],
        };
        let out = spider_fusion(d);
        assert_eq!(out.spiders.len(), 1);
        assert_eq!(out.spiders[0].phase_num, 4);
        assert!(out.edges.is_empty());
    }

    #[test]
    fn fusion_leaves_cross_color_edge_intact() {
        let d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(1), x(3)],
            edges: vec![(0, 1)],
        };
        let out = spider_fusion(d.clone());
        assert_eq!(out, d);
    }

    #[test]
    fn fusion_wraps_phase_modulo_denom() {
        let d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(5), z(5)],
            edges: vec![(0, 1)],
        };
        let out = spider_fusion(d);
        assert_eq!(out.spiders[0].phase_num, 2);
    }

    #[test]
    fn fusion_chain_of_three() {
        let d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(1), z(2), z(3)],
            edges: vec![(0, 1), (1, 2)],
        };
        let out = spider_fusion(d);
        assert_eq!(out.spiders.len(), 1);
        assert_eq!(out.spiders[0].phase_num, 6);
    }

    #[test]
    fn identity_removal_splices_wire() {
        let d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(1), z(0), z(2)],
            edges: vec![(0, 1), (1, 2)],
        };
        let out = identity_removal(d);
        assert_eq!(out.spiders.len(), 2);
        assert_eq!(out.edges, vec![(0, 1)]);
    }

    #[test]
    fn identity_removal_ignores_non_zero_phase() {
        let d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(1), z(1), z(2)],
            edges: vec![(0, 1), (1, 2)],
        };
        let out = identity_removal(d.clone());
        assert_eq!(out, d);
    }

    #[test]
    fn color_change_flips_color_preserving_phase() {
        let mut d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(3)],
            edges: vec![],
        };
        color_change(&mut d, 0);
        assert_eq!(d.spiders[0].color, ZxColor::X);
        assert_eq!(d.spiders[0].phase_num, 3);
        color_change(&mut d, 0);
        assert_eq!(d.spiders[0].color, ZxColor::Z);
    }

    #[test]
    fn simplified_diagram_combines_fusion_and_removal() {
        let d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(1), z(0), z(2)],
            edges: vec![(0, 1), (1, 2)],
        };
        let out = simplified_diagram(d);
        assert_eq!(out.spiders.len(), 1);
        assert_eq!(out.spiders[0].phase_num, 3);
        assert!(out.edges.is_empty());
    }

    #[test]
    fn identity_removal_preserves_self_loops() {
        let d = ZxDiagram {
            phase_denom: 8,
            spiders: vec![z(0)],
            edges: vec![(0, 0), (0, 0)],
        };
        let out = identity_removal(d.clone());
        assert_eq!(out, d);
    }
}
