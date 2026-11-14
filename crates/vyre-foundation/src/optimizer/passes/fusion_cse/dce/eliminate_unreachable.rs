use super::{const_loop_empty, const_truth};
use crate::ir::Node;

#[cfg(test)]
#[inline]
pub(crate) fn eliminate_unreachable(nodes: Vec<Node>) -> Vec<Node> {
    eliminate_unreachable_tracked(nodes).0
}

#[inline]
pub(crate) fn eliminate_unreachable_tracked(nodes: Vec<Node>) -> (Vec<Node>, bool) {
    let total_len = nodes.len();
    let mut changed = false;
    let mut out = Vec::with_capacity(total_len);
    for (idx, node) in nodes.into_iter().enumerate() {
        match node {
            Node::If {
                cond,
                then,
                otherwise,
            } => match const_truth(&cond) {
                Some(true) => {
                    changed = true;
                    out.extend(eliminate_unreachable_tracked(then).0);
                }
                Some(false) => {
                    changed = true;
                    out.extend(eliminate_unreachable_tracked(otherwise).0);
                }
                None => {
                    let (then, c1) = eliminate_unreachable_tracked(then);
                    let (otherwise, c2) = eliminate_unreachable_tracked(otherwise);
                    changed |= c1 || c2;
                    out.push(Node::If {
                        cond,
                        then,
                        otherwise,
                    });
                }
            },
            Node::Loop {
                var: _,
                from,
                to,
                body: _,
            } if const_loop_empty(&from, &to) => {
                changed = true;
            }
            Node::Loop {
                var,
                from,
                to,
                body,
            } => {
                let (body, c) = eliminate_unreachable_tracked(body);
                changed |= c;
                out.push(Node::Loop {
                    var,
                    from,
                    to,
                    body,
                });
            }
            Node::Block(block_nodes) => {
                let (block_nodes, c) = eliminate_unreachable_tracked(block_nodes);
                if block_nodes.is_empty() {
                    changed = true;
                } else {
                    changed |= c;
                    out.push(Node::Block(block_nodes));
                }
            }
            Node::Return => {
                if idx + 1 < total_len {
                    changed = true;
                }
                out.push(Node::Return);
                break;
            }
            Node::Region {
                generator,
                source_region,
                body,
            } => {
                let body_nodes =
                    std::sync::Arc::try_unwrap(body).unwrap_or_else(|arc| (*arc).clone());
                let (new_body, c) = eliminate_unreachable_tracked(body_nodes);
                changed |= c;
                out.push(Node::Region {
                    generator,
                    source_region,
                    body: std::sync::Arc::new(new_body),
                });
            }
            Node::TileElementwise {
                out: name,
                inputs,
                body,
            } => {
                let (body, c) = eliminate_unreachable_tracked(body);
                changed |= c;
                out.push(Node::TileElementwise {
                    out: name,
                    inputs,
                    body,
                });
            }
            other => out.push(other),
        }
    }
    (out, changed)
}
