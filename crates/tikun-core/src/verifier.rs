use crate::ir::SchedulePipeline;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierViolation {
    MisalignedCacheLine { node_name: String, tile_size: usize },
    InvalidVectorWidth { node_name: String, width: usize },
    EmptyPipeline,
}

pub struct Verifier;

impl Verifier {
    /// Static analysis pass: verifies schedule alignment and hardware safety in <1ms
    pub fn verify(pipeline: &SchedulePipeline) -> Result<(), Vec<VerifierViolation>> {
        let mut errors = Vec::new();

        if pipeline.nodes.is_empty() {
            errors.push(VerifierViolation::EmptyPipeline);
            return Err(errors);
        }

        for node in &pipeline.nodes {
            // Verify 64-byte L1 cache line alignment constraint
            if node.tile_size % 64 != 0 {
                errors.push(VerifierViolation::MisalignedCacheLine {
                    node_name: node.name.clone(),
                    tile_size: node.tile_size,
                });
            }

            // Verify vector width is a power of 2 (4, 8, 16, 32, 64)
            if !node.vector_width.is_power_of_two() || node.vector_width < 4 {
                errors.push(VerifierViolation::InvalidVectorWidth {
                    node_name: node.name.clone(),
                    width: node.vector_width,
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
