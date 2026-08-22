use serde::{Deserialize, Serialize};
use crate::arena::LayoutPlan;
use crate::ir::{PointwiseOp, SchedulePipeline};

/// Standardized Tikun Hardware IR Specification (v1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationSchema {
    pub schema_version: String,
    pub target_architecture: String,
    pub optimizer: OptimizerSpec,
    pub memory_topography: MemoryTopographySpec,
    pub execution_schedule: ExecutionScheduleSpec,
    pub performance_metrics: PerformanceMetricsSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerSpec {
    pub algorithm: String,
    pub precision: String,
    pub hyperparameters: HyperparameterSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpec {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: Option<f32>,
    pub epsilon: Option<f32>,
    pub weight_decay: f32,
    pub max_norm: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTopographySpec {
    pub total_parameters: usize,
    pub total_allocated_bytes: usize,
    pub cache_line_size_bytes: usize,
    pub is_cache_aligned: bool,
    pub slots: Vec<SlotDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotDescriptor {
    pub slot_id: usize,
    pub byte_offset: usize,
    pub num_elements: usize,
    pub byte_size: usize,
    pub cache_aligned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionScheduleSpec {
    pub l2_tile_size_elements: usize,
    pub simd_vector_lanes: usize,
    pub simd_unroll_factor: usize,
    pub prefetch_strategy: String,
    pub passes: Vec<SchedulePassSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulePassSpec {
    pub pass_id: usize,
    pub pass_name: String,
    pub pass_role: String,
    pub target_hardware_unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsSpec {
    pub bytes_read_per_step: usize,
    pub bytes_written_per_step: usize,
    pub total_memory_traffic_bytes: usize,
    pub arithmetic_intensity_flop_per_byte: f64,
}

impl CompilationSchema {
    /// Constructs the formal compilation schema from pipeline and layout plans
    pub fn build(pipeline: &SchedulePipeline, layout: &LayoutPlan, op: PointwiseOp) -> Self {
        let (alg, hyp) = match op {
            PointwiseOp::AdamW { learn_rate, beta_one, beta_two, eps, decay, .. } => (
                "AdamW".to_string(),
                HyperparameterSpec {
                    learning_rate: learn_rate,
                    beta1: beta_one,
                    beta2: Some(beta_two),
                    epsilon: Some(eps),
                    weight_decay: decay,
                    max_norm: pipeline.reduction.map(|_| 1.0),
                },
            ),
            PointwiseOp::Lion { learn_rate, beta_one, beta_two, decay } => (
                "Lion".to_string(),
                HyperparameterSpec {
                    learning_rate: learn_rate,
                    beta1: beta_one,
                    beta2: Some(beta_two),
                    epsilon: None,
                    weight_decay: decay,
                    max_norm: pipeline.reduction.map(|_| 1.0),
                },
            ),
            PointwiseOp::SGD { learn_rate, momentum, decay } => (
                "SGD".to_string(),
                HyperparameterSpec {
                    learning_rate: learn_rate,
                    beta1: momentum,
                    beta2: None,
                    epsilon: None,
                    weight_decay: decay,
                    max_norm: pipeline.reduction.map(|_| 1.0),
                },
            ),
            PointwiseOp::CustomDAG { .. } => (
                "CustomDAG".to_string(),
                HyperparameterSpec {
                    learning_rate: 0.0,
                    beta1: 0.0,
                    beta2: None,
                    epsilon: None,
                    weight_decay: 0.0,
                    max_norm: None,
                },
            ),
        };

        let total_elements: usize = layout.slots.iter().map(|s| s.num_elements).sum();
        let bytes_read = total_elements * 16; // P + G + M1 + M2
        let bytes_written = total_elements * 12; // P + M1 + M2
        let total_traffic = bytes_read + bytes_written;

        let slots = layout
            .slots
            .iter()
            .map(|s| SlotDescriptor {
                slot_id: s.slot_id,
                byte_offset: s.byte_offset,
                num_elements: s.num_elements,
                byte_size: s.byte_size,
                cache_aligned: s.byte_offset % 64 == 0,
            })
            .collect();

        let passes = pipeline
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| SchedulePassSpec {
                pass_id: i,
                pass_name: n.name.clone(),
                pass_role: format!("{:?}", n.role),
                target_hardware_unit: "ARM64 NEON Vector Execution Pipe".to_string(),
            })
            .collect();

        Self {
            schema_version: "tikun.ir.v1.0".to_string(),
            target_architecture: "Apple Silicon ARM64 (NEON + AMX Co-Design)".to_string(),
            optimizer: OptimizerSpec {
                algorithm: alg,
                precision: "float32 (in-register 4x SIMD)".to_string(),
                hyperparameters: hyp,
            },
            memory_topography: MemoryTopographySpec {
                total_parameters: total_elements,
                total_allocated_bytes: layout.total_bytes,
                cache_line_size_bytes: 64,
                is_cache_aligned: layout.cache_aligned,
                slots,
            },
            execution_schedule: ExecutionScheduleSpec {
                l2_tile_size_elements: 131072, // 512 KB
                simd_vector_lanes: 16,
                simd_unroll_factor: 4,
                prefetch_strategy: "prfm pldl1keep (64B & 128B dual-line prefetch)".to_string(),
                passes,
            },
            performance_metrics: PerformanceMetricsSpec {
                bytes_read_per_step: bytes_read,
                bytes_written_per_step: bytes_written,
                total_memory_traffic_bytes: total_traffic,
                arithmetic_intensity_flop_per_byte: 14.0 / 28.0, // ~0.50 FLOP/Byte
            },
        }
    }

    /// Emits formatted JSON
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }
}
