use crate::arena::LayoutPlan;
use crate::ir::{PointwiseOp, SchedulePipeline};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Table};

pub struct HardwarePlan;

impl HardwarePlan {
    pub fn report(
        pipeline: &SchedulePipeline,
        layout: &LayoutPlan,
        op: PointwiseOp,
        tile_kb: usize,
        unroll: usize,
        prefetch: usize,
        workers: usize,
    ) -> String {
        let op_name = match op {
            PointwiseOp::AdamW { .. } => "AdamW",
            PointwiseOp::Lion { .. } => "Lion",
            PointwiseOp::SGD { .. } => "SGD",
            PointwiseOp::CustomDAG { .. } => "Custom DAG",
        };

        let total_elements: usize = layout.slots.iter().map(|s| s.num_elements).sum();
        let total_mb = layout.total_bytes as f64 / (1024.0 * 1024.0);

        let mut out = String::new();
        out.push_str("\n--- Hardware Execution Plan ---\n");

        let profile = crate::autotune::HardwareProfile::active();
        let profile_source = if profile.is_custom_tuned {
            "Tuned Profile (~/.cache/tikun/profile.json)"
        } else {
            "In-Memory Silicon Topology Prober"
        };

        let mut summary_table = Table::new();
        summary_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header(vec!["Property", "Value"])
            .add_row(vec!["Algorithm", op_name])
            .add_row(vec!["Profile Source", profile_source])
            .add_row(vec!["Active Memory", &format!("{:.2} MB ({} floats)", total_mb, total_elements)])
            .add_row(vec!["Cache Alignment", if layout.cache_aligned { "64-Byte Aligned" } else { "Unaligned" }])
            .add_row(vec!["L2 Cache Tile", &format!("{} KB", tile_kb)])
            .add_row(vec!["SIMD Unroll", &format!("{}x ({} floats/cycle)", unroll, unroll * 4)])
            .add_row(vec!["Prefetch Offset", &format!("{} Bytes", prefetch)])
            .add_row(vec!["Parallel Workers", &format!("{} Threads", workers)]);

        out.push_str(&summary_table.to_string());
        out.push_str("\n\nSchedule Nodes:\n");

        let mut nodes_table = Table::new();
        nodes_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header(vec!["Role", "Op Name", "Tile Size", "Unroll", "Width", "Prefetch"]);

        for node in &pipeline.nodes {
            nodes_table.add_row(vec![
                Cell::new(format!("{:?}", node.role)),
                Cell::new(&node.name),
                Cell::new(format!("{} B", node.tile_size)),
                Cell::new(format!("{}x", node.unroll_factor)),
                Cell::new(format!("{} lanes", node.vector_width)),
                Cell::new(format!("{} B", node.prefetch_distance)),
            ]);
        }

        out.push_str(&nodes_table.to_string());
        out.push_str("\n\nMemory Layout:\n");

        let mut arena_table = Table::new();
        arena_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header(vec!["Slot", "Offset", "Elements", "Size", "Alignment"]);

        for slot in layout.slots.iter().take(8) {
            arena_table.add_row(vec![
                Cell::new(format!("{:02}", slot.slot_id)),
                Cell::new(format!("0x{:08x}", slot.byte_offset)),
                Cell::new(format!("{}", slot.num_elements)),
                Cell::new(format!("{:.2} MB", (slot.num_elements * 4) as f64 / (1024.0 * 1024.0))),
                Cell::new(if slot.byte_offset % 64 == 0 { "64-B Aligned" } else { "Unaligned" }),
            ]);
        }

        out.push_str(&arena_table.to_string());
        if layout.slots.len() > 8 {
            out.push_str(&format!("\n... and {} additional parameter slots", layout.slots.len() - 8));
        }

        out.push('\n');
        out
    }
}
