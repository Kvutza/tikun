/// Physical Memory Buffer Descriptor for Zero-Copy Optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferSlot {
    pub slot_id: usize,
    pub byte_offset: usize,
    pub num_elements: usize,
    pub byte_size: usize,
    pub is_persistent: bool,
}

impl BufferSlot {
    pub fn new(slot_id: usize, byte_offset: usize, num_elements: usize, is_persistent: bool) -> Self {
        let byte_size = num_elements * std::mem::size_of::<f32>();
        Self {
            slot_id,
            byte_offset,
            num_elements,
            byte_size,
            is_persistent,
        }
    }
}

/// Static Memory Layout Plan (Whole-Model Memory Planning)
#[derive(Debug, Clone, Default)]
pub struct LayoutPlan {
    pub slots: Vec<BufferSlot>,
    pub total_bytes: usize,
    pub cache_aligned: bool,
}

impl LayoutPlan {
    /// Builds a cache-aligned static memory plan for a given list of tensor dimensions
    pub fn from_shapes(shapes: &[usize]) -> Self {
        let mut slots = Vec::with_capacity(shapes.len());
        let mut current_offset = 0;
        let cache_line = 64; // Standard 64-byte L1/L2 cache line alignment

        for (idx, &num_elements) in shapes.iter().enumerate() {
            let byte_size = num_elements * std::mem::size_of::<f32>();
            
            // Align each buffer offset to 64-byte cache boundary
            let remainder = current_offset % cache_line;
            if remainder != 0 {
                current_offset += cache_line - remainder;
            }

            slots.push(BufferSlot::new(idx, current_offset, num_elements, true));
            current_offset += byte_size;
        }

        Self {
            slots,
            total_bytes: current_offset,
            cache_aligned: true,
        }
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn total_allocated_bytes(&self) -> usize {
        self.total_bytes
    }
}

/// Resident Contiguous Memory Arena
pub struct MemoryArena {
    pub raw_storage: Vec<f32>,
    pub layout_plan: LayoutPlan,
}

impl MemoryArena {
    pub fn allocate(layout_plan: LayoutPlan) -> Self {
        let float_count = (layout_plan.total_bytes + 3) / 4;
        let raw_storage = vec![0.0f32; float_count];
        Self {
            raw_storage,
            layout_plan,
        }
    }

    pub fn base_ptr(&self) -> *const f32 {
        self.raw_storage.as_ptr()
    }

    pub fn base_mut_ptr(&mut self) -> *mut f32 {
        self.raw_storage.as_mut_ptr()
    }

    pub fn get_slice(&self, slot_id: usize) -> Option<&[f32]> {
        let slot = self.layout_plan.slots.get(slot_id)?;
        let float_offset = slot.byte_offset / 4;
        Some(&self.raw_storage[float_offset..float_offset + slot.num_elements])
    }

    pub fn get_mut_slice(&mut self, slot_id: usize) -> Option<&mut [f32]> {
        let slot = self.layout_plan.slots.get(slot_id)?;
        let float_offset = slot.byte_offset / 4;
        Some(&mut self.raw_storage[float_offset..float_offset + slot.num_elements])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_alignment() {
        let shapes = vec![100, 513, 2048, 17];
        let plan = LayoutPlan::from_shapes(&shapes);
        assert_eq!(plan.slot_count(), 4);

        for slot in &plan.slots {
            assert_eq!(slot.byte_offset % 64, 0, "Slot must be 64-byte cache line aligned");
        }
    }

    #[test]
    fn test_arena_mutation() {
        let shapes = vec![128, 256];
        let plan = LayoutPlan::from_shapes(&shapes);
        let mut arena = MemoryArena::allocate(plan);

        {
            let slice = arena.get_mut_slice(0).unwrap();
            slice[0] = 42.0;
        }

        let slice = arena.get_slice(0).unwrap();
        assert_eq!(slice[0], 42.0);
    }
}
