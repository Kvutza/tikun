// Native In-Memory Silicon Topology Prober for Tikun JIT Engine

#[derive(Debug, Clone)]
pub struct SiliconTopology {
    pub chip_name: String,
    pub p_cores: usize,
    pub total_cores: usize,
    pub l1d_cache_kb: usize,
    pub l2_cache_kb: usize,
    pub recommended_tile_kb: usize,
    pub recommended_unroll: usize,
    pub recommended_prefetch: usize,
    pub recommended_workers: usize,
}

impl SiliconTopology {
    pub fn probe() -> Self {
        #[cfg(target_os = "macos")]
        {
            let p_cores = query_sysctl_u64("hw.perflevel0.logicalcpu")
                .or_else(|| query_sysctl_u64("hw.physicalcpu"))
                .unwrap_or(8) as usize;

            let total_cores = query_sysctl_u64("hw.ncpu").unwrap_or(8) as usize;
            let l1d_bytes = query_sysctl_u64("hw.l1dcachesize").unwrap_or(131072) as usize;
            let l2_bytes = query_sysctl_u64("hw.l2cachesize").unwrap_or(4194304) as usize;

            let l1d_kb = l1d_bytes / 1024;
            let l2_kb = l2_bytes / 1024;

            // Compute cache-optimal tile size for uninterrupted SIMD execution
            // Fits neatly into L2 cache cluster slice divided by active worker threads
            let workers = p_cores.max(1);
            let tile_kb = ((l2_kb / 2) / workers).clamp(64, 512);
            let unroll = 4; // Optimal for ARM NEON 4-wide execution pipelines
            let prefetch = 128; // 2 cache lines ahead (64B cache line size)

            Self {
                chip_name: "Apple Silicon (Probed In-Memory)".to_string(),
                p_cores,
                total_cores,
                l1d_cache_kb: l1d_kb,
                l2_cache_kb: l2_kb,
                recommended_tile_kb: tile_kb,
                recommended_unroll: unroll,
                recommended_prefetch: prefetch,
                recommended_workers: workers,
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            let cores = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(8);

            Self {
                chip_name: "Host CPU (Probed In-Memory)".to_string(),
                p_cores: cores,
                total_cores: cores,
                l1d_cache_kb: 32,
                l2_cache_kb: 512,
                recommended_tile_kb: 256,
                recommended_unroll: 4,
                recommended_prefetch: 64,
                recommended_workers: cores,
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn query_sysctl_u64(name: &str) -> Option<u64> {
    use std::ffi::CString;

    extern "C" {
        fn sysctlbyname(
            name: *const i8,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }

    let c_name = CString::new(name).ok()?;
    let mut val: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let ret = unsafe {
        sysctlbyname(
            c_name.as_ptr(),
            &mut val as *mut u64 as *mut std::ffi::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };

    if ret == 0 {
        Some(val)
    } else {
        None
    }
}
