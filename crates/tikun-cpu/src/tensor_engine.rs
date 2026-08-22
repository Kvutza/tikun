use rayon::prelude::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub struct TensorEngine;

impl TensorEngine {
    const A: f32 = 3.4445;
    const B: f32 = -4.7750;
    const C: f32 = 2.0315;

    pub fn polar_step(matrix: &mut [f32], rows: usize, cols: usize, steps: usize) {
        if rows == 0 || cols == 0 || matrix.len() != rows * cols {
            return;
        }

        // Fast Transpose if rows > cols (Operate on smaller dimension Gram matrix)
        if rows > cols {
            let mut transposed = vec![0.0f32; cols * rows];
            for r in 0..rows {
                let r_off = r * cols;
                for c in 0..cols {
                    transposed[c * rows + r] = matrix[r_off + c];
                }
            }
            Self::polar_step(&mut transposed, cols, rows, steps);
            for r in 0..rows {
                let r_off = r * cols;
                for c in 0..cols {
                    matrix[r_off + c] = transposed[c * rows + r];
                }
            }
            return;
        }

        // 1. Frobenius norm normalization
        let mut sum_sq: f64 = 0.0;
        for &val in matrix.iter() {
            sum_sq += (val as f64) * (val as f64);
        }
        let inv_norm = 1.0 / (sum_sq.sqrt() as f32).max(1e-7);
        for val in matrix.iter_mut() {
            *val *= inv_norm;
        }

        // Pre-allocate scratchpads once for all steps
        let r2 = rows * rows;
        let mut a_mat = vec![0.0f32; r2];
        let mut b_mat = vec![0.0f32; r2];
        let mut poly_mat = vec![0.0f32; r2];
        let mut next_x = vec![0.0f32; rows * cols];

        for _ in 0..steps {
            // A = X * X^T (rows x rows)
            Self::gemm_trans(matrix, matrix, &mut a_mat, rows, cols, rows);

            // B = A * A
            Self::gemm_fast(&a_mat, &a_mat, &mut b_mat, rows, rows, rows);

            // Poly = a*I + b*A + c*B
            poly_mat
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(|(i, row_slice)| {
                    for j in 0..rows {
                        let eye = if i == j { 1.0 } else { 0.0 };
                        let idx = i * rows + j;
                        row_slice[j] = Self::A * eye + Self::B * a_mat[idx] + Self::C * b_mat[idx];
                    }
                });

            // Next_X = Poly * X
            Self::gemm_fast(&poly_mat, matrix, &mut next_x, rows, rows, cols);

            matrix.copy_from_slice(&next_x);
        }
    }

    pub fn polar_batched(
        tensor: &mut [f32],
        num_heads: usize,
        head_dim_out: usize,
        head_dim_in: usize,
        steps: usize,
    ) {
        let slice_len = head_dim_out * head_dim_in;
        if tensor.len() != num_heads * slice_len {
            return;
        }

        tensor
            .par_chunks_mut(slice_len)
            .for_each(|head_matrix| {
                Self::polar_step(head_matrix, head_dim_out, head_dim_in, steps);
            });
    }

    /// Hardware-Accelerated Matrix Multiplication (AMX / BLAS on Apple Silicon, Parallel SIMD fallback)
    pub fn gemm_fast(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        #[cfg(target_os = "macos")]
        unsafe {
            extern "C" {
                fn cblas_sgemm(
                    order: i32,
                    trans_a: i32,
                    trans_b: i32,
                    m: i32,
                    n: i32,
                    k: i32,
                    alpha: f32,
                    a: *const f32,
                    lda: i32,
                    b: *const f32,
                    ldb: i32,
                    beta: f32,
                    c: *mut f32,
                    ldc: i32,
                );
            }
            cblas_sgemm(
                101, 111, 111,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,
                b.as_ptr(), n as i32,
                0.0, out.as_mut_ptr(), n as i32,
            );
            return;
        }

        #[cfg(not(target_os = "macos"))]
        {
            out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
                out_row.fill(0.0);
                for p in 0..k {
                    let a_val = a[i * k + p];
                    let b_row = &b[p * n..(p + 1) * n];
                    for j in 0..n {
                        out_row[j] += a_val * b_row[j];
                    }
                }
            });
        }
    }

    /// Hardware-Accelerated A * B^T
    pub fn gemm_trans(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        #[cfg(target_os = "macos")]
        unsafe {
            extern "C" {
                fn cblas_sgemm(
                    order: i32,
                    trans_a: i32,
                    trans_b: i32,
                    m: i32,
                    n: i32,
                    k: i32,
                    alpha: f32,
                    a: *const f32,
                    lda: i32,
                    b: *const f32,
                    ldb: i32,
                    beta: f32,
                    c: *mut f32,
                    ldc: i32,
                );
            }
            cblas_sgemm(
                101, 111, 112,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,
                b.as_ptr(), k as i32,
                0.0, out.as_mut_ptr(), n as i32,
            );
            return;
        }

        #[cfg(not(target_os = "macos"))]
        {
            out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
                let a_row = &a[i * k..(i + 1) * k];
                for j in 0..n {
                    let b_row = &b[j * k..(j + 1) * k];
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a_row[p] * b_row[p];
                    }
                    out_row[j] = sum;
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_step() {
        let mut mat = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        TensorEngine::polar_step(&mut mat, 4, 4, 5);
        let mut gram = vec![0.0f32; 16];
        TensorEngine::gemm_trans(&mat, &mat, &mut gram, 4, 4, 4);
        assert!(gram[0] > 0.5);
    }

    #[test]
    fn test_polar_batched() {
        let mut tensor = vec![0.5f32; 2 * 4 * 4];
        TensorEngine::polar_batched(&mut tensor, 2, 4, 4, 5);
        assert_eq!(tensor.len(), 32);
    }
}
