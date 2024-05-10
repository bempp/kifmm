use green_kernels::{traits::Kernel, types::EvalType};
use rlst::{empty_array, rlst_dynamic_array2, rlst_metal_array2, MetalDevice, MultIntoResize, RawAccess, RawAccessMut, Shape};

use crate::{fmm::{pinv::pinv, types::KiFmmMetalLaplace}, traits::{field::SourceAndTargetTranslationMetadata, tree::FmmTreeNode}, tree::{constants::{ALPHA_INNER, ALPHA_OUTER}, types::MortonKey}};


impl SourceAndTargetTranslationMetadata for KiFmmMetalLaplace {

    fn source(&mut self) {

        let device = MetalDevice::from_default();
        let root = MortonKey::<f32>::root();

        // Cast surface parameters
        let alpha_outer = ALPHA_OUTER as f32;
        let alpha_inner = ALPHA_INNER as f32;
        let domain = self.tree.domain;

        // Compute required surfaces
        let upward_equivalent_surface =
            root.surface_grid(self.expansion_order, &domain, alpha_inner);
        let upward_check_surface = root.surface_grid(self.expansion_order, &domain, alpha_outer);

        let nequiv_surface = upward_equivalent_surface.len() / self.dim;
        let ncheck_surface = upward_check_surface.len() / self.dim;

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using SVD
        let mut uc2e_t = rlst_dynamic_array2!(f32, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &upward_equivalent_surface[..],
            &upward_check_surface[..],
            uc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut uc2e = rlst_dynamic_array2!(f32, [nequiv_surface, ncheck_surface]);
        uc2e.fill_from(uc2e_t.transpose());

        let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(f32, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i] as f32;
        }

        let uc2e_inv_1 =
            empty_array::<f32, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let uc2e_inv_2 = ut;

        // Store in row major order
        let mut uc2e_inv_1_metal = rlst_metal_array2!(&device, f32, uc2e_inv_1.shape());
        let mut uc2e_inv_2_metal = rlst_metal_array2!(&device, f32, uc2e_inv_2.shape());

        let [m, n] = uc2e_inv_1.shape();
        for col_idx in 0..n {
            for row_idx in 0..m {
                let col_major_idx = col_idx * m + row_idx;
                let row_major_idx = row_idx * n + col_idx;
                uc2e_inv_1_metal.data_mut()[row_major_idx] = uc2e_inv_1.data()[col_major_idx];
            }
        }

        let [m, n] = uc2e_inv_2.shape();
        for col_idx in 0..n {
            for row_idx in 0..m {
                let col_major_idx = col_idx * m + row_idx;
                let row_major_idx = row_idx * n + col_idx;
                uc2e_inv_2_metal.data_mut()[row_major_idx] = uc2e_inv_2.data()[col_major_idx];
            }
        }

        // Calculate M2M operator matrices
        let children = root.children();
        let mut m2m = rlst_dynamic_array2!(
            f32,
            [nequiv_surface, 8 * nequiv_surface]
        );

        let mut m2m_metal = rlst_metal_array2!(&device, f32, [nequiv_surface, 8*nequiv_surface]);

        let mut m2m_vec = Vec::new();
        let mut m2m_vec_metal = Vec::new();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.surface_grid(self.expansion_order, &domain, alpha_inner);

            let mut pc2ce_t = rlst_dynamic_array2!(f32, [ncheck_surface, nequiv_surface]);

            self.kernel.assemble_st(
                EvalType::Value,
                &child_upward_equivalent_surface,
                &upward_check_surface,
                pc2ce_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut pc2ce = rlst_dynamic_array2!(f32, [nequiv_surface, ncheck_surface]);
            pc2ce.fill_from(pc2ce_t.transpose());

            let tmp = empty_array::<f32, 2>().simple_mult_into_resize(
                uc2e_inv_1.view(),
                empty_array::<f32, 2>()
                    .simple_mult_into_resize(uc2e_inv_2.view(), pc2ce.view()),
            );
            let l = i * nequiv_surface * nequiv_surface;
            let r = l + nequiv_surface * nequiv_surface;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());

            // Store in metal arrays
            let mut tmp_metal = rlst_metal_array2!(&device, f32, tmp.shape());
            let [m, n] = tmp.shape();
            for col_idx in 0..n {
                for row_idx in 0..m {
                    let col_major_idx = col_idx * m + row_idx;
                    let row_major_idx = row_idx * n + col_idx;
                    tmp_metal.data_mut()[row_major_idx] = tmp.data()[col_major_idx];
                }
            }

            m2m_vec.push(tmp);
            m2m_vec_metal.push(tmp_metal);
        }

        // Store in metal arrays
        let [m, n] = m2m.shape();
        for col_idx in 0..n {
            for row_idx in 0..m {
                let col_major_idx = col_idx * m + row_idx;
                let row_major_idx = row_idx * n + col_idx;
                m2m_metal.data_mut()[row_major_idx] = m2m.data()[col_major_idx];
            }
        }

        self.source = vec![m2m_metal];
        self.source_vec = vec![m2m_vec_metal];
        self.uc2e_inv_1 = vec![uc2e_inv_1_metal];
        self.uc2e_inv_2 = vec![uc2e_inv_2_metal];


    }

    fn target(&mut self) {

    }
}