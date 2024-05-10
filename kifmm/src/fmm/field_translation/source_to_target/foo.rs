
impl SourcetoTargetTranslationMetadata for KiFmmMetalLaplace
{
    fn source_to_target(&mut self) {

        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = ncoeffs_kifmm(self.expansion_order);
        let ncols = ncoeffs_kifmm(self.expansion_order);

        let mut se2tc_fat = rlst_dynamic_array2!(f32, [nrows, ncols * NTRANSFER_VECTORS_KIFMM]);
        let mut se2tc_thin = rlst_dynamic_array2!(f32, [nrows * NTRANSFER_VECTORS_KIFMM, ncols]);

        let alpha = f32::real(ALPHA_INNER);

        for (i, t) in self.source_to_target.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t.source.surface_grid(
                self.expansion_order,
                self.tree.source_tree().domain(),
                alpha,
            );
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface = t.target.surface_grid(
                self.expansion_order,
                self.tree.source_tree().domain(),
                alpha,
            );
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();

            let mut tmp_gram_t = rlst_dynamic_array2!(f32, [ntargets, nsources]);

            self.kernel.assemble_st(
                EvalType::Value,
                &source_equivalent_surface[..],
                &target_check_surface[..],
                tmp_gram_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut tmp_gram = rlst_dynamic_array2!(f32, [nsources, ntargets]);
            tmp_gram.fill_from(tmp_gram_t.transpose());

            let mut block = se2tc_fat
                .view_mut()
                .into_subview([0, i * ncols], [nrows, ncols]);
            block.fill_from(tmp_gram.view());

            let mut block_column = se2tc_thin
                .view_mut()
                .into_subview([i * nrows, 0], [nrows, ncols]);
            block_column.fill_from(tmp_gram.view());
        }

        let mu = se2tc_fat.shape()[0];
        let nvt = se2tc_fat.shape()[1];
        let k = std::cmp::min(mu, nvt);

        let mut u_big = rlst_dynamic_array2!(f32, [mu, k]);
        let mut sigma = vec![0f32; k];
        let mut vt_big = rlst_dynamic_array2!(f32, [k, nvt]);

        se2tc_fat
            .into_svd_alloc(
                u_big.view_mut(),
                vt_big.view_mut(),
                &mut sigma[..],
                SvdMode::Reduced,
            )
            .unwrap();
        let cutoff_rank = find_cutoff_rank(&sigma, self.source_to_target.threshold);
        let mut u = rlst_dynamic_array2!(f32, [mu, cutoff_rank]);
        let mut sigma_mat = rlst_dynamic_array2!(f32, [cutoff_rank, cutoff_rank]);
        let mut vt = rlst_dynamic_array2!(f32, [cutoff_rank, nvt]);

        u.fill_from(u_big.into_subview([0, 0], [mu, cutoff_rank]));
        vt.fill_from(vt_big.into_subview([0, 0], [cutoff_rank, nvt]));
        for (j, s) in sigma.iter().enumerate().take(cutoff_rank) {
            unsafe {
                *sigma_mat.get_unchecked_mut([j, j]) = *s as f32
            }
        }

        // Store compressed M2L operators
        let thin_nrows = se2tc_thin.shape()[0];
        let nst = se2tc_thin.shape()[1];
        let k = std::cmp::min(thin_nrows, nst);
        let mut _gamma = rlst_dynamic_array2!(f32, [thin_nrows, k]);
        let mut _r = vec![f32::zero().re(); k];
        let mut st = rlst_dynamic_array2!(f32, [k, nst]);

        se2tc_thin
            .into_svd_alloc(
                _gamma.view_mut(),
                st.view_mut(),
                &mut _r[..],
                SvdMode::Reduced,
            )
            .unwrap();

        let mut s_trunc = rlst_dynamic_array2!(f32, [nst, cutoff_rank]);
        for j in 0..cutoff_rank {
            for i in 0..nst {
                unsafe { *s_trunc.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
            }
        }



    }
}