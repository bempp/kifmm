use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::{BlasFieldTranslationSaRcmp, Evaluate, FftFieldTranslation, SingleNodeBuilder};

use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

fn main() {
    // Setup random sources and targets
    let n_sources = 1000;
    let n_targets = 2000;
    let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
    let targets = points_fixture::<f32>(n_targets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(150);
    let depth = None;
    let expansion_order = [5];
    let prune_empty = true;

    // FFT based M2L for a vector of charges
    {
        let nvecs = 1;
        let tmp = vec![1.0; n_sources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let mut fmm_fft = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                FftFieldTranslation::new(None),
            )
            .unwrap()
            .build()
            .unwrap();
        // fmm_fft.evaluate(false).unwrap();
    }
}