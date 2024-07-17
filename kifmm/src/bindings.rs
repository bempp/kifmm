#[no_mangle]
pub extern "C" fn add_from_rust(left: usize, right: usize) -> usize {
    left + right
}

#[no_mangle]
pub extern "C" fn hello_world() {
    println!("Hello world")
}

use green_kernels::laplace_3d::Laplace3dKernel;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

use crate::{fmm::KiFmm, tree::helpers::points_fixture, BlasFieldTranslationSaRcmp, SingleNodeBuilder};


/// All types
pub mod types {

    /// Type
    #[repr(C)]
    pub struct LaplaceBlas32;

    /// Type
    #[repr(C)]
    pub struct LaplaceBlas64;

    /// Type
    #[repr(C)]
    pub struct LaplaceFft32;

    /// Type
    #[repr(C)]
    pub struct LaplaceFft64;

    /// Type
    #[repr(C)]
    pub struct HelmholtzBlas32;

    /// Type
    #[repr(C)]
    pub struct HelmholtzBlas64;

    /// Type
    #[repr(C)]
    pub struct HelmholtzFft32;

    /// Type
    #[repr(C)]
    pub struct HelmholtzFft64;
}


/// All constructors
pub mod constructors {
    use super::*;


    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_blas_f32(
        _sources: *const f32,
        _nsources: usize,
        _targets: *const f32,
        _ntargets: usize,
        _charges: *const f32,
        _ncharges: usize
    ) -> *mut LaplaceBlas32 {


        // Vector of charges
        let nvecs = 1;
        let nsources = 1000;
        let ntargets = 2000;
        let sources = points_fixture::<f32>(nsources, None, None, Some(0));
        let targets = points_fixture::<f32>(ntargets, None, None, Some(1));
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f32));

        let singular_value_threshold = Some(1e-5);
        // FMM parameters
        let n_crit = Some(150);
        let depth = None;
        let expansion_order = [5];
        let prune_empty = true;

        let fmm = Box::new(SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                green_kernels::types::EvalType::Value,
                BlasFieldTranslationSaRcmp::new(
                    singular_value_threshold,
                    None,
                    crate::fmm::types::FmmSvdMode::Deterministic,
                ),
            )
            .unwrap()
            .build()
            .unwrap());

            println!("NLEAVES {:?}", fmm.tree.target_tree.leaves.len());

            Box::into_raw(fmm) as *mut LaplaceBlas32

    }
}


/// FMM API
pub mod api {
    use crate::Fmm;

    use super::*;

    pub fn evaluate_laplace_blas_f32(fmm: *mut LaplaceBlas32, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe { &mut *(fmm as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>) };
            fmm.evaluate(timed).unwrap();
        }
    }

    pub fn evaluate_laplace_blas_f64(fmm: *mut LaplaceBlas32, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe { &mut *(fmm as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>) };
            fmm.evaluate(timed).unwrap();
        }
    }


}


pub use constructors::*;
pub use api::*;
pub use types::*;
