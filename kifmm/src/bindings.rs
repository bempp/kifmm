use green_kernels::laplace_3d::Laplace3dKernel;
use rlst::{c32, c64, rlst_dynamic_array2, RawAccess, RawAccessMut};

use crate::{
    fmm::KiFmm, tree::helpers::points_fixture, BlasFieldTranslationSaRcmp, SingleNodeBuilder,
};

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
    use green_kernels::helmholtz_3d::Helmholtz3dKernel;
    use rlst::rlst_array_from_slice1;

    use crate::{BlasFieldTranslationIa, FftFieldTranslation};

    use super::*;

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_blas_f32(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const f32,
        nsources: usize,
        targets: *const f32,
        ntargets: usize,
        charges: *const f32,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f32,
        svd_mode: bool,
        rsvd_ncomponents: usize,
        rsvd_noversamples: usize,
        rsvd_random_state: usize,
    ) -> *mut LaplaceBlas32 {
        let sources = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(sources, nsources), [nsources])
        };
        let targets = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(targets, ntargets), [ntargets])
        };
        let charges = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(charges, ncharges), [ncharges])
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            None,
            crate::fmm::types::FmmSvdMode::Deterministic,
        );

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    expansion_order,
                    Laplace3dKernel::new(),
                    green_kernels::types::EvalType::Value,
                    field_translation,
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        Box::into_raw(fmm) as *mut LaplaceBlas32
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_blas_f64(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const f64,
        nsources: usize,
        targets: *const f64,
        ntargets: usize,
        charges: *const f64,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f64,
        svd_mode: bool,
        rsvd_ncomponents: usize,
        rsvd_noversamples: usize,
        rsvd_random_state: usize,
    ) -> *mut LaplaceBlas64 {
        let sources = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(sources, nsources), [nsources])
        };
        let targets = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(targets, ntargets), [ntargets])
        };
        let charges = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(charges, ncharges), [ncharges])
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            None,
            crate::fmm::types::FmmSvdMode::Deterministic,
        );

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    expansion_order,
                    Laplace3dKernel::new(),
                    green_kernels::types::EvalType::Value,
                    field_translation,
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        Box::into_raw(fmm) as *mut LaplaceBlas64
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_fft_f32(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const f32,
        nsources: usize,
        targets: *const f32,
        ntargets: usize,
        charges: *const f32,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut LaplaceFft32 {
        let sources = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(sources, nsources), [nsources])
        };
        let targets = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(targets, ntargets), [ntargets])
        };
        let charges = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(charges, ncharges), [ncharges])
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    expansion_order,
                    Laplace3dKernel::new(),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        Box::into_raw(fmm) as *mut LaplaceFft32
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_fft_f64(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const f64,
        nsources: usize,
        targets: *const f64,
        ntargets: usize,
        charges: *const f64,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut LaplaceFft64 {
        let sources = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(sources, nsources), [nsources])
        };
        let targets = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(targets, ntargets), [ntargets])
        };
        let charges = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(charges, ncharges), [ncharges])
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    expansion_order,
                    Laplace3dKernel::new(),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        Box::into_raw(fmm) as *mut LaplaceFft64
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn helmholtz_blas_f32(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const f32,
        nsources: usize,
        targets: *const f32,
        ntargets: usize,
        charges: *const f32,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        wavenumber: f32,
        singular_value_threshold: f32,
        _svd_mode: bool,
        _rsvd_ncomponents: usize,
        _rsvd_noversamples: usize,
        _rsvd_random_state: usize,
    ) -> *mut HelmholtzBlas32 {
        let sources = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(sources, nsources), [nsources])
        };
        let targets = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(targets, ntargets), [ntargets])
        };
        let charges = unsafe {
            rlst_array_from_slice1!(
                std::slice::from_raw_parts(charges as *const c32, ncharges),
                [ncharges]
            )
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);

        let field_translation = BlasFieldTranslationIa::new(singular_value_threshold, None);

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    green_kernels::types::EvalType::Value,
                    field_translation,
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        Box::into_raw(fmm) as *mut HelmholtzBlas32
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn helmholtz_blas_f64(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const f64,
        nsources: usize,
        targets: *const f64,
        ntargets: usize,
        charges: *const f64,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        wavenumber: f64,
        singular_value_threshold: f64,
        _svd_mode: bool,
        _rsvd_ncomponents: usize,
        _rsvd_noversamples: usize,
        _rsvd_random_state: usize,
    ) -> *mut HelmholtzBlas64 {
        let sources = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(sources, nsources), [nsources])
        };
        let targets = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(targets, ntargets), [ntargets])
        };
        let charges = unsafe {
            rlst_array_from_slice1!(
                std::slice::from_raw_parts(charges as *const c64, ncharges),
                [ncharges]
            )
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);

        let field_translation = BlasFieldTranslationIa::new(singular_value_threshold, None);

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    green_kernels::types::EvalType::Value,
                    field_translation,
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        Box::into_raw(fmm) as *mut HelmholtzBlas64
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn helmholtz_fft_f32(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const f32,
        nsources: usize,
        targets: *const f32,
        ntargets: usize,
        charges: *const f32,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        wavenumber: f32,
        block_size: usize,
    ) -> *mut HelmholtzFft32 {
        let sources = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(sources, nsources), [nsources])
        };
        let targets = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(targets, ntargets), [ntargets])
        };
        let charges = unsafe {
            rlst_array_from_slice1!(
                std::slice::from_raw_parts(charges as *const c32, ncharges),
                [ncharges]
            )
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        Box::into_raw(fmm) as *mut HelmholtzFft32
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn helmholtz_fft_f64(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const f64,
        nsources: usize,
        targets: *const f64,
        ntargets: usize,
        charges: *const f64,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        wavenumber: f64,
        block_size: usize,
    ) -> *mut HelmholtzFft64 {
        let sources = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(sources, nsources), [nsources])
        };
        let targets = unsafe {
            rlst_array_from_slice1!(std::slice::from_raw_parts(targets, ntargets), [ntargets])
        };
        let charges = unsafe {
            rlst_array_from_slice1!(
                std::slice::from_raw_parts(charges as *const c64, ncharges),
                [ncharges]
            )
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        Box::into_raw(fmm) as *mut HelmholtzFft64
    }
}

/// FMM API
pub mod api {
    use green_kernels::helmholtz_3d::Helmholtz3dKernel;

    use crate::{BlasFieldTranslationIa, FftFieldTranslation, Fmm};

    use super::*;

    #[no_mangle]
    pub extern "C" fn evaluate_laplace_blas_f32(fmm: *mut LaplaceBlas32, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<
                    f32,
                    Laplace3dKernel<f32>,
                    BlasFieldTranslationSaRcmp<f32>,
                >)
            };
            let times = fmm.evaluate(timed).unwrap();
        }
    }

    #[no_mangle]
    pub extern "C" fn evaluate_laplace_blas_f64(fmm: *mut LaplaceBlas64, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<
                    f64,
                    Laplace3dKernel<f64>,
                    BlasFieldTranslationSaRcmp<f64>,
                >)
            };
            let times = fmm.evaluate(timed).unwrap();
        }
    }

    #[no_mangle]
    pub extern "C" fn evaluate_laplace_fft_f32(fmm: *mut LaplaceFft32, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>)
            };
            let times = fmm.evaluate(timed).unwrap();
        }
    }

    #[no_mangle]
    pub extern "C" fn evaluate_laplace_fft_f64(fmm: *mut LaplaceFft64, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>)
            };
            let times = fmm.evaluate(timed).unwrap();
        }
    }

    #[no_mangle]
    pub extern "C" fn evaluate_helmholtz_blas_f32(fmm: *mut HelmholtzBlas32, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>)
            };
            let times = fmm.evaluate(timed).unwrap();
        }
    }

    #[no_mangle]
    pub extern "C" fn evaluate_helmholtz_blas_f64(fmm: *mut HelmholtzBlas64, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>)
            };
            let times = fmm.evaluate(timed).unwrap();
        }
    }

    #[no_mangle]
    pub extern "C" fn evaluate_helmholtz_fft_f32(fmm: *mut HelmholtzFft32, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>)
            };
            let times = fmm.evaluate(timed).unwrap();
        }
    }

    #[no_mangle]
    pub extern "C" fn evaluate_helmholtz_fft_f64(fmm: *mut HelmholtzFft64, timed: bool) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>)
            };
            let times = fmm.evaluate(timed).unwrap();
        }
    }

    #[no_mangle]
    pub extern "C" fn clear_laplace_blas_f32(
        fmm: *mut LaplaceBlas32,
        charges: *const f32,
        ncharges: usize,
    ) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<
                    f32,
                    Laplace3dKernel<f32>,
                    BlasFieldTranslationSaRcmp<f32>,
                >)
            };
            let charges = unsafe { std::slice::from_raw_parts(charges, ncharges) };
            fmm.clear(charges);
        }
    }

    #[no_mangle]
    pub extern "C" fn clear_laplace_blas_f64(
        fmm: *mut LaplaceBlas64,
        charges: *const f64,
        ncharges: usize,
    ) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<
                    f64,
                    Laplace3dKernel<f64>,
                    BlasFieldTranslationSaRcmp<f64>,
                >)
            };
            let charges = unsafe { std::slice::from_raw_parts(charges, ncharges) };
            fmm.clear(charges);
        }
    }

    #[no_mangle]
    pub extern "C" fn clear_laplace_fft_f32(
        fmm: *mut LaplaceFft32,
        charges: *const f32,
        ncharges: usize,
    ) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>)
            };
            let charges = unsafe { std::slice::from_raw_parts(charges, ncharges) };
            fmm.clear(charges);
        }
    }

    #[no_mangle]
    pub extern "C" fn clear_laplace_fft_f64(
        fmm: *mut LaplaceFft64,
        charges: *const f64,
        ncharges: usize,
    ) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>)
            };
            let charges = unsafe { std::slice::from_raw_parts(charges, ncharges) };
            fmm.clear(charges);
        }
    }

    #[no_mangle]
    pub extern "C" fn clear_helmholtz_blas_f32(
        fmm: *mut HelmholtzBlas32,
        charges: *const f32,
        ncharges: usize,
    ) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>)
            };
            let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, ncharges) };
            fmm.clear(charges);
        }
    }

    #[no_mangle]
    pub extern "C" fn clear_helmholtz_blas_f64(
        fmm: *mut HelmholtzBlas64,
        charges: *const f64,
        ncharges: usize,
    ) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>)
            };
            let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, ncharges) };
            fmm.clear(charges);
        }
    }

    #[no_mangle]
    pub extern "C" fn clear_helmholtz_fft_f32(
        fmm: *mut HelmholtzFft32,
        charges: *const f32,
        ncharges: usize,
    ) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>)
            };
            let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, ncharges) };
            fmm.clear(charges);
        }
    }

    #[no_mangle]
    pub extern "C" fn clear_helmholtz_fft_f64(
        fmm: *mut HelmholtzFft64,
        charges: *const f64,
        ncharges: usize,
    ) {
        if !fmm.is_null() {
            // Cast back to the original type
            let fmm = unsafe {
                &mut *(fmm as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>)
            };
            let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, ncharges) };
            fmm.clear(charges);
        }
    }
}

pub use api::*;
pub use constructors::*;
pub use types::*;
