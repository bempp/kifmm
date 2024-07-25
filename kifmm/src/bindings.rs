use green_kernels::laplace_3d::Laplace3dKernel;
use rlst::{c32, c64, rlst_dynamic_array2, RawAccess, RawAccessMut};

use crate::{
    fmm::KiFmm, tree::helpers::points_fixture, BlasFieldTranslationSaRcmp, SingleNodeBuilder,
};

/// All types
pub mod types {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub enum FmmCType {
        Laplace32,
        Laplace64,
        Helmholtz32,
        Helmholtz64,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub enum FmmTranslationCType {
        Blas,
        Fft,
    }

    #[repr(C)]
    pub struct FmmEvaluator {
        pub ctype: FmmCType,
        pub ctranslation_type: FmmTranslationCType,
        pub data: *mut c_void,
    }

    impl FmmEvaluator {
        pub fn get_ctype(&self) -> FmmCType {
            self.ctype
        }

        pub fn get_ctranslation_type(&self) -> FmmTranslationCType {
            self.ctranslation_type
        }

        pub fn get_pointer(&self) -> *mut c_void {
            self.data
        }
    }

    #[repr(C)]
    pub enum ScalarType {
        F32,
        F64,
        C32,
        C64,
    }

    #[repr(C)]
    pub struct Potentials {
        pub len: usize,
        pub data: *const c_void,
        pub scalar: ScalarType,
    }

    #[repr(C)]
    pub struct GlobalIndices {
        pub len: usize,
        pub data: *const c_void,
    }

    #[repr(C)]
    pub struct MortonKeys {
        pub len: usize,
        pub data: *const u64,
    }
}

/// All constructors
pub mod constructors {
    use std::{f32::consts::E, ffi::c_void};

    use green_kernels::helmholtz_3d::Helmholtz3dKernel;
    use rlst::rlst_array_from_slice1;

    use crate::{BlasFieldTranslationIa, FftFieldTranslation};

    use super::*;

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_blas_svd_f32_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f32,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, ncharges) };

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

        let fmm = SingleNodeBuilder::new()
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                green_kernels::types::EvalType::Value,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Laplace32,
            ctranslation_type: FmmTranslationCType::Blas,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_blas_svd_f64_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f64,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, ncharges) };

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

        let fmm = SingleNodeBuilder::new()
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                green_kernels::types::EvalType::Value,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Laplace64,
            ctranslation_type: FmmTranslationCType::Blas,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_blas_rsvd_f32_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f32,
        n_components: usize,
        n_oversamples: usize,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, ncharges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);

        let n_components = if n_components > 0 {
            Some(n_components)
        } else {
            None
        };

        let n_oversamples = if n_oversamples > 0 {
            Some(n_oversamples)
        } else {
            None
        };

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            None,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let fmm = SingleNodeBuilder::new()
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                green_kernels::types::EvalType::Value,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Laplace32,
            ctranslation_type: FmmTranslationCType::Blas,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_blas_rsvd_f64_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f64,
        n_components: usize,
        n_oversamples: usize,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, ncharges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);

        let n_components = if n_components > 0 {
            Some(n_components)
        } else {
            None
        };

        let n_oversamples = if n_oversamples > 0 {
            Some(n_oversamples)
        } else {
            None
        };

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            None,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let fmm = SingleNodeBuilder::new()
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                green_kernels::types::EvalType::Value,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Laplace64,
            ctranslation_type: FmmTranslationCType::Blas,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_fft_f32_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, ncharges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources, targets, n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Laplace32,
            ctranslation_type: FmmTranslationCType::Fft,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn laplace_fft_f64_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, ncharges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources, targets, n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Laplace64,
            ctranslation_type: FmmTranslationCType::Fft,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn helmholtz_blas_svd_f32_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        wavenumber: f32,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f32,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, ncharges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);

        let field_translation = BlasFieldTranslationIa::new(singular_value_threshold, None);

        let fmm = SingleNodeBuilder::new()
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                green_kernels::types::EvalType::Value,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Helmholtz32,
            ctranslation_type: FmmTranslationCType::Blas,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn helmholtz_blas_svd_f64_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        wavenumber: f64,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f64,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, ntargets) };

        let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, ncharges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);

        let field_translation = BlasFieldTranslationIa::new(singular_value_threshold, None);

        let fmm = SingleNodeBuilder::new()
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                green_kernels::types::EvalType::Value,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Helmholtz64,
            ctranslation_type: FmmTranslationCType::Blas,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn helmholtz_fft_f32_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        wavenumber: f32,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, ncharges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources, targets, n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Helmholtz32,
            ctranslation_type: FmmTranslationCType::Fft,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }

    /// Constructor
    #[no_mangle]
    pub extern "C" fn helmholtz_fft_f64_alloc(
        expansion_order: *const usize,
        nexpansion_order: usize,
        wavenumber: f64,
        sources: *const c_void,
        nsources: usize,
        targets: *const c_void,
        ntargets: usize,
        charges: *const c_void,
        ncharges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut FmmEvaluator {
        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, nsources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, ntargets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, ncharges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, nexpansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources, targets, n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = Box::new(FmmEvaluator {
            data: data,
            ctype: FmmCType::Helmholtz64,
            ctranslation_type: FmmTranslationCType::Fft,
        });

        Box::into_raw(evaluator) as *mut FmmEvaluator
    }
}

/// FMM API
pub mod api {
    use std::os::raw::c_void;

    use green_kernels::helmholtz_3d::Helmholtz3dKernel;
    use itertools::Itertools;

    use crate::{
        traits::tree::{FmmTree, Tree, TreeNode},
        BlasFieldTranslationIa, FftFieldTranslation, Fmm,
    };

    use super::*;

    #[no_mangle]
    pub extern "C" fn evaluate(fmm: *mut FmmEvaluator, timed: bool) {
        if !fmm.is_null() {
            let ctype = unsafe { (*fmm).get_ctype() };
            let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
            let pointer = unsafe { (*fmm).get_pointer() };

            match ctype {
                FmmCType::Laplace32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        f32,
                                        Laplace3dKernel<f32>,
                                        FftFieldTranslation<f32>,
                                    >,
                            )
                        };

                        fmm.evaluate(timed);
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        f32,
                                        Laplace3dKernel<f32>,
                                        BlasFieldTranslationSaRcmp<f32>,
                                    >,
                            )
                        };

                        fmm.evaluate(timed);
                    }
                },

                FmmCType::Laplace64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        f64,
                                        Laplace3dKernel<f64>,
                                        FftFieldTranslation<f64>,
                                    >,
                            )
                        };

                        fmm.evaluate(timed);
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        f64,
                                        Laplace3dKernel<f64>,
                                        BlasFieldTranslationSaRcmp<f64>,
                                    >,
                            )
                        };

                        fmm.evaluate(timed);
                    }
                },

                FmmCType::Helmholtz32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        c32,
                                        Helmholtz3dKernel<c32>,
                                        FftFieldTranslation<c32>,
                                    >,
                            )
                        };

                        fmm.evaluate(timed);
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        c32,
                                        Helmholtz3dKernel<c32>,
                                        BlasFieldTranslationIa<c32>,
                                    >,
                            )
                        };

                        fmm.evaluate(timed);
                    }
                },

                FmmCType::Helmholtz64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        c64,
                                        Helmholtz3dKernel<c64>,
                                        FftFieldTranslation<c64>,
                                    >,
                            )
                        };

                        fmm.evaluate(timed);
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        c64,
                                        Helmholtz3dKernel<c64>,
                                        BlasFieldTranslationIa<c64>,
                                    >,
                            )
                        };

                        fmm.evaluate(timed);
                    }
                },
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn clear(fmm: *mut FmmEvaluator, charges: *const c_void, ncharges: usize) {
        if !fmm.is_null() {
            let ctype = unsafe { (*fmm).get_ctype() };
            let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
            let pointer = unsafe { (*fmm).get_pointer() };

            match ctype {
                FmmCType::Laplace32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let mut fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        f32,
                                        Laplace3dKernel<f32>,
                                        FftFieldTranslation<f32>,
                                    >,
                            )
                        };

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, ncharges) };
                        fmm.clear(charges);
                    }

                    FmmTranslationCType::Blas => {
                        let mut fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        f32,
                                        Laplace3dKernel<f32>,
                                        BlasFieldTranslationSaRcmp<f32>,
                                    >,
                            )
                        };

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, ncharges) };
                        fmm.clear(charges);
                    }
                },

                FmmCType::Laplace64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let mut fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        f64,
                                        Laplace3dKernel<f64>,
                                        FftFieldTranslation<f64>,
                                    >,
                            )
                        };

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, ncharges) };
                        fmm.clear(charges);
                    }

                    FmmTranslationCType::Blas => {
                        let mut fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        f64,
                                        Laplace3dKernel<f64>,
                                        BlasFieldTranslationSaRcmp<f64>,
                                    >,
                            )
                        };

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, ncharges) };
                        fmm.clear(charges);
                    }
                },

                FmmCType::Helmholtz32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let mut fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        c32,
                                        Helmholtz3dKernel<c32>,
                                        FftFieldTranslation<c32>,
                                    >,
                            )
                        };

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, ncharges) };
                        fmm.clear(charges);
                    }

                    FmmTranslationCType::Blas => {
                        let mut fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        c32,
                                        Helmholtz3dKernel<c32>,
                                        BlasFieldTranslationIa<c32>,
                                    >,
                            )
                        };

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, ncharges) };
                        fmm.clear(charges);
                    }
                },

                FmmCType::Helmholtz64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let mut fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        c64,
                                        Helmholtz3dKernel<c64>,
                                        FftFieldTranslation<c64>,
                                    >,
                            )
                        };

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, ncharges) };
                        fmm.clear(charges);
                    }

                    FmmTranslationCType::Blas => {
                        let mut fmm = unsafe {
                            Box::from_raw(
                                pointer
                                    as *mut KiFmm<
                                        c64,
                                        Helmholtz3dKernel<c64>,
                                        BlasFieldTranslationIa<c64>,
                                    >,
                            )
                        };

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, ncharges) };
                        fmm.clear(charges);
                    }
                },
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn potentials(fmm: *mut FmmEvaluator) -> Potentials {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>,
                        )
                    };

                    Potentials {
                        len: fmm.potentials.len(),
                        data: unsafe { fmm.potentials.as_ptr() as *const c_void },
                        scalar: ScalarType::F32,
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    f32,
                                    Laplace3dKernel<f32>,
                                    BlasFieldTranslationSaRcmp<f32>,
                                >,
                        )
                    };

                    Potentials {
                        len: fmm.potentials.len(),
                        data: unsafe { fmm.potentials.as_ptr() as *const c_void },
                        scalar: ScalarType::F32,
                    }
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>,
                        )
                    };

                    Potentials {
                        len: fmm.potentials.len(),
                        data: unsafe { fmm.potentials.as_ptr() as *const c_void },
                        scalar: ScalarType::F64,
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    f64,
                                    Laplace3dKernel<f64>,
                                    BlasFieldTranslationSaRcmp<f64>,
                                >,
                        )
                    };

                    Potentials {
                        len: fmm.potentials.len(),
                        data: unsafe { fmm.potentials.as_ptr() as *const c_void },
                        scalar: ScalarType::F64,
                    }
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    FftFieldTranslation<c32>,
                                >,
                        )
                    };

                    Potentials {
                        len: fmm.potentials.len(),
                        data: unsafe { fmm.potentials.as_ptr() as *const c_void },
                        scalar: ScalarType::C32,
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    BlasFieldTranslationIa<c32>,
                                >,
                        )
                    };

                    Potentials {
                        len: fmm.potentials.len(),
                        data: unsafe { fmm.potentials.as_ptr() as *const c_void },
                        scalar: ScalarType::C32,
                    }
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    FftFieldTranslation<c64>,
                                >,
                        )
                    };

                    Potentials {
                        len: fmm.potentials.len(),
                        data: unsafe { fmm.potentials.as_ptr() as *const c_void },
                        scalar: ScalarType::C64,
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    BlasFieldTranslationIa<c64>,
                                >,
                        )
                    };

                    Potentials {
                        len: fmm.potentials.len(),
                        data: unsafe { fmm.potentials.as_ptr() as *const c_void },
                        scalar: ScalarType::C64,
                    }
                }
            },
        }
    }

    #[no_mangle]
    pub extern "C" fn global_indices_source_tree(fmm: *mut FmmEvaluator) -> GlobalIndices {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.source_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.source_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    f32,
                                    Laplace3dKernel<f32>,
                                    BlasFieldTranslationSaRcmp<f32>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.source_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.source_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.source_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.source_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    f64,
                                    Laplace3dKernel<f64>,
                                    BlasFieldTranslationSaRcmp<f64>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.source_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.source_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    FftFieldTranslation<c32>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.source_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.source_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    BlasFieldTranslationIa<c32>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.source_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.source_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    FftFieldTranslation<c64>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.source_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.source_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    BlasFieldTranslationIa<c64>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.source_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.source_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }
            },
        }
    }

    #[no_mangle]
    pub extern "C" fn global_indices_target_tree(fmm: *mut FmmEvaluator) -> GlobalIndices {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.target_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.target_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    f32,
                                    Laplace3dKernel<f32>,
                                    BlasFieldTranslationSaRcmp<f32>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.target_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.target_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.target_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.target_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    f64,
                                    Laplace3dKernel<f64>,
                                    BlasFieldTranslationSaRcmp<f64>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.target_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.target_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    FftFieldTranslation<c32>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.target_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.target_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    BlasFieldTranslationIa<c32>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.target_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.target_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    FftFieldTranslation<c64>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.target_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.target_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    BlasFieldTranslationIa<c64>,
                                >,
                        )
                    };

                    GlobalIndices {
                        len: fmm.tree.target_tree.global_indices.len(),
                        data: unsafe {
                            fmm.tree.target_tree.global_indices.as_ptr() as *const c_void
                        },
                    }
                }
            },
        }
    }

    #[no_mangle]
    pub extern "C" fn all_leaves_source_tree(fmm: *mut FmmEvaluator) -> MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>,
                        )
                    };

                    MortonKeys {
                        len: fmm.tree.source_tree.leaves.len(),
                        data: unsafe {
                            fmm.tree.source_tree.leaves.as_ptr() as *const u64
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    f32,
                                    Laplace3dKernel<f32>,
                                    BlasFieldTranslationSaRcmp<f32>,
                                >,
                        )
                    };

                    MortonKeys {
                        len: fmm.tree.source_tree.leaves.len(),
                        data: unsafe {
                            fmm.tree.source_tree.leaves.as_ptr() as *const u64
                        },
                    }
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>,
                        )
                    };

                    MortonKeys {
                        len: fmm.tree.source_tree.leaves.len(),
                        data: unsafe {
                            fmm.tree.source_tree.leaves.as_ptr() as *const c_void
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    f64,
                                    Laplace3dKernel<f64>,
                                    BlasFieldTranslationSaRcmp<f64>,
                                >,
                        )
                    };

                    MortonKeys {
                        len: fmm.tree.source_tree.leaves.len(),
                        data: unsafe {
                            fmm.tree.source_tree.leaves.as_ptr() as *const u64
                        },
                    }
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    FftFieldTranslation<c32>,
                                >,
                        )
                    };

                    MortonKeys {
                        len: fmm.tree.source_tree.leaves.len(),
                        data: unsafe {
                            fmm.tree.source_tree.leaves.as_ptr() as *const u64
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    BlasFieldTranslationIa<c32>,
                                >,
                        )
                    };

                    MortonKeys {
                        len: fmm.tree.source_tree.leaves.len(),
                        data: unsafe {
                            fmm.tree.source_tree.leaves.as_ptr() as *const u64
                        },
                    }
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    FftFieldTranslation<c64>,
                                >,
                        )
                    };

                    MortonKeys {
                        len: fmm.tree.source_tree.leaves.len(),
                        data: unsafe {
                            fmm.tree.source_tree.leaves.as_ptr() as *const u64
                        },
                    }
                }

                FmmTranslationCType::Blas => {
                    let mut fmm = unsafe {
                        Box::from_raw(
                            pointer
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    BlasFieldTranslationIa<c64>,
                                >,
                        )
                    };

                    MortonKeys {
                        len: fmm.tree.source_tree.leaves.len(),
                        data: unsafe {
                            fmm.tree.source_tree.leaves.as_ptr() as *const u64
                        },
                    }
                }
            },
        }
    }
}

pub use api::*;
pub use constructors::*;
pub use types::*;
