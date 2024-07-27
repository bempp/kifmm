#![allow(missing_docs)]

use green_kernels::laplace_3d::Laplace3dKernel;
use rlst::{c32, c64};

use crate::{
    fmm::KiFmm, BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation,
    SingleNodeBuilder,
};

/// All types
pub mod types {
    use std::ffi::c_void;

    use crate::traits::types::FmmOperatorTime;

    /// Static FMM type
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub enum FmmCType {
        Laplace32,
        Laplace64,
        Helmholtz32,
        Helmholtz64,
    }

    /// M2L field translation mode
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub enum FmmTranslationCType {
        Blas,
        Fft,
    }

    /// Runtime FMM type constructed from C
    #[repr(C)]
    pub struct FmmEvaluator {
        pub ctype: FmmCType,
        pub ctranslation_type: FmmTranslationCType,
        pub data: *mut c_void,
    }

    impl FmmEvaluator {
        /// Get the static FMM type
        pub fn get_ctype(&self) -> FmmCType {
            self.ctype
        }

        /// Get the M2L field translation type
        pub fn get_ctranslation_type(&self) -> FmmTranslationCType {
            self.ctranslation_type
        }

        /// Get the pointer to underlying runtime object
        pub fn get_pointer(&self) -> *mut c_void {
            self.data
        }
    }

    /// Scalar type
    #[repr(C)]
    pub enum ScalarType {
        /// Float
        F32,
        /// Double
        F64,
        /// Complex FLoat
        C32,
        /// Complex Double
        C64,
    }

    /// Coordinates
    #[repr(C)]
    pub struct Coordinates {
        /// Length of coordinates buffer of length 3*n_coordinates
        pub len: usize,
        /// Pointer to underlying buffer
        pub data: *const c_void,
        /// Associated scalar type
        pub scalar: ScalarType,
    }

    /// Potential data
    #[repr(C)]
    pub struct Potential {
        /// Length of underlying buffer, of length n_eval_mode*n_coordinates
        /// currently only support n_eval_mode=1, i.e. potentials
        pub len: usize,
        /// Pointer to underlying buffer
        pub data: *const c_void,
        /// Associated scalar type
        pub scalar: ScalarType,
    }

    /// Container for multiple Potentials. Used when FMM run over multiple
    /// charge vectors.
    #[repr(C)]
    pub struct Potentials {
        /// Number of charge vectors associated with FMM call ()
        pub n: usize,
        /// Pointer to underlying buffer
        pub data: *mut Potential,
        /// Associated scalar type
        pub scalar: ScalarType,
    }

    /// Implicit map between input coordinate index and global index
    /// after sorting during octree construction
    #[repr(C)]
    pub struct GlobalIndices {
        /// Number of global indices
        pub len: usize,
        /// Pointer to underlying buffer
        pub data: *const c_void,
    }

    /// Morton keys, used to describe octree boxes, each represented as unique u64.
    #[repr(C)]
    pub struct MortonKeys {
        /// Number of morton keys
        pub len: usize,
        /// Pointer to underlying buffer
        pub data: *const u64,
    }

    #[repr(C)]
    pub struct FmmOperatorTimes {
        pub times: *mut FmmOperatorTime,
        pub length: usize,
    }
}

impl Drop for MortonKeys {
    fn drop(&mut self) {
        let Self { len, data } = self;

        let tmp = unsafe { Vec::from_raw_parts(*data as *mut u64, *len, *len) };
        drop(tmp);
    }
}

impl Drop for FmmEvaluator {
    fn drop(&mut self) {
        let Self {
            ctype,
            ctranslation_type,
            data,
        } = self;

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Blas => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut Box<
                                    KiFmm<
                                        f32,
                                        Laplace3dKernel<f32>,
                                        BlasFieldTranslationSaRcmp<f32>,
                                    >,
                                >,
                        )
                    });
                }

                FmmTranslationCType::Fft => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut Box<
                                    KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>,
                                >,
                        )
                    });
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Blas => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut Box<
                                    KiFmm<
                                        f64,
                                        Laplace3dKernel<f64>,
                                        BlasFieldTranslationSaRcmp<f64>,
                                    >,
                                >,
                        )
                    });
                }

                FmmTranslationCType::Fft => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut Box<
                                    KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>,
                                >,
                        )
                    });
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Blas => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut Box<
                                    KiFmm<c32, Laplace3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                                >,
                        )
                    });
                }

                FmmTranslationCType::Fft => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut Box<
                                    KiFmm<c32, Laplace3dKernel<c32>, FftFieldTranslation<c32>>,
                                >,
                        )
                    });
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Blas => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut Box<
                                    KiFmm<c64, Laplace3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                                >,
                        )
                    });
                }

                FmmTranslationCType::Fft => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut Box<
                                    KiFmm<c64, Laplace3dKernel<c64>, FftFieldTranslation<c64>>,
                                >,
                        )
                    });
                }
            },
        }
    }
}

/// Free the FmmEvaluator object
///
/// # Safety
/// This function is intended to be called from C. The caller must ensure that:
/// - `fmm_p` is a valid pointer to a properly initialized `FmmEvaluator` instance.
/// - The `fmm_p` pointer remains valid for the duration of the function call.
#[no_mangle]
pub unsafe extern "C" fn free_fmm_evaluator(fmm_p: *mut FmmEvaluator) {
    assert!(!fmm_p.is_null());
    unsafe { drop(Box::from_raw(fmm_p)) }
}

/// All constructors
pub mod constructors {
    use std::ffi::c_void;

    use green_kernels::helmholtz_3d::Helmholtz3dKernel;

    use crate::{BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation};

    use super::{
        c32, c64, FmmCType, FmmEvaluator, FmmTranslationCType, Laplace3dKernel, SingleNodeBuilder,
    };

    /// Constructor for F32 Laplace FMM with BLAS based M2L translations compressed
    /// with deterministic SVD.
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_svd_f32_alloc(
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

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Laplace32,
            ctranslation_type: FmmTranslationCType::Blas,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F64 Laplace FMM with BLAS based M2L translations compressed
    /// with deterministic SVD.
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_svd_f64_alloc(
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

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Laplace64,
            ctranslation_type: FmmTranslationCType::Blas,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F32 Laplace FMM with BLAS based M2L translations compressed
    /// with randomised SVD.
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_rsvd_f32_alloc(
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

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Laplace32,
            ctranslation_type: FmmTranslationCType::Blas,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F64 Laplace FMM with BLAS based M2L translations compressed
    /// with randomised SVD.
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_rsvd_f64_alloc(
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

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Laplace64,
            ctranslation_type: FmmTranslationCType::Blas,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F32 Laplace FMM with FFT based M2L translations
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_fft_f32_alloc(
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
        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources, targets, n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(block_size),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Laplace32,
            ctranslation_type: FmmTranslationCType::Fft,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F64 Laplace FMM with FFT based M2L translations
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_fft_f64_alloc(
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
        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources, targets, n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(block_size),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Laplace64,
            ctranslation_type: FmmTranslationCType::Fft,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F32 Helmholtz FMM with BLAS based M2L translations compressed
    /// with deterministic SVD.
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_svd_f32_alloc(
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

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Helmholtz32,
            ctranslation_type: FmmTranslationCType::Blas,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F64 Helmholtz FMM with BLAS based M2L translations compressed
    /// with deterministic SVD.
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_svd_f64_alloc(
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

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Helmholtz64,
            ctranslation_type: FmmTranslationCType::Blas,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F32 Helmholtz FMM with FFT based M2L translations
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_fft_f32_alloc(
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
        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources, targets, n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(block_size),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Helmholtz32,
            ctranslation_type: FmmTranslationCType::Fft,
        };

        Box::into_raw(Box::new(evaluator))
    }

    /// Constructor for F64 Helmholtz FMM with FFT based M2L translations
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    ///
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `nexpansion_order`: The number of expansion orders.
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `nsources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `ntargets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///    reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_fft_f64_alloc(
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
        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let fmm = Box::new(
            SingleNodeBuilder::new()
                .tree(sources, targets, n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    green_kernels::types::EvalType::Value,
                    FftFieldTranslation::new(block_size),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Helmholtz64,
            ctranslation_type: FmmTranslationCType::Fft,
        };

        Box::into_raw(Box::new(evaluator))
    }
}

/// FMM API
pub mod api {
    use std::{mem::ManuallyDrop, os::raw::c_void};

    use green_kernels::helmholtz_3d::Helmholtz3dKernel;
    use itertools::Itertools;

    use crate::{
        fmm::types::FmmEvalType,
        traits::tree::{FmmTree, Tree, TreeNode},
        tree::types::MortonKey,
        BlasFieldTranslationIa, FftFieldTranslation, Fmm,
    };

    use super::{
        c32, c64, BlasFieldTranslationSaRcmp, Coordinates, FmmCType, FmmEvaluator,
        FmmOperatorTimes, FmmTranslationCType, GlobalIndices, KiFmm, Laplace3dKernel, MortonKeys,
        Potential, Potentials, ScalarType,
    };

    /// Evaluate the Fast Multipole Method (FMM).
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `timed`: Boolean flag to time each operator.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn evaluate(
        fmm: *mut FmmEvaluator,
        timed: bool,
    ) -> *mut FmmOperatorTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;
                    let _ = unsafe { (*fmm).evaluate(timed) };

                    let length = unsafe { (*fmm).times.len() };
                    let times = unsafe { (*fmm).times.as_mut_ptr() };

                    let times = FmmOperatorTimes { times, length };

                    Box::into_raw(Box::new(times))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let _ = unsafe { (*fmm).evaluate(timed) };
                    let length = unsafe { (*fmm).times.len() };
                    let times = unsafe { (*fmm).times.as_mut_ptr() };

                    let times = FmmOperatorTimes { times, length };

                    Box::into_raw(Box::new(times))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let _ = unsafe { (*fmm).evaluate(timed) };
                    let length = unsafe { (*fmm).times.len() };
                    let times = unsafe { (*fmm).times.as_mut_ptr() };

                    let times = FmmOperatorTimes { times, length };

                    Box::into_raw(Box::new(times))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let _ = unsafe { (*fmm).evaluate(timed) };
                    let length = unsafe { (*fmm).times.len() };
                    let times = unsafe { (*fmm).times.as_mut_ptr() };

                    let times = FmmOperatorTimes { times, length };

                    Box::into_raw(Box::new(times))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let _ = unsafe { (*fmm).evaluate(timed) };
                    let length = unsafe { (*fmm).times.len() };
                    let times = unsafe { (*fmm).times.as_mut_ptr() };

                    let times = FmmOperatorTimes { times, length };

                    Box::into_raw(Box::new(times))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let _ = unsafe { (*fmm).evaluate(timed) };
                    let length = unsafe { (*fmm).times.len() };
                    let times = unsafe { (*fmm).times.as_mut_ptr() };

                    let times = FmmOperatorTimes { times, length };

                    Box::into_raw(Box::new(times))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let _ = unsafe { (*fmm).evaluate(timed) };
                    let length = unsafe { (*fmm).times.len() };
                    let times = unsafe { (*fmm).times.as_mut_ptr() };

                    let times = FmmOperatorTimes { times, length };

                    Box::into_raw(Box::new(times))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let _ = unsafe { (*fmm).evaluate(timed) };
                    let length = unsafe { (*fmm).times.len() };
                    let times = unsafe { (*fmm).times.as_mut_ptr() };

                    let times = FmmOperatorTimes { times, length };

                    Box::into_raw(Box::new(times))
                }
            },
        }

        // match times {
        //     Ok(times) => {
        //         let length = times.len();
        //         let times: *mut crate::traits::types::FmmOperatorTime = times.into_boxed_slice().as_mut_ptr();
        //         FmmOperatorTimesResult {
        //             result: FmmOperatorTimes { times, length },
        //             error: FmmError::None,
        //         }
        //     }
        //     Err(_e) => FmmOperatorTimesResult {
        //         result: FmmOperatorTimes {
        //             times: std::ptr::null_mut(),
        //             length: 0,
        //         },
        //         error: FmmError::EvaluationFailed,
        //     },
        // }
    }

    /// Clear charges and attach new charges.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `charges`: A pointer to the new charges associated with the source points.
    /// - `ncharges`: The length of the charges buffer.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn clear(
        fmm: *mut FmmEvaluator,
        charges: *const c_void,
        ncharges: usize,
    ) {
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

    /// Query for all evaluated potentials, returned in order of global index.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn all_potentials(fmm: *mut FmmEvaluator) -> *mut Potential {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len(),
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::F32,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len(),
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::F32,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len(),
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::F64,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len(),
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::F64,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len(),
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C32,
                        }
                    };
                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len(),
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C32,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len(),
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C64,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len(),
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C64,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },
        }
    }

    /// Free Morton keys
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn free_morton_keys(keys_p: *mut MortonKeys) {
        if !keys_p.is_null() {
            unsafe { drop(Box::from_raw(keys_p)) }
        }
    }

    /// Query for global indices of target points, where each index position corresponds to input
    /// coordinate data index, and the elements correspond to the index as stored in the target tree
    /// and therefore in the evaluated potentials.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn global_indices_target_tree(
        fmm: *mut FmmEvaluator,
    ) -> *mut GlobalIndices {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.target_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.target_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.target_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.target_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.target_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.target_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.target_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.target_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.target_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.target_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.target_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.target_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.target_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.target_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.target_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.target_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }
            },
        }
    }

    /// Query for global indices of source points, where each index position corresponds to input
    /// coordinate data index, and the elements correspond to the index as stored in the source tree.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    pub unsafe extern "C" fn global_indices_source_tree(
        fmm: *mut FmmEvaluator,
    ) -> *mut GlobalIndices {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.source_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.source_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.source_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.source_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.source_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.source_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.source_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.source_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.source_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.source_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.source_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.source_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.source_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.source_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let global_indices = unsafe {
                        GlobalIndices {
                            len: (*fmm).tree.source_tree.global_indices.len(),
                            data: {
                                (*fmm).tree.source_tree.global_indices.as_ptr() as *const c_void
                            },
                        }
                    };

                    Box::into_raw(Box::new(global_indices))
                }
            },
        }
    }

    /// Query for potentials at a specific leaf.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `leaf`: The identifier of a leaf node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn leaf_potentials(fmm: *mut FmmEvaluator, leaf: u64) -> *mut Potentials {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;

                    let mut nvecs = unsafe {
                        match (*fmm).fmm_eval_type {
                            FmmEvalType::Matrix(n) => n,
                            FmmEvalType::Vector => 1,
                        }
                    };

                    let leaf = MortonKey::from_morton(leaf);

                    let mut potentials = Vec::new();

                    unsafe {
                        if let Some(tmp) = (*fmm).potential(&leaf) {
                            for &p in tmp.iter() {
                                potentials.push(Potential {
                                    len: p.len(),
                                    data: p.as_ptr() as *const c_void,
                                    scalar: ScalarType::F32,
                                })
                            }
                        } else {
                            nvecs = 0
                        }
                    }

                    let mut potentials = ManuallyDrop::new(potentials);

                    let potentials = Potentials {
                        n: nvecs,
                        data: potentials.as_mut_ptr(),
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let mut nvecs = unsafe {
                        match (*fmm).fmm_eval_type {
                            FmmEvalType::Matrix(n) => n,
                            FmmEvalType::Vector => 1,
                        }
                    };

                    let leaf = MortonKey::from_morton(leaf);

                    let mut potentials = Vec::new();

                    unsafe {
                        if let Some(tmp) = (*fmm).potential(&leaf) {
                            for &p in tmp.iter() {
                                potentials.push(Potential {
                                    len: p.len(),
                                    data: p.as_ptr() as *const c_void,
                                    scalar: ScalarType::F32,
                                })
                            }
                        } else {
                            nvecs = 0
                        }
                    }

                    let mut potentials = ManuallyDrop::new(potentials);
                    let potentials = Potentials {
                        n: nvecs,
                        data: potentials.as_mut_ptr(),
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let mut nvecs = unsafe {
                        match (*fmm).fmm_eval_type {
                            FmmEvalType::Matrix(n) => n,
                            FmmEvalType::Vector => 1,
                        }
                    };

                    let leaf = MortonKey::from_morton(leaf);

                    let mut potentials = Vec::new();

                    unsafe {
                        if let Some(tmp) = (*fmm).potential(&leaf) {
                            for &p in tmp.iter() {
                                potentials.push(Potential {
                                    len: p.len(),
                                    data: p.as_ptr() as *const c_void,
                                    scalar: ScalarType::F64,
                                })
                            }
                        } else {
                            nvecs = 0
                        }
                    }

                    let mut potentials = ManuallyDrop::new(potentials);
                    let potentials = Potentials {
                        n: nvecs,
                        data: potentials.as_mut_ptr(),
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let mut nvecs = unsafe {
                        match (*fmm).fmm_eval_type {
                            FmmEvalType::Matrix(n) => n,
                            FmmEvalType::Vector => 1,
                        }
                    };

                    let leaf = MortonKey::from_morton(leaf);

                    let mut potentials = Vec::new();

                    unsafe {
                        if let Some(tmp) = (*fmm).potential(&leaf) {
                            for &p in tmp.iter() {
                                potentials.push(Potential {
                                    len: p.len(),
                                    data: p.as_ptr() as *const c_void,
                                    scalar: ScalarType::F64,
                                })
                            }
                        } else {
                            nvecs = 0
                        }
                    }

                    let mut potentials = ManuallyDrop::new(potentials);
                    let potentials = Potentials {
                        n: nvecs,
                        data: potentials.as_mut_ptr(),
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let mut nvecs = unsafe {
                        match (*fmm).fmm_eval_type {
                            FmmEvalType::Matrix(n) => n,
                            FmmEvalType::Vector => 1,
                        }
                    };

                    let leaf = MortonKey::from_morton(leaf);

                    let mut potentials = Vec::new();

                    unsafe {
                        if let Some(tmp) = (*fmm).potential(&leaf) {
                            for &p in tmp.iter() {
                                potentials.push(Potential {
                                    len: p.len(),
                                    data: p.as_ptr() as *const c_void,
                                    scalar: ScalarType::C32,
                                })
                            }
                        } else {
                            nvecs = 0
                        }
                    }

                    let mut potentials = ManuallyDrop::new(potentials);
                    let potentials = Potentials {
                        n: nvecs,
                        data: potentials.as_mut_ptr(),
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let mut nvecs = unsafe {
                        match (*fmm).fmm_eval_type {
                            FmmEvalType::Matrix(n) => n,
                            FmmEvalType::Vector => 1,
                        }
                    };

                    let leaf = MortonKey::from_morton(leaf);

                    let mut potentials = Vec::new();

                    unsafe {
                        if let Some(tmp) = (*fmm).potential(&leaf) {
                            for &p in tmp.iter() {
                                potentials.push(Potential {
                                    len: p.len(),
                                    data: p.as_ptr() as *const c_void,
                                    scalar: ScalarType::C32,
                                })
                            }
                        } else {
                            nvecs = 0
                        }
                    }

                    let mut potentials = ManuallyDrop::new(potentials);
                    let potentials = Potentials {
                        n: nvecs,
                        data: potentials.as_mut_ptr(),
                        scalar: ScalarType::C32,
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let mut nvecs = unsafe {
                        match (*fmm).fmm_eval_type {
                            FmmEvalType::Matrix(n) => n,
                            FmmEvalType::Vector => 1,
                        }
                    };

                    let leaf = MortonKey::from_morton(leaf);

                    let mut potentials = Vec::new();

                    unsafe {
                        if let Some(tmp) = (*fmm).potential(&leaf) {
                            for &p in tmp.iter() {
                                potentials.push(Potential {
                                    len: p.len(),
                                    data: p.as_ptr() as *const c_void,
                                    scalar: ScalarType::C64,
                                })
                            }
                        } else {
                            nvecs = 0
                        }
                    }

                    let mut potentials = ManuallyDrop::new(potentials);
                    let potentials = Potentials {
                        n: nvecs,
                        data: potentials.as_mut_ptr(),
                        scalar: ScalarType::C64,
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let mut nvecs = unsafe {
                        match (*fmm).fmm_eval_type {
                            FmmEvalType::Matrix(n) => n,
                            FmmEvalType::Vector => 1,
                        }
                    };

                    let leaf = MortonKey::from_morton(leaf);

                    let mut potentials = Vec::new();

                    unsafe {
                        if let Some(tmp) = (*fmm).potential(&leaf) {
                            for &p in tmp.iter() {
                                potentials.push(Potential {
                                    len: p.len(),
                                    data: p.as_ptr() as *const c_void,
                                    scalar: ScalarType::C64,
                                })
                            }
                        } else {
                            nvecs = 0
                        }
                    }

                    let mut potentials = ManuallyDrop::new(potentials);
                    let potentials = Potentials {
                        n: nvecs,
                        data: potentials.as_mut_ptr(),
                        scalar: ScalarType::C64,
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },
        }
    }

    /// Query source tree for coordinates contained in a leaf box.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `leaf`: The identifier of the leaf node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn coordinates_source_tree(
        fmm: *mut FmmEvaluator,
        leaf: u64,
    ) -> *mut Coordinates {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().source_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(coordinates))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().source_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(coordinates))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().source_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(coordinates))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().source_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(coordinates))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().source_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(coordinates))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().source_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(coordinates))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().source_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(coordinates))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().source_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(coordinates))
                }
            },
        }
    }

    /// Query target tree for coordinates contained in a leaf box.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `leaf`: The identifier of the leaf node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn coordinates_target_tree(
        fmm: *mut FmmEvaluator,
        leaf: u64,
    ) -> *mut Coordinates {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().target_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(coordinates))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().target_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(coordinates))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().target_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(coordinates))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().target_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(coordinates))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().target_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(coordinates))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().target_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(coordinates))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().target_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(coordinates))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let leaf = MortonKey::from_morton(leaf);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(coords) = (*fmm).tree().target_tree().coordinates(&leaf) {
                            len = coords.len();
                            ptr = coords.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let coordinates = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(coordinates))
                }
            },
        }
    }

    /// Query target tree for coordinates contained in a leaf box.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `leaf`: The identifier of the leaf node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn leaves_target_tree(fmm: *mut FmmEvaluator) -> *mut MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;

                    let leaves: Vec<u64> = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }
            },
        }
    }

    /// Query source tree for coordinates contained in a leaf box.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `leaf`: The identifier of the leaf node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn leaves_source_tree(fmm: *mut FmmEvaluator) -> *mut MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>;

                    let leaves: Vec<u64> = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>,
                        >;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>>;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>,
                        >;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>,
                        >;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut Box<KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>>;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut Box<
                            KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>,
                        >;

                    let leaves = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .leaves
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let leaves = ManuallyDrop::new(leaves);

                    let leaves = MortonKeys {
                        len: leaves.len(),
                        data: leaves.as_ptr(),
                    };

                    Box::into_raw(Box::new(leaves))
                }
            },
        }
    }
}

pub use api::*;
pub use constructors::*;
pub use types::*;
