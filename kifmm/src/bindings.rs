//! C bindings for KIFMM-rs. Used as a basis for language bindings into Python, C and other C ABI compatible languages.
#![allow(missing_docs)]

use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use rlst::{c32, c64};

use crate::{
    BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation, KiFmm,
    SingleNodeBuilder,
};

/// All types
pub mod types {
    use std::ffi::c_void;

    use crate::traits::types::{CommunicationType, FmmOperatorType, MetadataType, OperatorTime};

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
        /// Length of underlying buffer, of length n_eval_mode*n_coordinates*n_evals
        pub len: usize,
        /// Pointer to underlying buffer
        pub data: *const c_void,
        /// Associated scalar type
        pub scalar: ScalarType,
    }

    /// Expansion data
    #[repr(C)]
    pub struct Expansion {
        /// Length of underlying buffer, of length n_eval_mode*n_evals*n_coeffs
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
    #[derive(Debug, Clone, Copy)]
    pub struct FmmOperatorEntry {
        pub op_type: FmmOperatorType,
        pub time: OperatorTime,
    }

    #[repr(C)]
    pub struct FmmOperatorTimes {
        pub times: *mut FmmOperatorEntry,
        pub length: usize,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct MetadataEntry {
        pub metadata_type: MetadataType,
        pub time: OperatorTime,
    }

    #[repr(C)]
    pub struct MetadataTimes {
        pub times: *mut MetadataEntry,
        pub length: usize,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct CommunicationEntry {
        pub comm_type: CommunicationType,
        pub time: OperatorTime,
    }

    #[repr(C)]
    pub struct CommunicationTimes {
        pub times: *mut CommunicationEntry,
        pub length: usize,
    }
}

#[cfg(feature = "mpi")]
pub mod mpi_types {
    use crate::KiFmmMulti;
    use std::os::raw::c_void;

    use super::{
        c32, c64, BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation,
        FmmCType, FmmTranslationCType, Helmholtz3dKernel, Laplace3dKernel,
    };

    /// Runtime FMM type constructed from C
    #[repr(C)]
    pub struct FmmEvaluatorMPI {
        pub ctype: FmmCType,
        pub ctranslation_type: FmmTranslationCType,
        pub data: *mut c_void,
    }

    impl FmmEvaluatorMPI {
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

    impl Drop for FmmEvaluatorMPI {
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
                                    as *mut KiFmmMulti<
                                        f32,
                                        Laplace3dKernel<f32>,
                                        BlasFieldTranslationSaRcmp<f32>,
                                    >,
                            )
                        });
                    }

                    FmmTranslationCType::Fft => {
                        drop(unsafe {
                            Box::from_raw(
                                *data
                                    as *mut KiFmmMulti<
                                        f32,
                                        Laplace3dKernel<f32>,
                                        FftFieldTranslation<f32>,
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
                                    as *mut KiFmmMulti<
                                        f64,
                                        Laplace3dKernel<f64>,
                                        BlasFieldTranslationSaRcmp<f64>,
                                    >,
                            )
                        });
                    }

                    FmmTranslationCType::Fft => {
                        drop(unsafe {
                            Box::from_raw(
                                *data
                                    as *mut KiFmmMulti<
                                        f64,
                                        Laplace3dKernel<f64>,
                                        FftFieldTranslation<f64>,
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
                                    as *mut KiFmmMulti<
                                        c32,
                                        Helmholtz3dKernel<c32>,
                                        BlasFieldTranslationIa<c32>,
                                    >,
                            )
                        });
                    }

                    FmmTranslationCType::Fft => {
                        drop(unsafe {
                            Box::from_raw(
                                *data
                                    as *mut KiFmmMulti<
                                        c32,
                                        Helmholtz3dKernel<c32>,
                                        FftFieldTranslation<c32>,
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
                                    as *mut KiFmmMulti<
                                        c64,
                                        Helmholtz3dKernel<c64>,
                                        BlasFieldTranslationIa<c64>,
                                    >,
                            )
                        });
                    }

                    FmmTranslationCType::Fft => {
                        drop(unsafe {
                            Box::from_raw(
                                *data
                                    as *mut KiFmmMulti<
                                        c64,
                                        Helmholtz3dKernel<c64>,
                                        FftFieldTranslation<c64>,
                                    >,
                            )
                        });
                    }
                },
            }
        }
    }

    /// Free the FmmEvaluatorMPI object
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - `fmm_p` is a valid pointer to a properly initialized `FmmEvaluator` instance.
    /// - The `fmm_p` pointer remains valid for the duration of the function call.
    #[no_mangle]
    pub unsafe extern "C" fn free_fmm_evaluator_mpi(fmm_p: *mut FmmEvaluatorMPI) {
        assert!(!fmm_p.is_null());
        unsafe { drop(Box::from_raw(fmm_p)) }
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
                                as *mut KiFmm<
                                    f32,
                                    Laplace3dKernel<f32>,
                                    BlasFieldTranslationSaRcmp<f32>,
                                >,
                        )
                    });
                }

                FmmTranslationCType::Fft => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>,
                        )
                    });
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Blas => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut KiFmm<
                                    f64,
                                    Laplace3dKernel<f64>,
                                    BlasFieldTranslationSaRcmp<f64>,
                                >,
                        )
                    });
                }

                FmmTranslationCType::Fft => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>,
                        )
                    });
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Blas => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    BlasFieldTranslationIa<c32>,
                                >,
                        )
                    });
                }

                FmmTranslationCType::Fft => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut KiFmm<
                                    c32,
                                    Helmholtz3dKernel<c32>,
                                    FftFieldTranslation<c32>,
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
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    BlasFieldTranslationIa<c64>,
                                >,
                        )
                    });
                }

                FmmTranslationCType::Fft => {
                    drop(unsafe {
                        Box::from_raw(
                            *data
                                as *mut KiFmm<
                                    c64,
                                    Helmholtz3dKernel<c64>,
                                    FftFieldTranslation<c64>,
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

    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::GreenKernelEvalType};

    use crate::{
        BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation, FmmSvdMode,
    };

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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_svd_f32_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f32,
        surface_diff: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::Deterministic,
        );

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_svd_f64_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f64,
        surface_diff: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::Deterministic,
        );

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_rsvd_f32_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f32,
        surface_diff: usize,
        n_components: usize,
        n_oversamples: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

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
            surface_diff,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_rsvd_f64_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f64,
        surface_diff: usize,
        n_components: usize,
        n_oversamples: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

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
            surface_diff,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_fft_f32_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };
        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_fft_f64_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };
        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_rsvd_f32_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f32,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f32,
        surface_diff: usize,
        n_components: usize,
        n_oversamples: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

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

        let field_translation = BlasFieldTranslationIa::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_rsvd_f64_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f64,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f64,
        surface_diff: usize,
        n_components: usize,
        n_oversamples: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };

        let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };
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

        let field_translation = BlasFieldTranslationIa::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
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

    /// Constructor for F32 Helmholtz FMM with BLAS based M2L translations compressed
    /// with deterministic SVD.
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_svd_f32_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f32,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f32,
        surface_diff: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

        let field_translation = BlasFieldTranslationIa::new(
            singular_value_threshold,
            surface_diff,
            FmmSvdMode::Deterministic,
        );

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_svd_f64_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f64,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        singular_value_threshold: f64,
        surface_diff: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };

        let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };

        let depth = if depth > 0 { Some(depth) } else { None };
        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

        let field_translation = BlasFieldTranslationIa::new(
            singular_value_threshold,
            surface_diff,
            FmmSvdMode::Deterministic,
        );

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_fft_f32_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f32,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };
        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_fft_f64_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f64,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        n_crit: u64,
        depth: u64,
        block_size: usize,
    ) -> *mut FmmEvaluator {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };
        let n_crit = if n_crit > 0 { Some(n_crit) } else { None };
        let depth = if depth > 0 { Some(depth) } else { None };
        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let fmm = SingleNodeBuilder::new(timed)
            .tree(sources, targets, n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluator {
            data,
            ctype: FmmCType::Helmholtz64,
            ctranslation_type: FmmTranslationCType::Fft,
        };

        Box::into_raw(Box::new(evaluator))
    }
}

/// All constructors
#[cfg(feature = "mpi")]
pub mod constructors_mpi {
    use core::panic;
    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::GreenKernelEvalType};
    use mpi::raw::{AsRaw, FromRaw};
    use std::ffi::c_void;

    use crate::{
        BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, DataAccessMulti, FftFieldTranslation, FmmSvdMode, MultiNodeBuilder
    };

    use super::{c32, c64, FmmCType, FmmEvaluatorMPI, FmmTranslationCType, Laplace3dKernel};

    /// Constructor for F32 Laplace FMM with BLAS based M2L translations compressed
    /// with deterministic SVD.
    ///
    /// Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
    /// must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
    /// at each level, and stored as a buffer of length `depth` + 1.
    ///
    ///
    /// # Parameters
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_svd_f32_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        singular_value_threshold: f32,
        surface_diff: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for Samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("Only '0' Simple Sort '1' SampleSort or '2' Hyksort are valid"),
        };

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::Deterministic,
        );

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_svd_f64_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        singular_value_threshold: f64,
        surface_diff: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for Samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("Only '0' Simple Sort '1' SampleSort or '2' Hyksort are valid"),
        };

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::Deterministic,
        );

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_rsvd_f32_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        singular_value_threshold: f32,
        surface_diff: usize,
        n_components: usize,
        n_oversamples: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

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

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for Samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("Only '0' Simple Sort '1' SampleSort or '2' Hyksort are valid"),
        };

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_blas_rsvd_f64_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        singular_value_threshold: f64,
        surface_diff: usize,
        n_components: usize,
        n_oversamples: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

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

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for Samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("Only '0' Simple Sort '1' SampleSort or '2' Hyksort are valid"),
        };

        let field_translation = BlasFieldTranslationSaRcmp::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_fft_f32_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        block_size: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };
        println!("Got communicator: {:?}", communicator.as_raw());

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("only '0' simple sort '1' samplesort or '2' hyksort are valid"),
        };

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();


        println!("Built FMM at rank = {:?}", fmm.rank());

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
            data,
            ctype: FmmCType::Laplace32,
            ctranslation_type: FmmTranslationCType::Fft,
        };

        println!("Evaluated FMM at rank");


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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn laplace_fft_f64_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        block_size: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };
        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for Samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("Only '0' Simple Sort '1' SampleSort or '2' Hyksort are valid"),
        };

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Laplace3dKernel::new(),
                eval_type,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_rsvd_f32_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f32,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        singular_value_threshold: f32,
        surface_diff: usize,
        n_components: usize,
        n_oversamples: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

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

        let field_translation = BlasFieldTranslationIa::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("only '0' simple sort '1' samplesort or '2' hyksort are valid"),
        };

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    /// - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
    /// - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_rsvd_f64_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f64,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        singular_value_threshold: f64,
        surface_diff: usize,
        n_components: usize,
        n_oversamples: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };

        let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };
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

        let field_translation = BlasFieldTranslationIa::new(
            singular_value_threshold,
            surface_diff,
            crate::fmm::types::FmmSvdMode::new(true, None, n_components, n_oversamples, None),
        );

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("only '0' simple sort '1' samplesort or '2' hyksort are valid"),
        };

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
            data,
            ctype: FmmCType::Helmholtz64,
            ctranslation_type: FmmTranslationCType::Blas,
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_svd_f32_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f32,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        singular_value_threshold: f32,
        surface_diff: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };
        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("only '0' simple sort '1' samplesort or '2' hyksort are valid"),
        };

        let field_translation = BlasFieldTranslationIa::new(
            singular_value_threshold,
            surface_diff,
            FmmSvdMode::Deterministic,
        );

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
    /// - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_blas_svd_f64_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f64,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        singular_value_threshold: f64,
        surface_diff: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };
        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };

        let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let singular_value_threshold = Some(singular_value_threshold);
        let surface_diff = if surface_diff > 0 {
            Some(surface_diff)
        } else {
            None
        };

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("only '0' simple sort '1' samplesort or '2' hyksort are valid"),
        };

        let field_translation = BlasFieldTranslationIa::new(
            singular_value_threshold,
            surface_diff,
            FmmSvdMode::Deterministic,
        );

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
                field_translation,
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_fft_f32_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f32,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        block_size: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };
        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("only '0' simple sort '1' samplesort or '2' hyksort are valid"),
        };

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
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
    /// - `timed`: Modulates whether operators and metadata are timed.
    /// - `expansion_order`: A pointer to an array of expansion orders.
    /// - `n_expansion_order`: The number of expansion orders.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `wavenumber`: The wavenumber.
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
    /// - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
    ///  reached based on a uniform particle distribution.
    /// - `depth`: The maximum depth of the tree, max supported depth is 16.
    /// - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_fft_f64_mpi_alloc(
        timed: bool,
        expansion_order: *const usize,
        n_expansion_order: usize,
        eval_type: bool,
        wavenumber: f64,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        prune_empty: bool,
        local_depth: u64,
        global_depth: u64,
        block_size: usize,
        sort_kind: u64,
        n_samples: usize,
        communicator: *mut c_void,
    ) -> *mut FmmEvaluatorMPI {
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        let communicator = unsafe {
            mpi::topology::SimpleCommunicator::from_raw(communicator as mpi_sys::MPI_Comm)
        };

        let sources = unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
        let targets = unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
        let charges = unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

        let local_depth = if local_depth > 0 {
            local_depth
        } else {
            panic!("local_depth must be >= 1")
        };

        let global_depth = if global_depth > 0 {
            global_depth
        } else {
            panic!("global_depth must be >= 1")
        };

        let expansion_order =
            unsafe { std::slice::from_raw_parts(expansion_order, n_expansion_order) };

        let block_size = if block_size > 0 {
            Some(block_size)
        } else {
            None
        };

        let sort_kind = match sort_kind {
            0 => crate::tree::SortKind::Simplesort,
            1 => {
                if n_samples < 1 {
                    panic!("'n_samples' must be >= 1 for samplesort")
                } else {
                    crate::tree::SortKind::Samplesort { n_samples }
                }
            }
            2 => crate::tree::SortKind::Hyksort { subcomm_size: 2 },
            _ => panic!("only '0' simple sort '1' samplesort or '2' hyksort are valid"),
        };

        let fmm = MultiNodeBuilder::new(timed)
            .tree(
                &communicator,
                sources,
                targets,
                local_depth,
                global_depth,
                prune_empty,
                sort_kind,
            )
            .unwrap()
            .parameters(
                charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                eval_type,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        let data = Box::into_raw(Box::new(fmm)) as *mut c_void;

        let evaluator = FmmEvaluatorMPI {
            data,
            ctype: FmmCType::Helmholtz64,
            ctranslation_type: FmmTranslationCType::Fft,
        };

        Box::into_raw(Box::new(evaluator))
    }
}

/// FMM API
#[cfg(feature = "mpi")]
pub mod api_mpi {
    use std::{mem::ManuallyDrop, os::raw::c_void};

    use green_kernels::{
        helmholtz_3d::Helmholtz3dKernel, traits::Kernel, types::GreenKernelEvalType,
    };
    use itertools::Itertools;

    use crate::{
        bindings::{CommunicationEntry, FmmOperatorEntry, MetadataEntry, MetadataTimes},
        fmm::types::FmmEvalType,
        traits::{
            fmm::{ChargeHandler, DataAccessMulti},
            tree::{FmmTreeNode, MultiFmmTree, MultiTree, TreeNode},
        },
        tree::types::MortonKey,
        BlasFieldTranslationIa, EvaluateMulti, FftFieldTranslation, KiFmmMulti,
    };

    use super::{
        c32, c64, BlasFieldTranslationSaRcmp, CommunicationTimes, Coordinates, Expansion, FmmCType,
        FmmEvaluatorMPI, FmmOperatorTimes, FmmTranslationCType, GlobalIndices, Laplace3dKernel,
        MortonKeys, Potential, Potentials, ScalarType,
    };

    /// Get the communication runtimes
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
    pub unsafe extern "C" fn communication_times_mpi(
        fmm: *mut FmmEvaluatorMPI,
    ) -> *mut CommunicationTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;
                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },
        }
    }

    /// Get the metadata runtimes
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
    pub unsafe extern "C" fn metadata_times_mpi(fmm: *mut FmmEvaluatorMPI) -> *mut MetadataTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },
        }
    }

    /// Get the operator runtimes
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
    pub unsafe extern "C" fn operator_times_mpi(
        fmm: *mut FmmEvaluatorMPI,
    ) -> *mut FmmOperatorTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },
        }
    }

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
    pub unsafe extern "C" fn evaluate_mpi(fmm: *mut FmmEvaluatorMPI) -> *mut FmmOperatorTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;
                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },
        }
    }

    /// Clear charges and attach new charges.
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
    pub unsafe extern "C" fn clear_mpi(fmm: *mut FmmEvaluatorMPI) {
        if !fmm.is_null() {
            let ctype = unsafe { (*fmm).get_ctype() };
            let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
            let pointer = unsafe { (*fmm).get_pointer() };

            match ctype {
                FmmCType::Laplace32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;
                        unsafe { (*fmm).clear().unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                f32,
                                Laplace3dKernel<f32>,
                                BlasFieldTranslationSaRcmp<f32>,
                            >;

                        unsafe { (*fmm).clear().unwrap() };
                    }
                },

                FmmCType::Laplace64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;
                        unsafe { (*fmm).clear().unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                f64,
                                Laplace3dKernel<f64>,
                                BlasFieldTranslationSaRcmp<f64>,
                            >;

                        unsafe { (*fmm).clear().unwrap() };
                    }
                },

                FmmCType::Helmholtz32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c32,
                                Helmholtz3dKernel<c32>,
                                FftFieldTranslation<c32>,
                            >;

                        unsafe { (*fmm).clear().unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c32,
                                Helmholtz3dKernel<c32>,
                                BlasFieldTranslationIa<c32>,
                            >;
                        unsafe { (*fmm).clear().unwrap() };
                    }
                },

                FmmCType::Helmholtz64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c64,
                                Helmholtz3dKernel<c64>,
                                FftFieldTranslation<c64>,
                            >;

                        unsafe { (*fmm).clear().unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c64,
                                Helmholtz3dKernel<c64>,
                                BlasFieldTranslationIa<c64>,
                            >;

                        unsafe { (*fmm).clear().unwrap() };
                    }
                },
            }
        }
    }

    /// Attach new charges, in final Morton ordering
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `charges`: A pointer to the new charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn attach_charges_ordered_mpi(
        fmm: *mut FmmEvaluatorMPI,
        charges: *const c_void,
        n_charges: usize,
    ) {
        if !fmm.is_null() {
            let ctype = unsafe { (*fmm).get_ctype() };
            let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
            let pointer = unsafe { (*fmm).get_pointer() };

            match ctype {
                FmmCType::Laplace32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;
                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                f32,
                                Laplace3dKernel<f32>,
                                BlasFieldTranslationSaRcmp<f32>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }
                },

                FmmCType::Laplace64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                f64,
                                Laplace3dKernel<f64>,
                                BlasFieldTranslationSaRcmp<f64>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }
                },

                FmmCType::Helmholtz32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c32,
                                Helmholtz3dKernel<c32>,
                                FftFieldTranslation<c32>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c32,
                                Helmholtz3dKernel<c32>,
                                BlasFieldTranslationIa<c32>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }
                },

                FmmCType::Helmholtz64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c64,
                                Helmholtz3dKernel<c64>,
                                FftFieldTranslation<c64>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c64,
                                Helmholtz3dKernel<c64>,
                                BlasFieldTranslationIa<c64>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }
                },
            }
        }
    }

    /// Attach new charges, in initial input ordering before global Morton sort
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `charges`: A pointer to the new charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn attach_charges_unordered_mpi(
        fmm: *mut FmmEvaluatorMPI,
        charges: *const c_void,
        n_charges: usize,
    ) {
        if !fmm.is_null() {
            let ctype = unsafe { (*fmm).get_ctype() };
            let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
            let pointer = unsafe { (*fmm).get_pointer() };

            match ctype {
                FmmCType::Laplace32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;
                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                f32,
                                Laplace3dKernel<f32>,
                                BlasFieldTranslationSaRcmp<f32>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }
                },

                FmmCType::Laplace64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                f64,
                                Laplace3dKernel<f64>,
                                BlasFieldTranslationSaRcmp<f64>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }
                },

                FmmCType::Helmholtz32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c32,
                                Helmholtz3dKernel<c32>,
                                FftFieldTranslation<c32>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c32,
                                Helmholtz3dKernel<c32>,
                                BlasFieldTranslationIa<c32>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }
                },

                FmmCType::Helmholtz64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c64,
                                Helmholtz3dKernel<c64>,
                                FftFieldTranslation<c64>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmmMulti<
                                c64,
                                Helmholtz3dKernel<c64>,
                                BlasFieldTranslationIa<c64>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
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
    pub unsafe extern "C" fn all_potentials_mpi(fmm: *mut FmmEvaluatorMPI) -> *mut Potential {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
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
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
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
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len() * 2,
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C32,
                        }
                    };
                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len() * 2,
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
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len() * 2,
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C64,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len() * 2,
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C64,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }
            },
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
    pub unsafe extern "C" fn global_indices_target_tree_mpi(
        fmm: *mut FmmEvaluatorMPI,
    ) -> *mut GlobalIndices {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
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
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
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
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
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
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
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
    pub unsafe extern "C" fn global_indices_source_tree_mpi(
        fmm: *mut FmmEvaluatorMPI,
    ) -> *mut GlobalIndices {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
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
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
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
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
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
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
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

    /// Query for locals at a specific key.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `key`: The identifier of a node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn local_mpi(fmm: *mut FmmEvaluatorMPI, key: u64) -> *mut Expansion {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }
            },
        }
    }

    /// Query for multipoles at a specific key.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `key`: The identifier of a node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn multipole_mpi(fmm: *mut FmmEvaluatorMPI, key: u64) -> *mut Expansion {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
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
    pub unsafe extern "C" fn leaf_potentials_mpi(
        fmm: *mut FmmEvaluatorMPI,
        leaf: u64,
    ) -> *mut Potentials {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
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
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
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
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                                    len: p.len() * 2,
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
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
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
                                    len: p.len() * 2,
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
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                                    len: p.len() * 2,
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
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
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
                                    len: p.len() * 2,
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

    /// Construct a surface for a given key
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
    pub unsafe extern "C" fn surface_mpi(
        fmm: *mut FmmEvaluatorMPI,
        alpha: f64,
        expansion_order: u64,
        key: u64,
    ) -> *mut Coordinates {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let key = MortonKey::from_morton(key);

                    let alpha = alpha as f32;
                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(surface))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let alpha = alpha as f32;
                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(surface))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let key = MortonKey::from_morton(key);

                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let key = MortonKey::from_morton(key);

                    let alpha = alpha as f32;
                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let alpha = alpha as f32;
                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let key = MortonKey::from_morton(key);

                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let key = MortonKey::from_morton(key);

                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }
            },
        }
    }

    /// Query target tree for local depth
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
    pub unsafe extern "C" fn target_tree_local_depth_mpi(fmm: *mut FmmEvaluatorMPI) -> u64 {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    (*fmm).tree().target_tree().local_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    (*fmm).tree().target_tree().local_depth()
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    (*fmm).tree().target_tree().local_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    (*fmm).tree().target_tree().local_depth()
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    (*fmm).tree().target_tree().local_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    (*fmm).tree().target_tree().local_depth()
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    (*fmm).tree().target_tree().local_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    (*fmm).tree().target_tree().local_depth()
                }
            },
        }
    }

    /// Query target tree for global depth
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
    pub unsafe extern "C" fn target_tree_global_depth_mpi(fmm: *mut FmmEvaluatorMPI) -> u64 {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    (*fmm).tree().target_tree().global_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    (*fmm).tree().target_tree().global_depth()
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    (*fmm).tree().target_tree().global_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    (*fmm).tree().target_tree().global_depth()
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    (*fmm).tree().target_tree().global_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    (*fmm).tree().target_tree().global_depth()
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    (*fmm).tree().target_tree().global_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    (*fmm).tree().target_tree().global_depth()
                }
            },
        }
    }

    /// Query source tree for local depth
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
    pub unsafe extern "C" fn source_tree_local_depth_mpi(fmm: *mut FmmEvaluatorMPI) -> u64 {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    (*fmm).tree().source_tree().local_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    (*fmm).tree().source_tree().local_depth()
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    (*fmm).tree().source_tree().local_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    (*fmm).tree().source_tree().local_depth()
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    (*fmm).tree().source_tree().local_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    (*fmm).tree().source_tree().local_depth()
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    (*fmm).tree().source_tree().local_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    (*fmm).tree().source_tree().local_depth()
                }
            },
        }
    }

    /// Query target tree for global depth
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
    pub unsafe extern "C" fn source_tree_global_depth_mpi(fmm: *mut FmmEvaluatorMPI) -> u64 {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    (*fmm).tree().source_tree().global_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    (*fmm).tree().source_tree().global_depth()
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    (*fmm).tree().source_tree().global_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    (*fmm).tree().source_tree().global_depth()
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    (*fmm).tree().source_tree().global_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    (*fmm).tree().source_tree().global_depth()
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    (*fmm).tree().source_tree().global_depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    (*fmm).tree().source_tree().global_depth()
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
    pub unsafe extern "C" fn coordinates_source_tree_mpi(
        fmm: *mut FmmEvaluatorMPI,
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
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
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
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
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
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
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
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
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
    pub unsafe extern "C" fn coordinates_target_tree_mpi(
        fmm: *mut FmmEvaluatorMPI,
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
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
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
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
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
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
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
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
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

    /// Query source tree for all keys
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
    pub unsafe extern "C" fn keys_source_tree_mpi(fmm: *mut FmmEvaluatorMPI) -> *mut MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let keys: Vec<u64> = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },
        }
    }

    /// Query target tree for all keys
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
    pub unsafe extern "C" fn keys_target_tree_mpi(fmm: *mut FmmEvaluatorMPI) -> *mut MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let keys: Vec<u64> = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },
        }
    }

    /// Query target tree for leaves.
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
    pub unsafe extern "C" fn leaves_target_tree_mpi(fmm: *mut FmmEvaluatorMPI) -> *mut MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
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
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
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
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
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
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
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
    pub unsafe extern "C" fn leaves_source_tree_mpi(fmm: *mut FmmEvaluatorMPI) -> *mut MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
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
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
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
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
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
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
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

    /// Evaluate the kernel in single threaded mode
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `result`: A pointer to the results associated with the target points.
    /// - `n_charges`: The length of the charges buffer.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn evaluate_kernel_st_mpi(
        fmm: *mut FmmEvaluatorMPI,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        result: *mut c_void,
        nresult: usize,
    ) {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }
            },
        }
    }

    /// Evaluate the kernel in multithreaded mode
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `result`: A pointer to the results associated with the target points.
    /// - `n_charges`: The length of the charges buffer.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn evaluate_kernel_mt_mpi(
        fmm: *mut FmmEvaluatorMPI,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        result: *mut c_void,
        nresult: usize,
    ) {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f32,
                            Laplace3dKernel<f32>,
                            BlasFieldTranslationSaRcmp<f32>,
                        >;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            f64,
                            Laplace3dKernel<f64>,
                            BlasFieldTranslationSaRcmp<f64>,
                        >;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c32,
                            Helmholtz3dKernel<c32>,
                            BlasFieldTranslationIa<c32>,
                        >;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmmMulti<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmmMulti<
                            c64,
                            Helmholtz3dKernel<c64>,
                            BlasFieldTranslationIa<c64>,
                        >;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }
            },
        }
    }
}

/// FMM API
pub mod api {
    use std::{mem::ManuallyDrop, os::raw::c_void};

    use green_kernels::{
        helmholtz_3d::Helmholtz3dKernel, traits::Kernel, types::GreenKernelEvalType,
    };
    use itertools::Itertools;

    use crate::{
        bindings::{CommunicationEntry, FmmOperatorEntry, MetadataEntry, MetadataTimes},
        fmm::types::FmmEvalType,
        traits::{
            fmm::{ChargeHandler, DataAccess},
            tree::{FmmTreeNode, SingleFmmTree, SingleTree, TreeNode},
        },
        tree::types::MortonKey,
        BlasFieldTranslationIa, Evaluate, FftFieldTranslation, KiFmm,
    };

    use super::{
        c32, c64, BlasFieldTranslationSaRcmp, CommunicationTimes, Coordinates, Expansion, FmmCType,
        FmmEvaluator, FmmOperatorTimes, FmmTranslationCType, GlobalIndices, Laplace3dKernel,
        MortonKeys, Potential, Potentials, ScalarType,
    };

    /// Get the communication runtimes
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
    pub unsafe extern "C" fn communication_times(
        fmm: *mut FmmEvaluator,
    ) -> *mut CommunicationTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;
                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let entries: Vec<CommunicationEntry> = unsafe {
                        (*fmm)
                            .communication_times
                            .iter()
                            .map(|(&comm_type, &time)| CommunicationEntry { comm_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(CommunicationTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },
        }
    }

    /// Get the metadata runtimes
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
    pub unsafe extern "C" fn metadata_times(fmm: *mut FmmEvaluator) -> *mut MetadataTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let entries: Vec<MetadataEntry> = unsafe {
                        (*fmm)
                            .metadata_times
                            .iter()
                            .map(|(&metadata_type, &time)| MetadataEntry {
                                metadata_type,
                                time,
                            })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(MetadataTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },
        }
    }

    /// Get the operator runtimes
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
    pub unsafe extern "C" fn operator_times(fmm: *mut FmmEvaluator) -> *mut FmmOperatorTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },
        }
    }

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
    pub unsafe extern "C" fn evaluate(fmm: *mut FmmEvaluator) -> *mut FmmOperatorTimes {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;
                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let _ = unsafe { (*fmm).evaluate() };

                    let entries: Vec<FmmOperatorEntry> = unsafe {
                        (*fmm)
                            .operator_times
                            .iter()
                            .map(|(&op_type, &time)| FmmOperatorEntry { op_type, time })
                            .collect()
                    };

                    // Leak the Vec to pass to C safely
                    let length = entries.len();
                    let mut boxed_entries = entries.into_boxed_slice();
                    let ptr = boxed_entries.as_mut_ptr();
                    std::mem::forget(boxed_entries); // Prevent Rust from freeing it

                    let boxed = Box::new(FmmOperatorTimes { times: ptr, length });

                    Box::into_raw(boxed)
                }
            },
        }
    }

    /// Clear charges and attach new charges.
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
    pub unsafe extern "C" fn clear(fmm: *mut FmmEvaluator) {
        if !fmm.is_null() {
            let ctype = unsafe { (*fmm).get_ctype() };
            let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
            let pointer = unsafe { (*fmm).get_pointer() };

            match ctype {
                FmmCType::Laplace32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;
                        unsafe { (*fmm).clear().unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<
                                f32,
                                Laplace3dKernel<f32>,
                                BlasFieldTranslationSaRcmp<f32>,
                            >;

                        unsafe { (*fmm).clear().unwrap() };
                    }
                },

                FmmCType::Laplace64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;
                        unsafe { (*fmm).clear().unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<
                                f64,
                                Laplace3dKernel<f64>,
                                BlasFieldTranslationSaRcmp<f64>,
                            >;

                        unsafe { (*fmm).clear().unwrap() };
                    }
                },

                FmmCType::Helmholtz32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                        unsafe { (*fmm).clear().unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;
                        unsafe { (*fmm).clear().unwrap() };
                    }
                },

                FmmCType::Helmholtz64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                        unsafe { (*fmm).clear().unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                        unsafe { (*fmm).clear().unwrap() };
                    }
                },
            }
        }
    }

    /// Attach new charges, in final Morton ordering
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `charges`: A pointer to the new charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn attach_charges_ordered(
        fmm: *mut FmmEvaluator,
        charges: *const c_void,
        n_charges: usize,
    ) {
        if !fmm.is_null() {
            let ctype = unsafe { (*fmm).get_ctype() };
            let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
            let pointer = unsafe { (*fmm).get_pointer() };

            match ctype {
                FmmCType::Laplace32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;
                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<
                                f32,
                                Laplace3dKernel<f32>,
                                BlasFieldTranslationSaRcmp<f32>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }
                },

                FmmCType::Laplace64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<
                                f64,
                                Laplace3dKernel<f64>,
                                BlasFieldTranslationSaRcmp<f64>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }
                },

                FmmCType::Helmholtz32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }
                },

                FmmCType::Helmholtz64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };
                        unsafe { (*fmm).attach_charges_ordered(charges).unwrap() };
                    }
                },
            }
        }
    }

    /// Attach new charges, in initial input ordering before global Morton sort
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `charges`: A pointer to the new charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn attach_charges_unordered(
        fmm: *mut FmmEvaluator,
        charges: *const c_void,
        n_charges: usize,
    ) {
        if !fmm.is_null() {
            let ctype = unsafe { (*fmm).get_ctype() };
            let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
            let pointer = unsafe { (*fmm).get_pointer() };

            match ctype {
                FmmCType::Laplace32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;
                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<
                                f32,
                                Laplace3dKernel<f32>,
                                BlasFieldTranslationSaRcmp<f32>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }
                },

                FmmCType::Laplace64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<
                                f64,
                                Laplace3dKernel<f64>,
                                BlasFieldTranslationSaRcmp<f64>,
                            >;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }
                },

                FmmCType::Helmholtz32 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }
                },

                FmmCType::Helmholtz64 => match ctranslation_type {
                    FmmTranslationCType::Fft => {
                        let fmm = pointer
                            as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
                    }

                    FmmTranslationCType::Blas => {
                        let fmm = pointer
                            as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                        let charges =
                            unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };
                        unsafe { (*fmm).attach_charges_unordered(charges).unwrap() };
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
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len() * 2,
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C32,
                        }
                    };
                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len() * 2,
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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len() * 2,
                            data: (*fmm).potentials.as_ptr() as *const c_void,
                            scalar: ScalarType::C64,
                        }
                    };

                    Box::into_raw(Box::new(potentials))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let potentials = unsafe {
                        Potential {
                            len: (*fmm).potentials.len() * 2,
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
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

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

    /// Query for locals at a specific key.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `key`: The identifier of a node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn local(fmm: *mut FmmEvaluator, key: u64) -> *mut Expansion {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).local(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }
            },
        }
    }

    /// Query for multipoles at a specific key.
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `key`: The identifier of a node.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn multipole(fmm: *mut FmmEvaluator, key: u64) -> *mut Expansion {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len();
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(m))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let key = MortonKey::from_morton(key);

                    let len;
                    let ptr;
                    unsafe {
                        if let Some(m) = (*fmm).multipole(&key) {
                            len = m.len() * 2;
                            ptr = m.as_ptr()
                        } else {
                            len = 0;
                            ptr = std::ptr::null()
                        }
                    }

                    let data = ptr as *const c_void;

                    let m = Expansion {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(m))
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
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                                    len: p.len() * 2,
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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

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
                                    len: p.len() * 2,
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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                                    len: p.len() * 2,
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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

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
                                    len: p.len() * 2,
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

    /// Query key for level
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `key`: The identifier of the key.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn level(key: u64) -> u64 {
        let key = MortonKey::<f32>::from_morton(key);
        key.level()
    }

    /// Construct a surface for a given key
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
    pub unsafe extern "C" fn surface(
        fmm: *mut FmmEvaluator,
        alpha: f64,
        expansion_order: u64,
        key: u64,
    ) -> *mut Coordinates {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let key = MortonKey::from_morton(key);

                    let alpha = alpha as f32;
                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(surface))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let key = MortonKey::from_morton(key);

                    let alpha = alpha as f32;
                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F32,
                    };

                    Box::into_raw(Box::new(surface))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let key = MortonKey::from_morton(key);

                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let key = MortonKey::from_morton(key);

                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let key = MortonKey::from_morton(key);

                    let alpha = alpha as f32;
                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let key = MortonKey::from_morton(key);

                    let alpha = alpha as f32;
                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let key = MortonKey::from_morton(key);

                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let key = MortonKey::from_morton(key);

                    let surface =
                        key.surface_grid(expansion_order as usize, (*fmm).tree().domain(), alpha);
                    let len = surface.len();
                    let ptr = surface.as_ptr();

                    let data = ptr as *const c_void;

                    let surface = Coordinates {
                        len,
                        data,
                        scalar: ScalarType::F64,
                    };

                    Box::into_raw(Box::new(surface))
                }
            },
        }
    }

    /// Query target tree for depth
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
    pub unsafe extern "C" fn target_tree_depth(fmm: *mut FmmEvaluator) -> u64 {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    (*fmm).tree().target_tree().depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    (*fmm).tree().target_tree().depth()
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    (*fmm).tree().target_tree().depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    (*fmm).tree().target_tree().depth()
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    (*fmm).tree().target_tree().depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    (*fmm).tree().target_tree().depth()
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    (*fmm).tree().target_tree().depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    (*fmm).tree().target_tree().depth()
                }
            },
        }
    }

    /// Query source tree for depth
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
    pub unsafe extern "C" fn source_tree_depth(fmm: *mut FmmEvaluator) -> u64 {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    (*fmm).tree().source_tree().depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    (*fmm).tree().source_tree().depth()
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    (*fmm).tree().source_tree().depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    (*fmm).tree().source_tree().depth()
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    (*fmm).tree().source_tree().depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    (*fmm).tree().source_tree().depth()
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    (*fmm).tree().source_tree().depth()
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    (*fmm).tree().source_tree().depth()
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
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

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

    /// Query source tree for all keys
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
    pub unsafe extern "C" fn keys_source_tree(fmm: *mut FmmEvaluator) -> *mut MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let keys: Vec<u64> = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .source_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },
        }
    }

    /// Query target tree for all keys
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
    pub unsafe extern "C" fn keys_target_tree(fmm: *mut FmmEvaluator) -> *mut MortonKeys {
        assert!(!fmm.is_null());
        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let keys: Vec<u64> = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let keys = unsafe {
                        (*fmm)
                            .tree()
                            .target_tree()
                            .keys
                            .iter()
                            .map(|l| l.raw())
                            .collect_vec()
                    };

                    let keys = ManuallyDrop::new(keys);

                    let keys = MortonKeys {
                        len: keys.len(),
                        data: keys.as_ptr(),
                    };

                    Box::into_raw(Box::new(keys))
                }
            },
        }
    }

    /// Query target tree for leaves.
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
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

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
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

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
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

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
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

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
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

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
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

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

    /// Evaluate the kernel in single threaded mode
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `result`: A pointer to the results associated with the target points.
    /// - `n_charges`: The length of the charges buffer.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn evaluate_kernel_st(
        fmm: *mut FmmEvaluator,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        result: *mut c_void,
        nresult: usize,
    ) {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_st(eval_type, sources, targets, charges, result)
                    };
                }
            },
        }
    }

    /// Evaluate the kernel in multithreaded mode
    ///
    /// # Parameters
    ///
    /// - `fmm`: Pointer to an `FmmEvaluator` instance.
    /// - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
    /// - `sources`: A pointer to the source points.
    /// - `n_sources`: The length of the source points buffer
    /// - `targets`: A pointer to the target points.
    /// - `n_targets`: The length of the target points buffer.
    /// - `charges`: A pointer to the charges associated with the source points.
    /// - `n_charges`: The length of the charges buffer.
    /// - `result`: A pointer to the results associated with the target points.
    /// - `n_charges`: The length of the charges buffer.
    ///
    /// # Safety
    /// This function is intended to be called from C. The caller must ensure that:
    /// - Input data corresponds to valid pointers
    /// - That they remain valid for the duration of the function call
    #[no_mangle]
    pub unsafe extern "C" fn evaluate_kernel_mt(
        fmm: *mut FmmEvaluator,
        eval_type: bool,
        sources: *const c_void,
        n_sources: usize,
        targets: *const c_void,
        n_targets: usize,
        charges: *const c_void,
        n_charges: usize,
        result: *mut c_void,
        nresult: usize,
    ) {
        assert!(!fmm.is_null());

        let ctype = unsafe { (*fmm).get_ctype() };
        let ctranslation_type = unsafe { (*fmm).get_ctranslation_type() };
        let pointer = unsafe { (*fmm).get_pointer() };
        let eval_type = if eval_type {
            GreenKernelEvalType::Value
        } else {
            GreenKernelEvalType::ValueDeriv
        };

        match ctype {
            FmmCType::Laplace32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Laplace64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm =
                        pointer as *mut KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut f64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const f64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Helmholtz32 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f32, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f32, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c32, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c32, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }
            },

            FmmCType::Helmholtz64 => match ctranslation_type {
                FmmTranslationCType::Fft => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }

                FmmTranslationCType::Blas => {
                    let fmm = pointer
                        as *mut KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>;

                    let sources =
                        unsafe { std::slice::from_raw_parts(sources as *const f64, n_sources) };
                    let targets =
                        unsafe { std::slice::from_raw_parts(targets as *const f64, n_targets) };
                    let result =
                        unsafe { std::slice::from_raw_parts_mut(result as *mut c64, nresult) };
                    let charges =
                        unsafe { std::slice::from_raw_parts(charges as *const c64, n_charges) };

                    unsafe {
                        (*fmm)
                            .kernel()
                            .evaluate_mt(eval_type, sources, targets, charges, result)
                    };
                }
            },
        }
    }
}

pub use api::*;
pub use constructors::*;
pub use types::*;

#[cfg(feature = "mpi")]
pub use api_mpi::*;

#[cfg(feature = "mpi")]
pub use mpi_types::*;

#[cfg(feature = "mpi")]
pub use constructors_mpi::*;

#[cfg(test)]
mod test {
    use std::ffi::c_void;

    use num::{Complex, One};
    use rlst::RawAccess;

    use crate::tree::helpers::points_fixture;

    use super::{
        evaluate, free_fmm_evaluator, helmholtz_blas_svd_f32_alloc, helmholtz_blas_svd_f64_alloc,
        helmholtz_fft_f32_alloc, helmholtz_fft_f64_alloc, laplace_blas_rsvd_f32_alloc,
        laplace_blas_rsvd_f64_alloc, laplace_blas_svd_f32_alloc, laplace_blas_svd_f64_alloc,
        laplace_fft_f32_alloc, laplace_fft_f64_alloc,
    };

    #[test]
    fn test_raw_laplace_constructors() {
        // f32
        {
            let timed = false;
            let n_points = 1000;
            let sources = points_fixture::<f32>(n_points, None, None, None);
            let targets = points_fixture::<f32>(n_points, None, None, None);
            let charges = vec![1.0; n_points];

            let n_sources = n_points * 3;
            let sources_p = sources.data().as_ptr() as *const c_void;

            let n_targets = n_points * 3;
            let targets_p = targets.data().as_ptr() as *const c_void;

            let n_charges = n_points;
            let charges_p = charges.as_ptr() as *const c_void;

            let expansion_order = [6usize];
            let expansion_order_p = expansion_order.as_ptr();
            let n_expansion_order = 1;

            let laplace_blas_rsvd_f32 = unsafe {
                laplace_blas_rsvd_f32_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    1e-10,
                    2,
                    10,
                    10,
                )
            };

            let laplace_blas_svd_f32 = unsafe {
                laplace_blas_svd_f32_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    1e-10,
                    2,
                )
            };

            let laplace_fft_f32 = unsafe {
                laplace_fft_f32_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    32,
                )
            };

            unsafe { evaluate(laplace_blas_rsvd_f32) };
            unsafe { evaluate(laplace_blas_svd_f32) };
            unsafe { evaluate(laplace_fft_f32) };
            unsafe { free_fmm_evaluator(laplace_blas_rsvd_f32) };
            unsafe { free_fmm_evaluator(laplace_blas_svd_f32) };
            unsafe { free_fmm_evaluator(laplace_fft_f32) };
        }

        // f64
        {
            let timed = false;
            let n_points = 1000;
            let sources = points_fixture::<f64>(n_points, None, None, None);
            let targets = points_fixture::<f64>(n_points, None, None, None);
            let charges = vec![1.0; n_points];

            let n_sources = n_points * 3;
            let sources_p = sources.data().as_ptr() as *const c_void;

            let n_targets = n_points * 3;
            let targets_p = targets.data().as_ptr() as *const c_void;

            let n_charges = n_points;
            let charges_p = charges.as_ptr() as *const c_void;

            let expansion_order = [6usize];
            let expansion_order_p = expansion_order.as_ptr();
            let n_expansion_order = 1;

            let laplace_blas_rsvd_f64 = unsafe {
                laplace_blas_rsvd_f64_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    1e-10,
                    2,
                    10,
                    10,
                )
            };

            let laplace_blas_svd_f64 = unsafe {
                laplace_blas_svd_f64_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    1e-10,
                    2,
                )
            };

            let laplace_fft_f64 = unsafe {
                laplace_fft_f64_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    32,
                )
            };

            unsafe { evaluate(laplace_blas_rsvd_f64) };
            unsafe { evaluate(laplace_blas_svd_f64) };
            unsafe { evaluate(laplace_fft_f64) };

            unsafe { free_fmm_evaluator(laplace_blas_rsvd_f64) };
            unsafe { free_fmm_evaluator(laplace_blas_svd_f64) };
            unsafe { free_fmm_evaluator(laplace_fft_f64) };
        }
    }

    #[test]
    fn test_raw_helmholtz_constructors() {
        // f32
        {
            let timed = false;
            let n_points = 1000;
            let sources = points_fixture::<f32>(n_points, None, None, None);
            let targets = points_fixture::<f32>(n_points, None, None, None);
            let charges = vec![Complex::<f32>::one(); n_points];
            let wavenumber = 10.;

            let n_sources = n_points * 3;
            let sources_p = sources.data().as_ptr() as *const c_void;

            let n_targets = n_points * 3;
            let targets_p = targets.data().as_ptr() as *const c_void;

            let n_charges = n_points;
            let charges_p = charges.as_ptr() as *const c_void;

            let expansion_order = [6usize];
            let expansion_order_p = expansion_order.as_ptr();
            let n_expansion_order = 1;

            let helmholtz_blas_svd_f32 = unsafe {
                helmholtz_blas_svd_f32_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    wavenumber,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    1e-10,
                    2,
                )
            };

            let helmholtz_fft_f32 = unsafe {
                helmholtz_fft_f32_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    wavenumber,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    32,
                )
            };

            unsafe { evaluate(helmholtz_blas_svd_f32) };
            unsafe { evaluate(helmholtz_fft_f32) };
            unsafe { free_fmm_evaluator(helmholtz_blas_svd_f32) };
            unsafe { free_fmm_evaluator(helmholtz_fft_f32) };
        }

        // f64
        {
            let timed = false;
            let n_points = 1000;
            let sources = points_fixture::<f64>(n_points, None, None, None);
            let targets = points_fixture::<f64>(n_points, None, None, None);
            let charges = vec![Complex::<f64>::one(); n_points];
            let wavenumber = 10.;

            let n_sources = n_points * 3;
            let sources_p = sources.data().as_ptr() as *const c_void;

            let n_targets = n_points * 3;
            let targets_p = targets.data().as_ptr() as *const c_void;

            let n_charges = n_points * 2;
            let charges_p = charges.as_ptr() as *const c_void;

            let expansion_order = [6usize];
            let expansion_order_p = expansion_order.as_ptr();
            let n_expansion_order = 1;

            let helmholtz_blas_svd_f64 = unsafe {
                helmholtz_blas_svd_f64_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    wavenumber,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    1e-10,
                    2,
                )
            };

            let helmholtz_fft_f64 = unsafe {
                helmholtz_fft_f64_alloc(
                    timed,
                    expansion_order_p,
                    n_expansion_order,
                    true,
                    wavenumber,
                    sources_p,
                    n_sources,
                    targets_p,
                    n_targets,
                    charges_p,
                    n_charges,
                    true,
                    150,
                    0,
                    32,
                )
            };

            unsafe { evaluate(helmholtz_blas_svd_f64) };
            unsafe { evaluate(helmholtz_fft_f64) };
            unsafe { free_fmm_evaluator(helmholtz_blas_svd_f64) };
            unsafe { free_fmm_evaluator(helmholtz_fft_f64) };
        }
    }
}
