//! Python bindings for constructors and basic methods
use std::collections::HashMap;

use crate::fmm::KiFmm;
use crate::traits::fmm::Fmm;
use crate::traits::tree::Tree;
use crate::tree::types::MortonKey;
use crate::{
    BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder,
};
use green_kernels::helmholtz_3d::Helmholtz3dKernel;
use green_kernels::traits::Kernel;
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use numpy::{
    ndarray::Dim, npyffi::NPY_ORDER, PyArray, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use rlst::{
    c32, c64, rlst_array_from_slice2, rlst_dynamic_array2, RawAccess, RawAccessMut, RlstScalar,
};

use pyo3::{pymodule, types::PyModule, Bound, PyResult};

macro_rules! define_pyclass {
    ($name: ident, $type: ident, $kernel: ident, $field_translation: ident) => {
        /// Python interface
        #[pyclass]
        pub struct $name {
            fmm: KiFmm<$type, $kernel<$type>, $field_translation<$type>>,
            source_keys: Vec<u64>,
            target_keys: Vec<u64>,
            source_leaves: Vec<u64>,
            target_leaves: Vec<u64>,
            source_key_map: HashMap<u64, MortonKey<<$type as RlstScalar>::Real>>,
            target_key_map: HashMap<u64, MortonKey<<$type as RlstScalar>::Real>>,
        }
    };
}

define_pyclass!(LaplaceFft64, f64, Laplace3dKernel, FftFieldTranslation);
define_pyclass!(LaplaceFft32, f32, Laplace3dKernel, FftFieldTranslation);
define_pyclass!(
    LaplaceBlas64,
    f64,
    Laplace3dKernel,
    BlasFieldTranslationSaRcmp
);
define_pyclass!(
    LaplaceBlas32,
    f32,
    Laplace3dKernel,
    BlasFieldTranslationSaRcmp
);

define_pyclass!(HelmholtzFft64, c64, Helmholtz3dKernel, FftFieldTranslation);
define_pyclass!(HelmholtzFft32, c32, Helmholtz3dKernel, FftFieldTranslation);
define_pyclass!(
    HelmholtzBlas64,
    c64,
    Helmholtz3dKernel,
    BlasFieldTranslationIa
);
define_pyclass!(
    HelmholtzBlas32,
    c32,
    Helmholtz3dKernel,
    BlasFieldTranslationIa
);

macro_rules! laplace_fft_constructors {
    ($name: ident, $type: ident) => {
        #[pymethods]
        impl $name {
            /// Constructor
            #[new]
            #[allow(clippy::too_many_arguments)]
            pub fn new<'py>(
                expansion_order: usize,
                sources: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                targets: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                charges: PyReadonlyArrayDyn<'py, $type>,
                n_crit: u64,
                sparse: bool,
                kernel_eval_type: usize,
            ) -> PyResult<Self> {
                let kernel_eval_type = if kernel_eval_type == 0 {
                    EvalType::Value
                } else if kernel_eval_type == 1 {
                    EvalType::ValueDeriv
                } else {
                    return Err(PyErr::new::<PyTypeError, _>(
                        "Invalid Kernel Evaluation Mode",
                    ));
                };

                let shape = sources.shape();
                let sources_slice =
                    rlst_array_from_slice2!(sources.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut sources_arr =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [shape[0], shape[1]]);
                let p1 = sources_slice.data().as_ptr();
                let p2 = sources_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, sources.as_slice().unwrap().len());
                }

                let shape = targets.shape();
                let targets_slice =
                    rlst_array_from_slice2!(targets.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut targets_arr =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [shape[0], shape[1]]);
                let p1 = targets_slice.data().as_ptr();
                let p2 = targets_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, targets_slice.data().len());
                }

                let shape = charges.shape();

                if shape[1] > 1 {
                    return Err(PyErr::new::<PyTypeError, _>(
                        "Multipole charge vectors not supported with FFT field translation",
                    ));
                }

                let charges_slice =
                    rlst_array_from_slice2!(charges.as_slice().unwrap(), [shape[0], 1]);
                let mut charges_arr = rlst_dynamic_array2!($type, [shape[0], 1]);
                let p1 = charges_slice.data().as_ptr();
                let p2 = charges_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, charges_slice.data().len());
                }

                let fmm = SingleNodeBuilder::new()
                    .tree(&sources_arr, &targets_arr, Some(n_crit), sparse)
                    .unwrap()
                    .parameters(
                        &charges_arr,
                        expansion_order,
                        Laplace3dKernel::new(),
                        kernel_eval_type,
                        FftFieldTranslation::new(),
                    )
                    .unwrap()
                    .build()
                    .unwrap();

                let mut source_key_map = HashMap::new();
                let mut source_keys = Vec::new();
                let mut source_leaves = Vec::new();

                for key in fmm.tree.source_tree.all_keys().unwrap() {
                    source_key_map.insert(
                        key.morton(),
                        MortonKey::<<$type as RlstScalar>::Real>::from(key.clone()),
                    );
                    source_keys.push(key.morton);
                    if fmm.tree.source_tree.leaves_set.contains(&key) {
                        source_leaves.push(key.morton)
                    }
                }

                let mut target_key_map = HashMap::new();
                let mut target_keys = Vec::new();
                let mut target_leaves = Vec::new();
                for key in fmm.tree.target_tree.all_keys().unwrap() {
                    target_key_map.insert(
                        key.morton(),
                        MortonKey::<<$type as RlstScalar>::Real>::from(key.clone()),
                    );
                    target_keys.push(key.morton);
                    if fmm.tree.target_tree.leaves_set.contains(&key) {
                        target_leaves.push(key.morton)
                    }
                }

                Ok(Self {
                    fmm,
                    source_key_map,
                    target_key_map,
                    target_keys,
                    source_keys,
                    target_leaves,
                    source_leaves,
                })
            }
        }
    };
}

macro_rules! laplace_blas_constructors {
    ($name: ident, $type: ident) => {
        #[pymethods]
        impl $name {
            /// Constructor
            #[new]
            #[allow(clippy::too_many_arguments)]
            pub fn new<'py>(
                expansion_order: usize,
                sources: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                targets: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                charges: PyReadonlyArrayDyn<'py, $type>,
                n_crit: u64,
                sparse: bool,
                kernel_eval_type: usize,
                svd_threshold: <$type as RlstScalar>::Real,
            ) -> PyResult<Self> {
                let kernel_eval_type = if kernel_eval_type == 0 {
                    EvalType::Value
                } else if kernel_eval_type == 1 {
                    EvalType::ValueDeriv
                } else {
                    return Err(PyErr::new::<PyTypeError, _>(
                        "Invalid Kernel Evaluation Mode",
                    ));
                };

                let shape = sources.shape();
                let sources_slice =
                    rlst_array_from_slice2!(sources.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut sources_arr =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [shape[0], shape[1]]);
                let p1 = sources_slice.data().as_ptr();
                let p2 = sources_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, sources.as_slice().unwrap().len());
                }

                let shape = targets.shape();
                let targets_slice =
                    rlst_array_from_slice2!(targets.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut targets_arr =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [shape[0], shape[1]]);
                let p1 = targets_slice.data().as_ptr();
                let p2 = targets_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, targets_slice.data().len());
                }

                let shape = charges.shape();
                let charges_slice =
                    rlst_array_from_slice2!(charges.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut charges_arr = rlst_dynamic_array2!($type, [shape[0], shape[1]]);
                let p1 = charges_slice.data().as_ptr();
                let p2 = charges_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, charges_slice.data().len());
                }

                let fmm = SingleNodeBuilder::new()
                    .tree(&sources_arr, &targets_arr, Some(n_crit), sparse)
                    .unwrap()
                    .parameters(
                        &charges_arr,
                        expansion_order,
                        Laplace3dKernel::new(),
                        kernel_eval_type,
                        BlasFieldTranslationSaRcmp::new(Some(svd_threshold)),
                    )
                    .unwrap()
                    .build()
                    .unwrap();

                let mut source_key_map = HashMap::new();
                let mut source_keys = Vec::new();
                let mut source_leaves = Vec::new();

                for key in fmm.tree.source_tree.all_keys().unwrap() {
                    source_key_map.insert(
                        key.morton(),
                        MortonKey::<<$type as RlstScalar>::Real>::from(key.clone()),
                    );
                    source_keys.push(key.morton);
                    if fmm.tree.source_tree.leaves_set.contains(&key) {
                        source_leaves.push(key.morton)
                    }
                }

                let mut target_key_map = HashMap::new();
                let mut target_keys = Vec::new();
                let mut target_leaves = Vec::new();
                for key in fmm.tree.target_tree.all_keys().unwrap() {
                    target_key_map.insert(
                        key.morton(),
                        MortonKey::<<$type as RlstScalar>::Real>::from(key.clone()),
                    );
                    target_keys.push(key.morton);
                    if fmm.tree.target_tree.leaves_set.contains(&key) {
                        target_leaves.push(key.morton)
                    }
                }

                Ok(Self {
                    fmm,
                    source_key_map,
                    target_key_map,
                    target_keys,
                    source_keys,
                    target_leaves,
                    source_leaves,
                })
            }
        }
    };
}

macro_rules! helmholtz_fft_constructors {
    ($name: ident, $type: ident) => {
        #[pymethods]
        impl $name {
            /// Constructor for Helmholtz KiFmm with FFT Field translation
            #[new]
            #[allow(clippy::too_many_arguments)]
            pub fn new<'py>(
                expansion_order: usize,
                sources: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                targets: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                charges: PyReadonlyArrayDyn<'py, $type>,
                n_crit: u64,
                sparse: bool,
                kernel_eval_type: usize,
                wavenumber: <$type as RlstScalar>::Real,
            ) -> PyResult<Self> {
                let kernel_eval_type = if kernel_eval_type == 0 {
                    EvalType::Value
                } else if kernel_eval_type == 1 {
                    EvalType::ValueDeriv
                } else {
                    return Err(PyErr::new::<PyTypeError, _>(
                        "Invalid Kernel Evaluation Mode",
                    ));
                };

                let shape = sources.shape();
                let sources_slice =
                    rlst_array_from_slice2!(sources.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut sources_arr =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [shape[0], shape[1]]);
                let p1 = sources_slice.data().as_ptr();
                let p2 = sources_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, sources.as_slice().unwrap().len());
                }

                let shape = targets.shape();
                let targets_slice =
                    rlst_array_from_slice2!(targets.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut targets_arr =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [shape[0], shape[1]]);
                let p1 = targets_slice.data().as_ptr();
                let p2 = targets_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, targets_slice.data().len());
                }

                let shape = charges.shape();

                if shape[1] > 1 {
                    return Err(PyErr::new::<PyTypeError, _>(
                        "Multipole charge vectors not supported with FFT field translation",
                    ));
                }

                let charges_slice =
                    rlst_array_from_slice2!(charges.as_slice().unwrap(), [shape[0], 1]);
                let mut charges_arr = rlst_dynamic_array2!($type, [shape[0], 1]);
                let p1 = charges_slice.data().as_ptr();
                let p2 = charges_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, charges_slice.data().len());
                }

                let fmm = SingleNodeBuilder::new()
                    .tree(&sources_arr, &targets_arr, Some(n_crit), sparse)
                    .unwrap()
                    .parameters(
                        &charges_arr,
                        expansion_order,
                        Helmholtz3dKernel::new(wavenumber),
                        kernel_eval_type,
                        FftFieldTranslation::new(),
                    )
                    .unwrap()
                    .build()
                    .unwrap();

                let mut source_key_map = HashMap::new();
                let mut source_keys = Vec::new();
                let mut source_leaves = Vec::new();

                for key in fmm.tree.source_tree.all_keys().unwrap() {
                    source_key_map.insert(
                        key.morton(),
                        MortonKey::<<$type as RlstScalar>::Real>::from(key.clone()),
                    );
                    source_keys.push(key.morton);
                    if fmm.tree.source_tree.leaves_set.contains(&key) {
                        source_leaves.push(key.morton)
                    }
                }

                let mut target_key_map = HashMap::new();
                let mut target_keys = Vec::new();
                let mut target_leaves = Vec::new();
                for key in fmm.tree.target_tree.all_keys().unwrap() {
                    target_key_map.insert(
                        key.morton(),
                        MortonKey::<<$type as RlstScalar>::Real>::from(key.clone()),
                    );
                    target_keys.push(key.morton);
                    if fmm.tree.target_tree.leaves_set.contains(&key) {
                        target_leaves.push(key.morton)
                    }
                }

                Ok(Self {
                    fmm,
                    source_key_map,
                    target_key_map,
                    target_keys,
                    source_keys,
                    target_leaves,
                    source_leaves,
                })
            }
        }
    };
}

macro_rules! helmholtz_blas_constructors {
    ($name: ident, $type: ident) => {
        #[pymethods]
        impl $name {
            /// Constructor
            #[new]
            #[allow(clippy::too_many_arguments)]
            pub fn new<'py>(
                expansion_order: usize,
                sources: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                targets: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                charges: PyReadonlyArrayDyn<'py, $type>,
                n_crit: u64,
                sparse: bool,
                kernel_eval_type: usize,
                wavenumber: <$type as RlstScalar>::Real,
                svd_threshold: <$type as RlstScalar>::Real,
            ) -> PyResult<Self> {
                let kernel_eval_type = if kernel_eval_type == 0 {
                    EvalType::Value
                } else if kernel_eval_type == 1 {
                    EvalType::ValueDeriv
                } else {
                    return Err(PyErr::new::<PyTypeError, _>(
                        "Invalid Kernel Evaluation Mode",
                    ));
                };

                let shape = sources.shape();
                let sources_slice =
                    rlst_array_from_slice2!(sources.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut sources_arr =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [shape[0], shape[1]]);
                let p1 = sources_slice.data().as_ptr();
                let p2 = sources_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, sources.as_slice().unwrap().len());
                }

                let shape = targets.shape();
                let targets_slice =
                    rlst_array_from_slice2!(targets.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut targets_arr =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [shape[0], shape[1]]);
                let p1 = targets_slice.data().as_ptr();
                let p2 = targets_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, targets_slice.data().len());
                }

                let shape = charges.shape();
                let charges_slice =
                    rlst_array_from_slice2!(charges.as_slice().unwrap(), [shape[0], shape[1]]);
                let mut charges_arr = rlst_dynamic_array2!($type, [shape[0], shape[1]]);
                let p1 = charges_slice.data().as_ptr();
                let p2 = charges_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, charges_slice.data().len());
                }

                let fmm = SingleNodeBuilder::new()
                    .tree(&sources_arr, &targets_arr, Some(n_crit), sparse)
                    .unwrap()
                    .parameters(
                        &charges_arr,
                        expansion_order,
                        Helmholtz3dKernel::new(wavenumber),
                        kernel_eval_type,
                        BlasFieldTranslationIa::new(Some(svd_threshold)),
                    )
                    .unwrap()
                    .build()
                    .unwrap();

                let mut source_key_map = HashMap::new();
                let mut source_keys = Vec::new();
                let mut source_leaves = Vec::new();

                for key in fmm.tree.source_tree.all_keys().unwrap() {
                    source_key_map.insert(
                        key.morton(),
                        MortonKey::<<$type as RlstScalar>::Real>::from(key.clone()),
                    );
                    source_keys.push(key.morton);
                    if fmm.tree.source_tree.leaves_set.contains(&key) {
                        source_leaves.push(key.morton)
                    }
                }

                let mut target_key_map = HashMap::new();
                let mut target_keys = Vec::new();
                let mut target_leaves = Vec::new();
                for key in fmm.tree.target_tree.all_keys().unwrap() {
                    target_key_map.insert(
                        key.morton(),
                        MortonKey::<<$type as RlstScalar>::Real>::from(key.clone()),
                    );
                    target_keys.push(key.morton);
                    if fmm.tree.target_tree.leaves_set.contains(&key) {
                        target_leaves.push(key.morton)
                    }
                }

                Ok(Self {
                    fmm,
                    source_key_map,
                    target_key_map,
                    target_keys,
                    source_keys,
                    target_leaves,
                    source_leaves,
                })
            }
        }
    };
}

laplace_fft_constructors!(LaplaceFft64, f64);
laplace_fft_constructors!(LaplaceFft32, f32);
laplace_blas_constructors!(LaplaceBlas64, f64);
laplace_blas_constructors!(LaplaceBlas32, f32);
helmholtz_fft_constructors!(HelmholtzFft64, c64);
helmholtz_fft_constructors!(HelmholtzFft32, c32);
helmholtz_blas_constructors!(HelmholtzBlas64, c64);
helmholtz_blas_constructors!(HelmholtzBlas32, c32);

macro_rules! define_class_methods {
    ($name: ident, $type: ident, $kernel: ident, $field_translation: ident) => {
        /// Python interface
        #[pymethods]
        impl $name {
            fn evaluate(&self) -> PyResult<()> {
                self.fmm.evaluate().unwrap();
                Ok(())
            }

            fn clear(&mut self, charges: PyReadonlyArrayDyn<'_, $type>) -> PyResult<()> {
                let shape = charges.shape();
                let charges_slice =
                    rlst_array_from_slice2!(charges.as_slice().unwrap(), [shape[0], 1]);
                let mut charges_arr = rlst_dynamic_array2!($type, [shape[0], 1]);
                let p1 = charges_slice.data().as_ptr();
                let p2 = charges_arr.data_mut().as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(p1, p2, charges_slice.data().len());
                }
                self.fmm.clear(&charges_arr);
                Ok(())
            }

            fn evaluate_kernel_st<'py>(
                &self,
                py: Python<'py>,
                sources: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                targets: PyReadonlyArrayDyn<'py, <$type as RlstScalar>::Real>,
                charges: PyReadonlyArrayDyn<'py, $type>,
            ) -> PyResult<Bound<'py, PyArray<$type, Dim<[usize; 2]>>>> {
                let shape = sources.shape();
                let sources_slice =
                    rlst_array_from_slice2!(sources.as_slice().unwrap(), [shape[0], shape[1]]);

                let shape = targets.shape();
                let targets_slice =
                    rlst_array_from_slice2!(targets.as_slice().unwrap(), [shape[0], shape[1]]);

                let shape = charges.shape();
                let charges_slice =
                    rlst_array_from_slice2!(charges.as_slice().unwrap(), [shape[0], 1]);

                let shape = targets.shape();
                let ntargets = shape[0];
                let mut result_arr =
                    rlst_dynamic_array2!($type, [shape[0] * self.fmm.kernel_eval_size, 1]);

                self.fmm.kernel.evaluate_st(
                    self.fmm.kernel_eval_type,
                    sources_slice.data(),
                    targets_slice.data(),
                    charges_slice.data(),
                    result_arr.data_mut(),
                );

                let result_arr = result_arr
                    .data()
                    .to_pyarray_bound(py)
                    .reshape_with_order(
                        [ntargets, self.fmm.kernel_eval_size],
                        NPY_ORDER::NPY_FORTRANORDER,
                    )
                    .unwrap();

                Ok(result_arr)
            }

            fn source_key_to_anchor<'py>(
                &self,
                py: Python<'py>,
                key: u64,
            ) -> PyResult<Bound<'py, PyArray<u64, Dim<[usize; 1]>>>> {
                let key = self.source_key_map.get(&key).unwrap();
                let anchor = key.anchor;
                let level = key.level();
                let result = [anchor[0], anchor[1], anchor[2], level].to_pyarray_bound(py);
                Ok(result)
            }

            fn target_key_to_anchor<'py>(
                &self,
                py: Python<'py>,
                key: u64,
            ) -> PyResult<Bound<'py, PyArray<u64, Dim<[usize; 1]>>>> {
                let key = self.target_key_map.get(&key).unwrap();
                let anchor = key.anchor;
                let level = key.level();
                let result = [anchor[0], anchor[1], anchor[2], level].to_pyarray_bound(py);
                Ok(result)
            }

            fn source_coordinates<'py>(
                &self,
                py: Python<'py>,
                leaf: u64,
            ) -> PyResult<Bound<'py, PyArray<<$type as RlstScalar>::Real, Dim<[usize; 2]>>>> {
                let key = self.source_key_map.get(&leaf).unwrap();
                let slice = self.fmm.tree.source_tree.coordinates(&key).unwrap();
                let ncoords = slice.len() / self.fmm.dim();
                let coords_row_major =
                    rlst_array_from_slice2!(slice, [ncoords, self.fmm.dim()], [self.fmm.dim(), 1]);
                let mut coords_col_major =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [ncoords, self.fmm.dim()]);
                coords_col_major.fill_from(coords_row_major.view());
                let coords = coords_col_major
                    .data()
                    .to_pyarray_bound(py)
                    .reshape_with_order([ncoords, self.fmm.dim()], NPY_ORDER::NPY_FORTRANORDER)
                    .unwrap();
                Ok(coords)
            }

            fn target_coordinates<'py>(
                &self,
                py: Python<'py>,
                leaf: u64,
            ) -> PyResult<Bound<'py, PyArray<<$type as RlstScalar>::Real, Dim<[usize; 2]>>>> {
                let key = self.target_key_map.get(&leaf).unwrap();
                let slice = self.fmm.tree.target_tree.coordinates(&key).unwrap();
                let ncoords = slice.len() / self.fmm.dim();
                let coords_row_major =
                    rlst_array_from_slice2!(slice, [ncoords, self.fmm.dim()], [self.fmm.dim(), 1]);
                let mut coords_col_major =
                    rlst_dynamic_array2!(<$type as RlstScalar>::Real, [ncoords, self.fmm.dim()]);
                coords_col_major.fill_from(coords_row_major.view());
                let coords = coords_col_major
                    .data()
                    .to_pyarray_bound(py)
                    .reshape_with_order([ncoords, self.fmm.dim()], NPY_ORDER::NPY_FORTRANORDER)
                    .unwrap();
                Ok(coords)
            }

            fn potentials<'py>(
                &self,
                py: Python<'py>,
                leaf: u64,
            ) -> PyResult<Vec<Bound<'py, PyArray<$type, Dim<[usize; 2]>>>>> {
                let key = self.target_key_map.get(&leaf).unwrap();
                let potentials = self.fmm.potential(&key).unwrap();
                let n_matvec = potentials.len();
                let mut result = Vec::new();

                for i in 0..n_matvec {
                    let n_potentials = potentials[i].len() / self.fmm.kernel_eval_size;
                    let potentials_i = potentials[i].to_pyarray_bound(py);

                    let potentials_i = potentials_i
                        .reshape_with_order(
                            [n_potentials, self.fmm.kernel_eval_size],
                            NPY_ORDER::NPY_FORTRANORDER,
                        )
                        .unwrap();
                    result.push(potentials_i)
                }
                Ok(result)
            }

            #[getter]
            fn source_tree_depth(&self) -> PyResult<u64> {
                Ok(self.fmm.tree.source_tree.depth())
            }

            #[getter]
            fn target_tree_depth(&self) -> PyResult<u64> {
                Ok(self.fmm.tree.source_tree.depth())
            }

            #[getter]
            fn source_keys<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray<u64, Dim<[usize; 1]>>>> {
                let array = self.source_keys.as_slice().to_pyarray_bound(py);
                Ok(array)
            }

            #[getter]
            fn target_keys<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray<u64, Dim<[usize; 1]>>>> {
                let array = self.target_keys.as_slice().to_pyarray_bound(py);
                Ok(array)
            }

            #[getter]
            fn source_leaves<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray<u64, Dim<[usize; 1]>>>> {
                let array = self.source_leaves.as_slice().to_pyarray_bound(py);
                Ok(array)
            }

            #[getter]
            fn target_leaves<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray<u64, Dim<[usize; 1]>>>> {
                let array = self.target_leaves.as_slice().to_pyarray_bound(py);
                Ok(array)
            }

            #[getter]
            fn source_global_indices<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray<usize, Dim<[usize; 1]>>>> {
                let array = self
                    .fmm
                    .tree
                    .source_tree
                    .global_indices
                    .as_slice()
                    .to_pyarray_bound(py);
                Ok(array)
            }

            #[getter]
            fn target_global_indices<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray<usize, Dim<[usize; 1]>>>> {
                let array = self
                    .fmm
                    .tree
                    .target_tree
                    .global_indices
                    .as_slice()
                    .to_pyarray_bound(py);
                Ok(array)
            }
        }
    };
}

define_class_methods!(LaplaceFft32, f32, Laplace3dKernel, FftFieldTranslation);
define_class_methods!(LaplaceFft64, f64, Laplace3dKernel, FftFieldTranslation);
define_class_methods!(
    LaplaceBlas32,
    f32,
    Laplace3dKernel,
    BlasFieldTranslationSaRcmp
);
define_class_methods!(
    LaplaceBlas64,
    f64,
    Laplace3dKernel,
    BlasFieldTranslationSaRcmp
);

define_class_methods!(HelmholtzFft32, c32, Helmholtz3dKernel, FftFieldTranslation);
define_class_methods!(HelmholtzFft64, c64, Helmholtz3dKernel, FftFieldTranslation);
define_class_methods!(
    HelmholtzBlas32,
    c32,
    Helmholtz3dKernel,
    BlasFieldTranslationIa
);
define_class_methods!(
    HelmholtzBlas64,
    c64,
    Helmholtz3dKernel,
    BlasFieldTranslationIa
);

/// Python bindings to KiFMM-RS
#[pymodule]
pub fn kifmm_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LaplaceFft32>()?;
    m.add_class::<LaplaceFft64>()?;
    m.add_class::<LaplaceBlas32>()?;
    m.add_class::<LaplaceBlas64>()?;
    m.add_class::<HelmholtzFft32>()?;
    m.add_class::<HelmholtzFft64>()?;
    m.add_class::<HelmholtzBlas32>()?;
    m.add_class::<HelmholtzBlas64>()?;
    Ok(())
}