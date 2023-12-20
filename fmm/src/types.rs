use std::collections::HashMap;

use bempp_traits::{field::FieldTranslationData, fmm::Fmm, kernel::Kernel, tree::Tree};
use cauchy::Scalar;
use num::{Complex, Float};
use rlst::dense::traits::*;
use rlst::dense::{base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix};

/// Type alias for charge data
pub type Charge<T> = T;

/// Type alias for global index for identifying charge data with a point
pub type GlobalIdx = usize;

/// Type alias for mapping charge data to global indices.
pub type ChargeDict<T> = HashMap<GlobalIdx, Charge<T>>;

/// Type alias for multipole/local expansion containers.
pub type Expansions<T> = Matrix<T, BaseMatrix<T, VectorContainer<T>, Dynamic>, Dynamic>;

/// Type alias for potential containers.
pub type Potentials<T> = Matrix<T, BaseMatrix<T, VectorContainer<T>, Dynamic>, Dynamic>;

/// Type alias for approximation of FMM operator matrices.
pub type C2EType<T> = Matrix<T, BaseMatrix<T, VectorContainer<T>, Dynamic>, Dynamic>;

pub struct FmmDataLinear<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
{
    /// The associated FMM object, which implements an FMM interface
    pub fmm: T,

    /// The multipole expansion data at each box.
    pub multipoles: Vec<U>,

    /// Multipole expansions at leaf level
    pub leaf_multipoles: Vec<SendPtrMut<U>>,

    /// Multipole expansions at each level
    pub level_multipoles: Vec<Vec<SendPtrMut<U>>>,

    /// The local expansion at each box
    pub locals: Vec<U>,

    /// Local expansions at the leaf level
    pub leaf_locals: Vec<SendPtrMut<U>>,

    /// The local expansion data at each level.
    pub level_locals: Vec<Vec<SendPtrMut<U>>>,

    /// The evaluated potentials at each leaf box.
    pub potentials: Vec<U>,

    /// The evaluated potentials at each leaf box.
    pub potentials_send_pointers: Vec<SendPtrMut<U>>,

    /// All upward surfaces
    pub upward_surfaces: Vec<U>,

    /// All downward surfaces
    pub downward_surfaces: Vec<U>,

    /// Leaf upward surfaces
    pub leaf_upward_surfaces: Vec<U>,

    /// Leaf downward surfaces
    pub leaf_downward_surfaces: Vec<U>,

    /// The charge data at each leaf box.
    pub charges: Vec<U>,

    /// Index pointer between leaf keys and charges
    pub charge_index_pointer: Vec<(usize, usize)>,

    /// Scales of each leaf operator
    pub scales: Vec<U>,

    /// Global indices of each charge
    pub global_indices: Vec<usize>,
}

/// Type to store data associated with the kernel independent (KiFMM) in.
pub struct KiFmmLinear<T, U, V, W>
where
    T: Tree,
    U: Kernel<T = W>,
    V: FieldTranslationData<U>,
    W: Scalar + Float + Default,
{
    /// The expansion order
    pub order: usize,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_1: C2EType<W>,
    pub uc2e_inv_2: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_1: C2EType<W>,
    pub dc2e_inv_2: C2EType<W>,

    /// The ratio of the inner check surface diamater in comparison to the surface discretising a box.
    pub alpha_inner: W,

    /// The ratio of the outer check surface diamater in comparison to the surface discretising a box.
    pub alpha_outer: W,

    /// The multipole to multipole operator matrices, each index is associated with a child box (in sequential Morton order),
    pub m2m: C2EType<W>,

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub l2l: Vec<C2EType<W>>,

    /// The tree (single or multi node) associated with this FMM
    pub tree: T,

    /// The kernel associated with this FMM.
    pub kernel: U,

    /// The M2L operator matrices, as well as metadata associated with this FMM.
    pub m2l: V,
}

/// A threadsafe mutable raw pointer
#[derive(Clone, Debug, Copy)]
pub struct SendPtrMut<T> {
    pub raw: *mut T,
}

unsafe impl<T> Sync for SendPtrMut<T> {}
unsafe impl<T> Send for SendPtrMut<Complex<T>> {}

impl<T> Default for SendPtrMut<T> {
    fn default() -> Self {
        SendPtrMut {
            raw: std::ptr::null_mut(),
        }
    }
}

/// A threadsafe raw pointer
#[derive(Clone, Debug, Copy)]
pub struct SendPtr<T> {
    pub raw: *const T,
}

unsafe impl<T> Sync for SendPtr<T> {}

impl<T> Default for SendPtr<T> {
    fn default() -> Self {
        SendPtr {
            raw: std::ptr::null(),
        }
    }
}