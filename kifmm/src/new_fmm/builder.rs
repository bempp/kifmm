//! Builder objects to construct FMMs
use crate::new_fmm::helpers::{
    coordinate_index_pointer, homogenous_kernel_scale, leaf_expansion_pointers, leaf_scales,
    leaf_surfaces, level_expansion_pointers, level_index_pointer, map_charges, ncoeffs_kifmm,
    potential_pointers,
};
use crate::{
    new_fmm::types::{
        Charges, Coordinates, FmmEvalType, KiFmm, SingleNodeBuilder, SingleNodeFmmTree,
    },
    Epsilon,
};

use crate::traits::field::ConfigureSourceToTargetData;
use crate::traits::tree::FmmTreeNode;
use crate::traits::{field::SourceToTargetData as SourceToTargetDataTrait, tree::Tree};
use crate::tree::{
    constants::{ALPHA_INNER, ALPHA_OUTER},
    types::{Domain, MortonKey, SingleNodeTree},
};
use green_kernels::{traits::Kernel as KernelTrait, types::EvalType};
use rlst::{
    empty_array, rlst_dynamic_array2, Array, BaseArray, MatrixSvd, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar, Shape, VectorContainer,
};

use super::{constants::DEFAULT_NCRIT, pinv::pinv};

impl<Scalar, Kernel, SourceToTargetData> SingleNodeBuilder<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Epsilon,
    <Scalar as RlstScalar>::Real: Default + Epsilon,
    Kernel: KernelTrait<T = Scalar> + Clone + Default,
    SourceToTargetData: ConfigureSourceToTargetData<Scalar = Scalar, Kernel = Kernel, Domain = Domain<Scalar::Real>>
        + Default,
    Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>: MatrixSvd<Item = Scalar>,
{
    /// Initialise an empty kernel independent FMM builder
    pub fn new() -> Self {
        Self {
            tree: None,
            kernel: None,
            charges: None,
            source_to_target: None,
            domain: None,
            expansion_order: None,
            ncoeffs: None,
            kernel_eval_type: None,
            fmm_eval_type: None,
        }
    }

    /// Associate FMM builder with an FMM Tree
    ///
    /// # Arguments
    /// * `sources` - Source coordinates, data expected in column major order such that the shape is [n_coords, dim]
    /// * `target` - Target coordinates,  data expected in column major order such that the shape is [n_coords, dim]
    /// * `n_crit` - Maximum number of particles per leaf box, if none specified a default of 150 is used.
    /// * `sparse` - Optionally drop empty leaf boxes for performance.`
    pub fn tree(
        mut self,
        sources: &Coordinates<Scalar::Real>,
        targets: &Coordinates<Scalar::Real>,
        n_crit: Option<u64>,
        sparse: bool,
    ) -> Result<Self, std::io::Error> {
        let [nsources, dims] = sources.shape();
        let [ntargets, dimt] = targets.shape();

        if dims < 3 || dimt < 3 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Only 3D FMM supported",
            ))
        } else if nsources == 0 || ntargets == 0 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Must have a positive number of source or target particles",
            ))
        } else {
            // Source and target trees calcualted over the same domain
            let source_domain = Domain::from_local_points(sources.data());
            let target_domain = Domain::from_local_points(targets.data());

            // Calculate union of domains for source and target points, needed to define operators
            let domain = source_domain.union(&target_domain);
            self.domain = Some(domain);

            // If not specified estimate from point data estimate critical value
            let n_crit = n_crit.unwrap_or(DEFAULT_NCRIT);
            let [nsources, _dim] = sources.shape();
            let [ntargets, _dim] = targets.shape();

            // Estimate depth based on a uniform distribution
            let source_depth =
                SingleNodeTree::<Scalar::Real>::minimum_depth(nsources as u64, n_crit);
            let target_depth =
                SingleNodeTree::<Scalar::Real>::minimum_depth(ntargets as u64, n_crit);
            let depth = source_depth.max(target_depth); // refine source and target trees to same depth

            let source_tree = SingleNodeTree::new(sources.data(), depth, sparse, self.domain)?;
            let target_tree = SingleNodeTree::new(targets.data(), depth, sparse, self.domain)?;

            let fmm_tree = SingleNodeFmmTree {
                source_tree,
                target_tree,
                domain,
            };

            self.tree = Some(fmm_tree);
            Ok(self)
        }
    }

    /// For an FMM builder with an associated FMM tree, specify simulation specific parameters
    ///
    /// # Arguments
    /// * `charges` - 2D RLST array, of dimensions `[ncharges, nvecs]` where each of `nvecs` is associated with `ncharges`
    /// * `expansion_order` - The expansion order of the FMM
    /// * `kernel` - The kernel associated with this FMM
    /// * `eval_type` - Either `ValueDeriv` - to evaluate potentials and gradients, or `Value` to evaluate potentials alone
    pub fn parameters(
        mut self,
        charges: &Charges<Scalar>,
        expansion_order: usize,
        kernel: Kernel,
        eval_type: EvalType,
        mut source_to_target: SourceToTargetData,
    ) -> Result<Self, std::io::Error> {
        if self.tree.is_none() {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Must build tree before specifying FMM parameters",
            ))
        } else {
            // Set FMM parameters
            let global_indices = self
                .tree
                .as_ref()
                .unwrap()
                .source_tree
                .all_global_indices()
                .unwrap();

            let [_ncharges, nmatvecs] = charges.shape();

            self.charges = Some(map_charges(global_indices, charges));

            if nmatvecs > 1 {
                self.fmm_eval_type = Some(FmmEvalType::Matrix(nmatvecs))
            } else {
                self.fmm_eval_type = Some(FmmEvalType::Vector)
            }
            self.expansion_order = Some(expansion_order);
            self.ncoeffs = Some(ncoeffs_kifmm(expansion_order));

            self.kernel = Some(kernel);
            self.kernel_eval_type = Some(eval_type);

            // Calculate source to target translation metadata
            // Set the expansion order
            source_to_target.expansion_order(self.expansion_order.unwrap());

            // Set the associated kernel
            let kernel = self.kernel.as_ref().unwrap().clone();
            source_to_target.kernel(kernel);

            // Compute the field translation operators
            source_to_target.operator_data(self.expansion_order.unwrap(), self.domain.unwrap());

            self.source_to_target = Some(source_to_target);

            Ok(self)
        }
    }

    /// Finalize and build the single node FMM
    pub fn build(self) -> Result<KiFmm<Scalar, Kernel, SourceToTargetData>, std::io::Error> {
        if self.tree.is_none() {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Must create a tree, and FMM metadata before building",
            ))
        } else {
            // Configure with tree, expansion parameters and source to target field translation operators
            let kernel = self.kernel.unwrap();
            let dim = kernel.space_dimension();

            let mut result = KiFmm {
                tree: self.tree.unwrap(),
                expansion_order: self.expansion_order.unwrap(),
                ncoeffs: self.ncoeffs.unwrap(),
                source_to_target: self.source_to_target.unwrap(),
                fmm_eval_type: self.fmm_eval_type.unwrap(),
                kernel_eval_type: self.kernel_eval_type.unwrap(),
                kernel,
                dim,
                ..Default::default()
            };

            result.set_source_and_target_operator_data();

            result.set_metadata(self.kernel_eval_type.unwrap(), &self.charges.unwrap());

            Ok(result)
        }
    }
}

impl<Scalar, Kernel, SourceToTargetData> KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Epsilon + Default,
    Kernel: KernelTrait<T = Scalar>,
    SourceToTargetData: SourceToTargetDataTrait,
    <Scalar as RlstScalar>::Real: Default,
    Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>: MatrixSvd<Item = Scalar>,
{
    fn set_source_and_target_operator_data(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain;

        // Compute required surfaces
        let upward_equivalent_surface =
            root.surface_grid(self.expansion_order, &domain, alpha_inner);
        let upward_check_surface = root.surface_grid(self.expansion_order, &domain, alpha_outer);
        let downward_equivalent_surface =
            root.surface_grid(self.expansion_order, &domain, alpha_outer);
        let downward_check_surface = root.surface_grid(self.expansion_order, &domain, alpha_inner);

        let nequiv_surface = upward_equivalent_surface.len() / self.dim;
        let ncheck_surface = upward_check_surface.len() / self.dim;

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using SVD
        let mut uc2e_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &upward_equivalent_surface[..],
            &upward_check_surface[..],
            uc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut uc2e = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
        uc2e.fill_from(uc2e_t.transpose());

        let mut dc2e_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &downward_equivalent_surface[..],
            &downward_check_surface[..],
            dc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut dc2e = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
        dc2e.fill_from(dc2e_t.transpose());

        let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = Scalar::from_real(s[i]);
        }

        let uc2e_inv_1 = empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let uc2e_inv_2 = ut;

        let (s, ut, v) = pinv::<Scalar>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = Scalar::from_real(s[i]);
        }

        let dc2e_inv_1 = empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let dc2e_inv_2 = ut;

        // Calculate M2M and L2L operator matrices
        let children = root.children();
        let mut m2m = rlst_dynamic_array2!(Scalar, [nequiv_surface, 8 * nequiv_surface]);
        let mut m2m_vec = Vec::new();
        let mut l2l = Vec::new();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.surface_grid(self.expansion_order, &domain, alpha_inner);
            let child_downward_check_surface =
                child.surface_grid(self.expansion_order, &domain, alpha_inner);

            let mut pc2ce_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);

            self.kernel.assemble_st(
                EvalType::Value,
                &child_upward_equivalent_surface,
                &upward_check_surface,
                pc2ce_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut pc2ce = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
            pc2ce.fill_from(pc2ce_t.transpose());

            let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                uc2e_inv_1.view(),
                empty_array::<Scalar, 2>().simple_mult_into_resize(uc2e_inv_2.view(), pc2ce.view()),
            );
            let l = i * nequiv_surface * nequiv_surface;
            let r = l + nequiv_surface * nequiv_surface;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());
            m2m_vec.push(tmp);

            let mut cc2pe_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);

            self.kernel.assemble_st(
                EvalType::Value,
                &downward_equivalent_surface,
                &child_downward_check_surface,
                cc2pe_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut cc2pe = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
            cc2pe.fill_from(cc2pe_t.transpose());
            let mut tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                dc2e_inv_1.view(),
                empty_array::<Scalar, 2>().simple_mult_into_resize(dc2e_inv_2.view(), cc2pe.view()),
            );
            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= homogenous_kernel_scale(child.level()));

            l2l.push(tmp);
        }

        self.source = m2m;
        self.source_vec = m2m_vec;
        self.target_vec = l2l;
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }

    fn set_metadata(&mut self, eval_type: EvalType, charges: &Charges<Scalar>) {
        let alpha_outer = Scalar::real(ALPHA_OUTER);

        // Check if computing potentials, or potentials and derivatives
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => self.dim + 1,
        };

        // Check if we are computing matvec or matmul
        let [_ncharges, nmatvecs] = charges.shape();
        let ntarget_points = self.tree.target_tree.all_coordinates().unwrap().len() / self.dim;
        let nsource_keys = self.tree.source_tree.n_keys_tot().unwrap();
        let ntarget_keys = self.tree.target_tree.n_keys_tot().unwrap();
        let ntarget_leaves = self.tree.target_tree.n_leaves().unwrap();
        let nsource_leaves = self.tree.source_tree.n_leaves().unwrap();

        // Buffers to store all multipole and local data
        let multipoles = vec![Scalar::default(); self.ncoeffs * nsource_keys * nmatvecs];
        let locals = vec![Scalar::default(); self.ncoeffs * ntarget_keys * nmatvecs];

        // Index pointers of multipole and local data, indexed by level
        let level_index_pointer_multipoles = level_index_pointer(&self.tree.source_tree);
        let level_index_pointer_locals = level_index_pointer(&self.tree.target_tree);

        // Buffer to store evaluated potentials and/or gradients at target points
        let potentials = vec![Scalar::default(); ntarget_points * eval_size * nmatvecs];

        // Kernel scale at each target and source leaf
        let source_leaf_scales = leaf_scales(&self.tree.source_tree, self.ncoeffs);

        // Pre compute check surfaces
        let leaf_upward_surfaces_sources = leaf_surfaces(
            &self.tree.source_tree,
            self.ncoeffs,
            alpha_outer,
            self.expansion_order,
        );
        let leaf_upward_surfaces_targets = leaf_surfaces(
            &self.tree.target_tree,
            self.ncoeffs,
            alpha_outer,
            self.expansion_order,
        );

        // Mutable pointers to multipole and local data, indexed by level
        let level_multipoles =
            level_expansion_pointers(&self.tree.source_tree, self.ncoeffs, nmatvecs, &multipoles);

        let level_locals =
            level_expansion_pointers(&self.tree.source_tree, self.ncoeffs, nmatvecs, &locals);

        // Mutable pointers to multipole and local data only at leaf level
        let leaf_multipoles = leaf_expansion_pointers(
            &self.tree.source_tree,
            self.ncoeffs,
            nmatvecs,
            nsource_leaves,
            &multipoles,
        );

        let leaf_locals = leaf_expansion_pointers(
            &self.tree.target_tree,
            self.ncoeffs,
            nmatvecs,
            ntarget_leaves,
            &locals,
        );

        // Mutable pointers to potential data at each target leaf
        let potentials_send_pointers = potential_pointers(
            &self.tree.target_tree,
            nmatvecs,
            ntarget_leaves,
            ntarget_points,
            eval_size,
            &potentials,
        );

        // Index pointer of charge data at each target leaf
        let charge_index_pointer_targets = coordinate_index_pointer(&self.tree.target_tree);
        let charge_index_pointer_sources = coordinate_index_pointer(&self.tree.source_tree);

        // Set data
        self.multipoles = multipoles;
        self.locals = locals;
        self.leaf_multipoles = leaf_multipoles;
        self.level_multipoles = level_multipoles;
        self.leaf_locals = leaf_locals;
        self.level_locals = level_locals;
        self.level_index_pointer_locals = level_index_pointer_locals;
        self.level_index_pointer_multipoles = level_index_pointer_multipoles;
        self.potentials = potentials;
        self.potentials_send_pointers = potentials_send_pointers;
        self.leaf_upward_surfaces_sources = leaf_upward_surfaces_sources;
        self.leaf_upward_surfaces_targets = leaf_upward_surfaces_targets;
        self.charges = charges.data().to_vec();
        self.charge_index_pointer_targets = charge_index_pointer_targets;
        self.charge_index_pointer_sources = charge_index_pointer_sources;
        self.leaf_scales_sources = source_leaf_scales;
        self.kernel_eval_size = eval_size;
    }
}
