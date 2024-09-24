use mpi::{
    topology::{Communicator, SimpleCommunicator},
    traits::Equivalence,
};
use num::Float;
use rlst::{rlst_dynamic_array2, MatrixSvd, RlstScalar};

use crate::{
    fmm::{
        helpers::single_node::ncoeffs_kifmm,
        types::{
            FmmEvalType, Isa, KiFmmMulti, Layout, MultiNodeBuilder, NeighbourhoodCommunicator,
            Query,
        },
        KiFmm,
    },
    traits::{
        field::{
            SourceAndTargetTranslationMetadata, SourceToTargetData as SourceToTargetDataTrait,
            SourceToTargetTranslationMetadata,
        },
        fmm::{FmmMetadata, FmmMetadataAccess, HomogenousKernel},
        general::single_node::Epsilon,
    },
    tree::{
        types::{Domain, SortKind},
        MultiNodeTree, SingleNodeTree,
    },
    MultiNodeFmmTree,
};
use green_kernels::{traits::Kernel as KernelTrait, types::GreenKernelEvalType};

impl<Scalar, Kernel, SourceToTargetData> MultiNodeBuilder<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Epsilon + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Clone + Default,
    SourceToTargetData: SourceToTargetDataTrait + Default,
    KiFmmMulti<Scalar, Kernel, SourceToTargetData>: SourceToTargetTranslationMetadata
        + SourceAndTargetTranslationMetadata
        + FmmMetadata<Scalar = Scalar>
        + FmmMetadataAccess,
    KiFmm<Scalar, Kernel, SourceToTargetData>: SourceToTargetTranslationMetadata
        + SourceAndTargetTranslationMetadata
        + FmmMetadata<Scalar = Scalar>
        + FmmMetadataAccess,
{
    /// Init
    pub fn new() -> Self {
        Self {
            communicator: None,
            isa: None,
            tree: None,
            kernel: None,
            charges: None,
            source_to_target: None,
            domain: None,
            equivalent_surface_order: None,
            check_surface_order: None,
            n_coeffs_equivalent_surface: None,
            n_coeffs_check_surface: None,
            kernel_eval_type: None,
            fmm_eval_type: None,
        }
    }

    /// Tree
    pub fn tree(
        mut self,
        comm: &SimpleCommunicator,
        sources: &[Scalar::Real],
        targets: &[Scalar::Real],
        local_depth: u64,
        global_depth: u64,
        prune_empty: bool,
        sort_kind: SortKind,
    ) -> Result<Self, std::io::Error> {
        let dim = 3;
        let n_sources = sources.len() / dim;
        let n_targets = targets.len() / dim;

        let dims = sources.len() % dim;
        let dimt = targets.len() % dim;

        if dims != 0 || dimt != 0 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Only 3D FMM supported",
            ))
        } else if n_sources == 0 || n_targets == 0 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Must have a positive number of source or target particles",
            ))
        } else {
            // Source and target trees calcualted over the same domain
            let source_domain = Domain::from_global_points(sources, comm);
            let target_domain = Domain::from_global_points(targets, comm);

            // Calculate union of domains for source and target points, needed to define operators
            let domain = source_domain.union(&target_domain);

            let source_tree = MultiNodeTree::new(
                comm,
                sources,
                local_depth,
                global_depth,
                Some(domain),
                sort_kind.clone(),
                prune_empty,
            )?;

            let target_tree = MultiNodeTree::new(
                comm,
                targets,
                local_depth,
                global_depth,
                Some(domain),
                sort_kind.clone(),
                prune_empty,
            )?;

            // Create an FMM tree, and set its layout of source boxes
            let mut fmm_tree = MultiNodeFmmTree {
                source_tree,
                target_tree,
                domain,
                source_layout: Layout::default(),
                v_list_query: Query::default(),
                u_list_query: Query::default(),
            };

            // Global communication to set the source layout required
            fmm_tree.set_source_layout();

            // Set requires queries at this point for U and V list data, can be intensive for deep trees
            // as manually constructs interaction lists
            fmm_tree.set_queries(true);
            fmm_tree.set_queries(false);

            self.communicator = Some(comm.duplicate());
            self.tree = Some(fmm_tree);
            Ok(self)
        }
    }

    /// Parameters
    pub fn parameters(
        mut self,
        expansion_order: usize,
        kernel: Kernel,
        source_to_target: SourceToTargetData,
    ) -> Result<Self, std::io::Error> {
        if self.tree.is_none() {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Must build tree before specifying FMM parameters",
            ))
        } else {
            // TODO: Mapping of global indices needs to happen here eventually.
            self.n_coeffs_equivalent_surface = Some(ncoeffs_kifmm(expansion_order));
            self.n_coeffs_check_surface = Some(ncoeffs_kifmm(expansion_order));
            self.kernel = Some(kernel);
            self.fmm_eval_type = Some(FmmEvalType::Vector);
            self.kernel_eval_type = Some(GreenKernelEvalType::Value);
            self.isa = Some(Isa::new());
            self.source_to_target = Some(source_to_target);
            self.equivalent_surface_order = Some(expansion_order);
            self.check_surface_order = Some(expansion_order);
            Ok(self)
        }
    }

    /// Initialise
    pub fn build(self) -> Result<KiFmmMulti<Scalar, Kernel, SourceToTargetData>, std::io::Error> {
        if self.tree.is_none() {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Must create a tree, and FMM metadata before building",
            ))
        } else {
            let kernel = self.kernel.unwrap();
            let communicator = self.communicator.unwrap();
            let neighbourhood_communicator_v = NeighbourhoodCommunicator::from_comm(&communicator);
            let neighbourhood_communicator_u = NeighbourhoodCommunicator::from_comm(&communicator);
            let rank = communicator.rank();
            let source_to_target = self.source_to_target.unwrap();
            let fmm_eval_type = self.fmm_eval_type.unwrap();
            let kernel_eval_type = self.kernel_eval_type.unwrap();
            let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface.unwrap();
            let n_coeffs_check_surface = self.n_coeffs_check_surface.unwrap();
            let equivalent_surface_order = self.equivalent_surface_order.unwrap();
            let check_surface_order = self.check_surface_order.unwrap();

            let tmp_arr = rlst_dynamic_array2!(Scalar, [1, 1]);
            let global_fmm: KiFmm<Scalar, Kernel, SourceToTargetData> = KiFmm {
                isa: self.isa.unwrap(),
                equivalent_surface_order: vec![equivalent_surface_order],
                check_surface_order: vec![check_surface_order],
                variable_expansion_order: false,
                n_coeffs_equivalent_surface: vec![n_coeffs_check_surface],
                n_coeffs_check_surface: vec![n_coeffs_equivalent_surface],
                fmm_eval_type: fmm_eval_type.clone(),
                kernel_eval_type: kernel_eval_type.clone(),
                kernel: kernel.clone(),
                dim: 3,
                ..Default::default()
            };

            let mut result = KiFmmMulti {
                dim: 3,
                times: Vec::default(),
                isa: self.isa.unwrap(),
                communicator,
                neighbourhood_communicator_v,
                neighbourhood_communicator_u,
                rank,
                kernel,
                tree: self.tree.unwrap(),
                equivalent_surface_order: equivalent_surface_order,
                check_surface_order: check_surface_order,
                n_coeffs_equivalent_surface: n_coeffs_equivalent_surface,
                n_coeffs_check_surface: n_coeffs_check_surface,
                source_to_target: source_to_target,
                fmm_eval_type: fmm_eval_type,
                kernel_eval_type: kernel_eval_type,
                global_fmm,
                kernel_eval_size: 1,
                source: tmp_arr,
                charges: Vec::default(),
                ghost_charges: Vec::default(),
                charge_index_pointer_sources: Vec::default(),
                charge_index_pointer_targets: Vec::default(),
                charge_index_pointer_ghost_sources: Vec::default(),
                leaf_upward_check_surfaces_sources: Vec::default(),
                leaf_downward_equivalent_surfaces_targets: Vec::default(),
                leaf_upward_equivalent_surfaces_sources: Vec::default(),
                leaf_scales_sources: Vec::default(),
                uc2e_inv_1: Vec::default(),
                uc2e_inv_2: Vec::default(),
                dc2e_inv_1: Vec::default(),
                dc2e_inv_2: Vec::default(),
                source_vec: Vec::default(),
                target_vec: Vec::default(),
                multipoles: Vec::default(),
                ghost_multipoles: Vec::default(),
                locals: Vec::default(),
                potentials: Vec::default(),
                leaf_multipoles: Vec::default(),
                level_multipoles: Vec::default(),
                ghost_level_multipoles: Vec::default(),
                leaf_locals: Vec::default(),
                level_locals: Vec::default(),
                level_index_pointer_locals: Vec::default(),
                level_index_pointer_multipoles: Vec::default(),
                potentials_send_pointers: Vec::default(),
                ghost_tree_u: SingleNodeTree::default(),
                ghost_tree_v: SingleNodeTree::default(),
                local_roots: Vec::default(),
                local_roots_counts: Vec::default(),
                local_roots_displacements: Vec::default(),
                local_roots_ranks: Vec::default(),
            };

            result.source();
            result.target();
            result.source_to_target();

            // pass dummy charges for now.
            result.metadata(self.kernel_eval_type.unwrap(), &[Scalar::zero(); 1]); // Everything required for the local upward passes
                                                                                   // result.displacements();

            Ok(result)
        }
    }
}
