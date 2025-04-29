use itertools::Itertools;
use mpi::{
    topology::{Communicator, SimpleCommunicator},
    traits::Equivalence,
};
use num::Float;
use rlst::{rlst_dynamic_array2, MatrixSvd, RlstScalar};
use std::collections::HashMap;

use crate::{
    fmm::{
        helpers::single_node::{level_index_pointer_single_node, ncoeffs_kifmm, optionally_time},
        types::{
            FmmEvalType, Isa, KiFmmMulti, Layout, MultiNodeBuilder, NeighbourhoodCommunicator,
            Query,
        },
    },
    traits::{
        field::{
            FieldTranslation as FieldTranslationTrait, SourceToTargetTranslationMetadata,
            SourceTranslationMetadata, TargetTranslationMetadata,
        },
        fmm::{HomogenousKernel, Metadata, MetadataAccess},
        general::{multi_node::GlobalFmmMetadata, single_node::Epsilon},
        types::{CommunicationTime, CommunicationType, MetadataTime, MetadataType},
    },
    tree::{
        types::{Domain, SortKind},
        MultiNodeTree,
    },
    KiFmm, MultiNodeFmmTree, SingleNodeFmmTree,
};

use green_kernels::{traits::Kernel as KernelTrait, types::GreenKernelEvalType};

impl<Scalar, Kernel, FieldTranslation> MultiNodeBuilder<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Epsilon + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Clone + Default,
    FieldTranslation: FieldTranslationTrait + Default + Clone,
    KiFmmMulti<Scalar, Kernel, FieldTranslation>: SourceToTargetTranslationMetadata
        + SourceTranslationMetadata
        + TargetTranslationMetadata
        + Metadata<Scalar = Scalar>
        + MetadataAccess,
    KiFmm<Scalar, Kernel, FieldTranslation>: SourceToTargetTranslationMetadata
        + SourceTranslationMetadata
        + TargetTranslationMetadata
        + Metadata<Scalar = Scalar>
        + MetadataAccess
        + GlobalFmmMetadata<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>,
{
    /// Init
    pub fn new(timed: bool) -> Self {
        Self {
            timed: Some(timed),
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
            communication_times: None,
            variable_expansion_order: None,
        }
    }

    /// Tree
    #[allow(clippy::too_many_arguments)]
    pub fn tree(
        mut self,
        global_communicator: &SimpleCommunicator,
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

        let timed = self.timed.unwrap();

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
            let mut communication_times = Vec::new();

            // Source and target trees calculated over the same domain
            let (source_domain, d) = optionally_time(timed, || {
                Domain::from_global_points(sources, global_communicator)
            });

            if let Some(d) = d {
                communication_times.push(CommunicationTime::from_duration(
                    CommunicationType::SourceDomain,
                    d,
                ))
            }

            let (target_domain, d) = optionally_time(timed, || {
                Domain::from_global_points(targets, global_communicator)
            });

            if let Some(d) = d {
                communication_times.push(CommunicationTime::from_duration(
                    CommunicationType::TargetDomain,
                    d,
                ))
            }

            // Calculate union of domains for source and target points, needed to define operators
            let domain = source_domain.union(&target_domain);

            let (source_tree, d) = optionally_time(timed, || {
                MultiNodeTree::new(
                    global_communicator,
                    sources,
                    local_depth,
                    global_depth,
                    Some(domain),
                    sort_kind.clone(),
                    prune_empty,
                )
            });

            let source_tree = source_tree?;

            if let Some(d) = d {
                communication_times.push(CommunicationTime::from_duration(
                    CommunicationType::SourceTree,
                    d,
                ))
            }

            let (target_tree, d) = optionally_time(timed, || {
                MultiNodeTree::new(
                    global_communicator,
                    targets,
                    local_depth,
                    global_depth,
                    Some(domain),
                    sort_kind.clone(),
                    prune_empty,
                )
            });

            let target_tree = target_tree?;

            if let Some(d) = d {
                communication_times.push(CommunicationTime::from_duration(
                    CommunicationType::TargetTree,
                    d,
                ))
            }

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
            let (_, duration) = optionally_time(timed, || fmm_tree.set_source_layout());

            if let Some(d) = duration {
                communication_times.push(CommunicationTime::from_duration(
                    CommunicationType::Layout,
                    d,
                ))
            }

            // Set requires queries at this point for U and V list data, can be intensive for deep trees
            // as manually constructs interaction lists
            fmm_tree.set_queries(true);
            fmm_tree.set_queries(false);

            self.communicator = Some(global_communicator.duplicate());
            self.tree = Some(fmm_tree);
            self.communication_times = Some(communication_times);
            Ok(self)
        }
    }

    /// Parameters
    pub fn parameters(
        mut self,
        charges: &[Scalar],
        expansion_order: &[usize],
        kernel: Kernel,
        eval_type: GreenKernelEvalType,
        source_to_target: FieldTranslation,
    ) -> Result<Self, std::io::Error> {
        if self.tree.is_none() {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Must build tree before specifying FMM parameters",
            ))
        } else {
            let total_depth = self.tree.as_ref().unwrap().source_tree.total_depth;

            let equivalent_surface_order;

            if expansion_order.len() > 1 {
                self.variable_expansion_order = Some(true);

                if expansion_order.len() != (total_depth as usize) + 1 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Number of expansion orders must either be 1, or match the depth of the tree",
                    ));
                }

                equivalent_surface_order = expansion_order.to_vec();
            } else {
                self.variable_expansion_order = Some(false);
                equivalent_surface_order = vec![expansion_order[0]; (total_depth + 1) as usize];
            }

            let check_surface_order = if source_to_target.overdetermined() {
                equivalent_surface_order
                    .iter()
                    .map(|&e| e + source_to_target.surface_diff())
                    .collect_vec()
            } else {
                equivalent_surface_order.to_vec()
            };

            self.n_coeffs_equivalent_surface = Some(
                equivalent_surface_order
                    .iter()
                    .map(|&e| ncoeffs_kifmm(e))
                    .collect_vec(),
            );

            self.n_coeffs_check_surface = Some(
                check_surface_order
                    .iter()
                    .map(|&c| ncoeffs_kifmm(c))
                    .collect_vec(),
            );

            self.equivalent_surface_order = Some(equivalent_surface_order.to_vec());
            self.check_surface_order = Some(check_surface_order.to_vec());

            self.kernel = Some(kernel);
            self.fmm_eval_type = Some(FmmEvalType::Vector);
            self.kernel_eval_type = Some(eval_type);
            self.isa = Some(Isa::new());
            self.source_to_target = Some(source_to_target);
            self.charges = Some(charges.to_vec()); // un-ordered charges
            Ok(self)
        }
    }

    /// Initialise
    pub fn build(self) -> Result<KiFmmMulti<Scalar, Kernel, FieldTranslation>, std::io::Error> {
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
            let neighbourhood_communicator_charge =
                NeighbourhoodCommunicator::from_comm(&communicator);
            let rank = communicator.rank();
            let source_to_target = self.source_to_target.unwrap();
            let fmm_eval_type = self.fmm_eval_type.unwrap();
            let kernel_eval_type = self.kernel_eval_type.unwrap();
            let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface.unwrap();
            let n_coeffs_check_surface = self.n_coeffs_check_surface.unwrap();
            let equivalent_surface_order = self.equivalent_surface_order.unwrap();
            let check_surface_order = self.check_surface_order.unwrap();
            let communication_times = self.communication_times.unwrap();
            let timed = self.timed.unwrap();
            let global_depth = self.tree.as_ref().unwrap().source_tree.global_depth;
            let total_depth = self.tree.as_ref().unwrap().source_tree.total_depth;
            let variable_expansion_order = self.variable_expansion_order.unwrap();

            let global_equivalent_surface_order =
                equivalent_surface_order[0..=(global_depth as usize)].to_vec();
            let global_check_surface_order =
                check_surface_order[0..=(global_depth as usize)].to_vec();
            let global_n_coeffs_equivalent_surface =
                n_coeffs_equivalent_surface[0..=(global_depth as usize)].to_vec();
            let global_n_coeffs_check_surface =
                n_coeffs_check_surface[0..=(global_depth as usize)].to_vec();

            let local_equivalent_surface_order =
                equivalent_surface_order[(global_depth as usize)..=(total_depth as usize)].to_vec();
            let local_check_surface_order =
                check_surface_order[(global_depth as usize)..=(total_depth as usize)].to_vec();
            let local_n_coeffs_equivalent_surface = n_coeffs_equivalent_surface
                [(global_depth as usize)..=(total_depth as usize)]
                .to_vec();
            let local_n_coeffs_check_surface =
                n_coeffs_check_surface[(global_depth as usize)..=(total_depth as usize)].to_vec();

            let global_fmm: KiFmm<Scalar, Kernel, FieldTranslation> = KiFmm {
                isa: self.isa.unwrap(),
                equivalent_surface_order: global_equivalent_surface_order,
                check_surface_order: global_check_surface_order,
                variable_expansion_order,
                n_coeffs_equivalent_surface: global_n_coeffs_equivalent_surface,
                n_coeffs_check_surface: global_n_coeffs_check_surface,
                fmm_eval_type,
                kernel_eval_type,
                kernel: kernel.clone(),
                dim: 3,
                timed,
                ..Default::default()
            };

            let ghost_fmm_v: KiFmm<Scalar, Kernel, FieldTranslation> = KiFmm {
                isa: self.isa.unwrap(),
                equivalent_surface_order: local_equivalent_surface_order.to_vec(),
                check_surface_order: local_check_surface_order.to_vec(),
                variable_expansion_order,
                n_coeffs_equivalent_surface: local_n_coeffs_equivalent_surface.to_vec(),
                n_coeffs_check_surface: local_check_surface_order.to_vec(),
                fmm_eval_type,
                kernel_eval_type,
                kernel: kernel.clone(),
                dim: 3,
                timed,
                ..Default::default()
            };

            let ghost_fmm_u: KiFmm<Scalar, Kernel, FieldTranslation> = KiFmm {
                isa: self.isa.unwrap(),
                equivalent_surface_order: local_equivalent_surface_order.to_vec(),
                check_surface_order: local_check_surface_order.to_vec(),
                variable_expansion_order,
                n_coeffs_equivalent_surface: local_n_coeffs_equivalent_surface,
                n_coeffs_check_surface: local_n_coeffs_check_surface,
                fmm_eval_type,
                kernel_eval_type,
                kernel: kernel.clone(),
                dim: 3,
                timed,
                ..Default::default()
            };

            let mut result = KiFmmMulti {
                timed,
                dim: 3,
                variable_expansion_order,
                communication_times,
                isa: self.isa.unwrap(),
                communicator,
                neighbourhood_communicator_v,
                neighbourhood_communicator_u,
                neighbourhood_communicator_charge,
                rank,
                kernel,
                tree: self.tree.unwrap(),
                equivalent_surface_order,
                check_surface_order,
                n_coeffs_equivalent_surface,
                n_coeffs_check_surface,
                source_to_target,
                fmm_eval_type,
                kernel_eval_type,
                global_fmm,
                ghost_fmm_v,
                ghost_fmm_u,
                kernel_eval_size: 1,
                source: Vec::default(),
                operator_times: Vec::default(),
                charges: Vec::default(),
                charge_index_pointer_sources: Vec::default(),
                charge_index_pointer_targets: Vec::default(),
                leaf_upward_check_surfaces_sources: Vec::default(),
                leaf_downward_equivalent_surfaces_targets: Vec::default(),
                leaf_upward_equivalent_surfaces_sources: Vec::default(),
                uc2e_inv_1: Vec::default(),
                uc2e_inv_2: Vec::default(),
                dc2e_inv_1: Vec::default(),
                dc2e_inv_2: Vec::default(),
                source_vec: Vec::default(),
                target_vec: Vec::default(),
                multipoles: Vec::default(),
                locals: Vec::default(),
                potentials: Vec::default(),
                leaf_multipoles: Vec::default(),
                level_multipoles: Vec::default(),
                leaf_locals: Vec::default(),
                level_locals: Vec::default(),
                level_index_pointer_locals: Vec::default(),
                level_index_pointer_multipoles: Vec::default(),
                potentials_send_pointers: Vec::default(),
                metadata_times: Vec::default(),
                ghost_requested_queries_v: Vec::default(),
                ghost_requested_queries_key_to_index_v: HashMap::default(),
                ghost_requested_queries_counts_v: Vec::default(),
                ghost_received_queries_displacements_v: Vec::default(),
                ghost_received_queries_counts_v: Vec::default(),
                ghost_received_queries_v: Vec::default(),
                local_count_charges: 0u64,
                local_displacement_charges: 0u64,
                charge_send_queries_counts: Vec::default(),
                charge_send_queries_displacements: Vec::default(),
                charge_receive_queries_counts: Vec::default(),
                charge_receive_queries_displacements: Vec::default(),
                ghost_received_queries_charge: Vec::default(),
                ghost_received_queries_charge_counts: Vec::default(),
                ghost_received_queries_charge_displacements: Vec::default(),
            };

            // Calculate required metadata
            let (_, duration) = optionally_time(timed, || result.source());

            if let Some(d) = duration {
                result
                    .metadata_times
                    .push(MetadataTime::from_duration(MetadataType::SourceData, d))
            }

            let (_, duration) = optionally_time(timed, || result.target());

            if let Some(d) = duration {
                result
                    .metadata_times
                    .push(MetadataTime::from_duration(MetadataType::TargetData, d))
            }

            // TODO
            let (_, duration) = optionally_time(timed, || result.source_to_target());

            if let Some(d) = duration {
                result.metadata_times.push(MetadataTime::from_duration(
                    MetadataType::SourceToTargetData,
                    d,
                ))
            }

            // Metadata for global FMM and FMM

            // // On nominated node only
            if result.communicator.rank() == 0 {
                result.global_fmm.set_source_tree(
                    &result.tree.domain,
                    result.tree.source_tree.global_depth,
                    result.tree.source_tree.all_roots.clone(),
                );

                result.global_fmm.set_target_tree(
                    &result.tree.domain,
                    result.tree.source_tree.global_depth,
                    result.tree.source_tree.all_roots.clone(),
                );

                result.global_fmm.level_index_pointer_multipoles =
                    level_index_pointer_single_node(&result.global_fmm.tree.source_tree);
                result.global_fmm.global_fmm_local_metadata();
                result.global_fmm.displacements(None);
            }

            result.metadata(self.kernel_eval_type.unwrap(), &self.charges.unwrap());
            result.displacements(None);

            Ok(result)
        }
    }
}
