use mpi::{
    collective::CommunicatorCollectives, datatype::{Partition, PartitionMut}, topology::{Communicator, SimpleCommunicator}, traits::Equivalence, Count
};
use itertools::Itertools;
use num::Float;
use rlst::{rlst_dynamic_array2, MatrixSvd, RlstScalar};
use std::{collections::HashMap};

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
        }
    }

    /// Tree
    #[allow(clippy::too_many_arguments)]
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
            let (source_domain, d) =
                optionally_time(timed, || Domain::from_global_points(sources, comm));

            if let Some(d) = d {
                communication_times.push(CommunicationTime::from_duration(
                    CommunicationType::SourceDomain,
                    d,
                ))
            }

            let (target_domain, d) =
                optionally_time(timed, || Domain::from_global_points(targets, comm));

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
                    comm,
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
                    comm,
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

            self.communicator = Some(comm.duplicate());
            self.tree = Some(fmm_tree);
            self.communication_times = Some(communication_times);
            Ok(self)
        }
    }

    /// Parameters
    pub fn parameters(
        mut self,
        charges: &[Scalar],
        expansion_order: usize,
        kernel: Kernel,
        source_to_target: FieldTranslation,
    ) -> Result<Self, std::io::Error> {
        if self.tree.is_none() {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Must build tree before specifying FMM parameters",
            ))
        } else {

            // Impose local ordering on charges after input points have been sorted locally, before send
            let coordinate_sort_indices = &self
                .tree
                .as_ref()
                .unwrap()
                .source_tree
                .coordinate_sort_indices;

            // Global indices of charges to be sent alongside them to their new destination ranks
            let unsorted_global_indices = &self
                .tree
                .as_ref()
                .unwrap()
                .source_tree
                .unsorted_global_indices
                .iter()
                .map(|&i| i as i64)
                .collect_vec();

            // New destination ranks of charges
            let coordinate_destination_ranks= &self
                .tree
                .as_ref()
                .unwrap()
                .source_tree
                .coordinate_destination_ranks;

            let comm = &self.tree.as_ref().unwrap().source_tree.communicator;
            let size = comm.size();

            // Sort input charges by their destination ranks
            let charges = {
                let mut tmp = Vec::new();
                for &i in coordinate_sort_indices.iter() {
                    tmp.push(charges[i])
                }
                tmp
            };

            // Can now communicate charges to same destination ranks as their corresponding points
            let mut counts_snd = vec![0i32; size as usize];
            for &rank in coordinate_destination_ranks.iter() {
                counts_snd[rank as usize] += 1
            }

            let counts_snd_clone = counts_snd.iter().cloned().collect_vec();

            let displs_snd = counts_snd
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect_vec();

            let mut counts_recv = vec![0 as Count; size as usize];
            comm.all_to_all_into(&counts_snd, &mut counts_recv);
            let counts_recv_clone = counts_recv.iter().cloned().collect_vec();

            let displs_recv = counts_recv
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect_vec();

            let total = counts_recv.iter().sum::<Count>();

            // Send global indices
            let mut received_global_indices = vec![i64::default(); total as usize];
            let mut partition_received =
                PartitionMut::new(&mut received_global_indices[..], counts_recv, &displs_recv[..]);
            let partition_snd = Partition::new(&unsorted_global_indices[..], counts_snd, &displs_snd[..]);

            comm.all_to_all_varcount_into(&partition_snd, &mut partition_received);

            // Send charges
            let mut received_charges= vec![Scalar::default(); total as usize];
            let mut partition_received =
                PartitionMut::new(&mut received_charges[..], counts_recv_clone, &displs_recv[..]);
            let partition_snd = Partition::new(&charges[..], counts_snd_clone, &displs_snd[..]);

            comm.all_to_all_varcount_into(&partition_snd, &mut partition_received);

            let unsorted_global_indices = received_global_indices.into_iter().map(|i| i as usize).collect_vec();

            // Have now got both received charges, their corresponding global indices as well as the global indices after global sort

            let mut mapping = HashMap::new();
            for (global_index, charge) in unsorted_global_indices.iter().zip(received_charges.iter()) {
                mapping.insert(global_index, charge);
            }

            let global_indices = &self
                .tree
                .as_ref()
                .unwrap()
                .source_tree
                .global_indices;

            let mut charges = Vec::new();
            for global_index in global_indices.iter() {
                let charge = **mapping.get(global_index).unwrap();
                charges.push(charge);
            }

            println!("RANK {:?} N local sources IN BUILDER {:?}", comm.rank(), charges.len());

            self.charges = Some(charges);
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

            let tmp_arr = rlst_dynamic_array2!(Scalar, [1, 1]);
            let global_fmm: KiFmm<Scalar, Kernel, FieldTranslation> = KiFmm {
                isa: self.isa.unwrap(),
                equivalent_surface_order: vec![equivalent_surface_order],
                check_surface_order: vec![check_surface_order],
                variable_expansion_order: false,
                n_coeffs_equivalent_surface: vec![n_coeffs_check_surface],
                n_coeffs_check_surface: vec![n_coeffs_equivalent_surface],
                fmm_eval_type,
                kernel_eval_type,
                kernel: kernel.clone(),
                dim: 3,
                timed,
                ..Default::default()
            };

            let ghost_fmm_v: KiFmm<Scalar, Kernel, FieldTranslation> = KiFmm {
                isa: self.isa.unwrap(),
                equivalent_surface_order: vec![equivalent_surface_order],
                check_surface_order: vec![check_surface_order],
                variable_expansion_order: false,
                n_coeffs_equivalent_surface: vec![n_coeffs_check_surface],
                n_coeffs_check_surface: vec![n_coeffs_equivalent_surface],
                fmm_eval_type,
                kernel_eval_type,
                kernel: kernel.clone(),
                dim: 3,
                timed,
                ..Default::default()
            };

            let ghost_fmm_u: KiFmm<Scalar, Kernel, FieldTranslation> = KiFmm {
                isa: self.isa.unwrap(),
                equivalent_surface_order: vec![equivalent_surface_order],
                check_surface_order: vec![check_surface_order],
                variable_expansion_order: false,
                n_coeffs_equivalent_surface: vec![n_coeffs_check_surface],
                n_coeffs_check_surface: vec![n_coeffs_equivalent_surface],
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
                communication_times,
                isa: self.isa.unwrap(),
                communicator,
                neighbourhood_communicator_v,
                neighbourhood_communicator_u,
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
                source: tmp_arr,
                operator_times: Vec::default(),
                charges: Vec::default(),
                charge_index_pointer_sources: Vec::default(),
                charge_index_pointer_targets: Vec::default(),
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
                ghost_requested_queries_key_to_index_v: HashMap::default(),
                ghost_requested_queries_counts_v: Vec::default(),
                ghost_received_queries_displacements_v: Vec::default(),
                ghost_received_queries_counts_v: Vec::default(),
                ghost_received_queries_v: Vec::default(),
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

            let (_, duration) = optionally_time(timed, || result.source_to_target());

            if let Some(d) = duration {
                result.metadata_times.push(MetadataTime::from_duration(
                    MetadataType::SourceToTargetData,
                    d,
                ))
            }

            // Metadata for global FMM and FMM

            // On nominated node only
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

            // pass dummy charges for now.
            result.metadata(self.kernel_eval_type.unwrap(),  &self.charges.unwrap());
            result.displacements(None);

            Ok(result)
        }
    }
}
