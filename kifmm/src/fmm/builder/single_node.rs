//! Builder objects to construct FMMs
use std::collections::HashSet;

use green_kernels::{traits::Kernel as KernelTrait, types::GreenKernelEvalType};
use itertools::Itertools;
use rlst::{MatrixSvd, RlstScalar};

use crate::{
    fmm::{
        helpers::single_node::{map_charges, ncoeffs_kifmm},
        types::{FmmEvalType, Isa, KiFmm, SingleNodeBuilder, SingleNodeFmmTree},
    },
    traits::{
        field::{
            FieldTranslation as FieldTranslationTrait, SourceToTargetTranslationMetadata,
            SourceTranslationMetadata, TargetTranslationMetadata,
        },
        fmm::{HomogenousKernel, Metadata},
        general::single_node::Epsilon,
        tree::{SingleFmmTree, SingleTree},
    },
    tree::{types::Domain, SingleNodeTree},
};

impl<Scalar, Kernel, FieldTranslation> SingleNodeBuilder<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd,
    <Scalar as RlstScalar>::Real: Default + Epsilon,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Clone + Default,
    FieldTranslation: FieldTranslationTrait + Default,
    KiFmm<Scalar, Kernel, FieldTranslation>: SourceToTargetTranslationMetadata
        + SourceTranslationMetadata
        + TargetTranslationMetadata
        + Metadata<Scalar = Scalar>,
{
    /// Initialise an empty kernel independent FMM builder
    pub fn new() -> Self {
        Self {
            isa: None,
            tree: None,
            kernel: None,
            charges: None,
            source_to_target: None,
            domain: None,
            equivalent_surface_order: None,
            check_surface_order: None,
            variable_expansion_order: None,
            n_coeffs_equivalent_surface: None,
            n_coeffs_check_surface: None,
            kernel_eval_type: None,
            fmm_eval_type: None,
            depth_set: None,
        }
    }

    /// Associate FMM builder with an FMM Tree
    ///
    /// # Arguments
    /// * `sources` - Source coordinates, data expected in row major order such that the shape is [dim, n_coords]
    /// * `target` - Target coordinates,  data expected in row major order such that the shape is [dim, n_coords]
    /// * `n_crit` - Maximum number of particles per leaf box, if none specified a default of 150 is used.
    /// * `prune_empty` - Optionally drop empty leaf boxes for performance.`
    pub fn tree(
        mut self,
        sources: &[Scalar::Real],
        targets: &[Scalar::Real],
        n_crit: Option<u64>,
        depth: Option<u64>,
        prune_empty: bool,
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
            let source_domain = Domain::from_local_points(sources);
            let target_domain = Domain::from_local_points(targets);

            // Calculate union of domains for source and target points, needed to define operators
            let domain = source_domain.union(&target_domain);
            self.domain = Some(domain);

            let source_depth;
            let target_depth;

            if depth.is_some() && n_crit.is_none() {
                source_depth = depth.unwrap();
                target_depth = depth.unwrap();
                self.depth_set = Some(true);
            } else if depth.is_none() && n_crit.is_some() {
                // Estimate depth based on a uniform distribution
                source_depth = SingleNodeTree::<Scalar::Real>::minimum_depth(
                    n_sources as u64,
                    n_crit.unwrap(),
                );
                target_depth = SingleNodeTree::<Scalar::Real>::minimum_depth(
                    n_targets as u64,
                    n_crit.unwrap(),
                );
                self.depth_set = Some(false);
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Either of `ncrit` or `depth` must be supplied, not both or neither.",
                ));
            }

            let depth = source_depth.max(target_depth); // refine source and target trees to same depth

            let source_tree =
                SingleNodeTree::new(sources, depth, prune_empty, self.domain, None, None)?;
            let target_tree =
                SingleNodeTree::new(targets, depth, prune_empty, self.domain, None, None)?;

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
    /// * `charges` - 2D RLST array, of dimensions `[n_charges, n_vecs]` where each of `n_vecs` is associated with `n_charges`
    /// * `expansion_order` - The expansion order of the FMM
    /// * `kernel` - The kernel associated with this FMM
    /// * `eval_type` - Either `ValueDeriv` - to evaluate potentials and gradients, or `Value` to evaluate potentials alone
    /// * `source_to_target` - A field translation method.
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
            // Set FMM parameters
            let global_indices = self
                .tree
                .as_ref()
                .unwrap()
                .source_tree
                .all_global_indices()
                .unwrap();

            let ncharges = &self
                .tree
                .as_ref()
                .unwrap()
                .source_tree()
                .n_coordinates_tot()
                .unwrap();
            let n_matvecs = charges.len() / ncharges;

            self.charges = Some(map_charges(global_indices, charges, n_matvecs));

            if n_matvecs > 1 {
                self.fmm_eval_type = Some(FmmEvalType::Matrix(n_matvecs))
            } else {
                self.fmm_eval_type = Some(FmmEvalType::Vector)
            }

            let depth = self.tree.as_ref().unwrap().source_tree().depth();
            let depth_set = self.depth_set.unwrap();

            let expected_len = if depth_set { (depth + 1) as usize } else { 1 };

            if expansion_order.len() != expected_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Number of expansion orders must either be 1, or match the depth of the tree",
                ));
            }

            let unique_expansion_orders: HashSet<_> = expansion_order.iter().cloned().collect();

            if unique_expansion_orders.len() > 1 {
                self.variable_expansion_order = Some(true)
            } else {
                self.variable_expansion_order = Some(false)
            }

            let check_surface_order = if source_to_target.overdetermined() {
                expansion_order
                    .iter()
                    .map(|&e| e + source_to_target.surface_diff())
                    .collect_vec()
            } else {
                expansion_order.to_vec()
            };

            self.n_coeffs_equivalent_surface = Some(
                expansion_order
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

            self.isa = Some(Isa::new());
            self.equivalent_surface_order = Some(expansion_order.to_vec());
            self.check_surface_order = Some(check_surface_order.to_vec());
            self.kernel = Some(kernel);
            self.kernel_eval_type = Some(eval_type);
            self.source_to_target = Some(source_to_target);

            Ok(self)
        }
    }

    /// Finalize and build the single node FMM
    pub fn build(self) -> Result<KiFmm<Scalar, Kernel, FieldTranslation>, std::io::Error> {
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
                isa: self.isa.unwrap(),
                tree: self.tree.unwrap(),
                equivalent_surface_order: self.equivalent_surface_order.unwrap(),
                check_surface_order: self.check_surface_order.unwrap(),
                variable_expansion_order: self.variable_expansion_order.unwrap(),
                n_coeffs_equivalent_surface: self.n_coeffs_equivalent_surface.unwrap(),
                n_coeffs_check_surface: self.n_coeffs_check_surface.unwrap(),
                source_to_target: self.source_to_target.unwrap(),
                fmm_eval_type: self.fmm_eval_type.unwrap(),
                kernel_eval_type: self.kernel_eval_type.unwrap(),
                kernel,
                dim,
                ..Default::default()
            };

            // Calculate required metadata
            result.source();
            result.target();
            result.source_to_target();
            result.metadata(self.kernel_eval_type.unwrap(), &self.charges.unwrap());
            SourceToTargetTranslationMetadata::displacements(&mut result, None, None);
            Ok(result)
        }
    }
}
