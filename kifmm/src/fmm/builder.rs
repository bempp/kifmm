//! Builder objects to construct FMMs
use green_kernels::{traits::Kernel as KernelTrait, types::EvalType};
use rlst::{
    empty_array, rlst_dynamic_array2, Array, BaseArray, MatrixSvd, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar, Shape, VectorContainer,
};

use crate::{
    fmm::{
        constants::DEFAULT_NCRIT,
        helpers::{
            coordinate_index_pointer, homogenous_kernel_scale, leaf_expansion_pointers,
            leaf_scales, leaf_surfaces, level_expansion_pointers, level_index_pointer, map_charges,
            ncoeffs_kifmm, potential_pointers,
        },
        pinv::pinv,
        types::{Charges, Coordinates, FmmEvalType, KiFmm, SingleNodeBuilder, SingleNodeFmmTree},
    },
    traits::{
        field::{
            ConfigureSourceToTargetData, KernelMetadataFieldTranslation,
            KernelMetadataSourceTarget, SourceToTargetData as SourceToTargetDataTrait,
        },
        fmm::FmmMetadata,
        general::Epsilon,
        tree::{FmmTreeNode, Tree},
    },
    tree::{
        constants::{ALPHA_INNER, ALPHA_OUTER},
        types::{Domain, MortonKey, SingleNodeTree},
    },
};

impl<Scalar, Kernel, SourceToTargetData> SingleNodeBuilder<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Epsilon,
    <Scalar as RlstScalar>::Real: Default + Epsilon,
    Kernel: KernelTrait<T = Scalar> + Clone + Default,
    SourceToTargetData: SourceToTargetDataTrait
        + Default,
    Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>: MatrixSvd<Item = Scalar>,
    KiFmm<Scalar, Kernel, SourceToTargetData>:
        KernelMetadataFieldTranslation + KernelMetadataSourceTarget + FmmMetadata<Scalar = Scalar>,
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
    /// * `source_to_target` - A field translation method.
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

            // Calculate required metadata
            result.source();
            result.target();
            result.field_translation();
            result.metadata(self.kernel_eval_type.unwrap(), &self.charges.unwrap());

            Ok(result)
        }
    }
}
