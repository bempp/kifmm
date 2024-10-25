//! Should eventually store all parallel traits

use rlst::RlstScalar;

use crate::{
    traits::{
        field::SourceToTargetTranslationMetadata,
        fmm::DataAccess,
        tree::{SingleFmmTree, SingleTree},
    },
    tree::types::Domain,
};

/// Interface for ghost exchange implementations
pub trait GhostExchange {
    /// Exchange V List keys and metadata, which can be done during pre-computation
    fn v_list_exchange(&mut self);

    /// Exchange V list data, must be done at runtime as relies on partially completed upward pass
    /// Only the actual multipole coefficients have to be exchanged at runtime, but this isn't required
    /// for calculating the displacements
    fn v_list_exchange_runtime(&mut self);

    /// Exchange U list data, can be done during pre-computation
    fn u_list_exchange(&mut self);

    /// Gather root multipoles from local source trees at nominated node
    fn gather_global_fmm_at_root(&mut self);

    /// Scatter root locals back to local target trees
    fn scatter_global_fmm_from_root(&mut self);
}

/// Set metadata for global FMM when received in the form of Ghost data
pub trait GlobalFmmMetadata
where
    Self: DataAccess + SourceToTargetTranslationMetadata,
{
    /// Set the source tree from some leaves
    fn set_source_tree(
        &mut self,
        domain: &Domain<<Self::Scalar as RlstScalar>::Real>,
        depth: u64,
        leaves: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
    );

    /// Set the target tree from some leaves
    fn set_target_tree(
        &mut self,
        domain: &Domain<<Self::Scalar as RlstScalar>::Real>,
        depth: u64,
        leaves: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
    );

    /// Set data associated with received multipoles for global tree
    #[allow(clippy::too_many_arguments)]
    fn global_fmm_multipole_metadata(
        &mut self,
        leaves: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        multipoles: Vec<Self::Scalar>,
    );

    /// Set data associated with received locals for global tree
    #[allow(clippy::too_many_arguments)]
    fn global_fmm_local_metadata(&mut self);
}
