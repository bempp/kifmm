//! Should eventually store all parallel traits

use std::collections::HashSet;

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
    /// Exchange V list data, must be done at runtime as relies on partially completed upward pass
    fn v_list_exchange(&mut self);

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
    /// Set data associated with received multipoles for global tree
    #[allow(clippy::too_many_arguments)]
    fn global_fmm_multipole_metadata(
        &mut self,
        domain: &Domain<<Self::Scalar as RlstScalar>::Real>,
        depth: u64,
        keys: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        keys_set: HashSet<
            <<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
        >,
        leaves: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        leaves_set: HashSet<
            <<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
        >,
        multipoles: Vec<Self::Scalar>,
    );

    /// Set data associated with received locals for global tree
    #[allow(clippy::too_many_arguments)]
    fn global_fmm_local_metadata(
        &mut self,
        domain: &Domain<<Self::Scalar as RlstScalar>::Real>,
        depth: u64,
        keys: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        keys_set: HashSet<
            <<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
        >,
        leaves: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        leaves_set: HashSet<
            <<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
        >,
        locals: Vec<Self::Scalar>,
    );
}