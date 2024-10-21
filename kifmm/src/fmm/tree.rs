//! Implementation of FMM compatible trees
use num::traits::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::SingleNodeFmmTree,
    traits::tree::{SingleFmmTree, SingleTree},
    tree::types::SingleNodeTree,
};

impl<T> SingleFmmTree for SingleNodeFmmTree<T>
where
    T: RlstScalar + Float + Default,
{
    type Tree = SingleNodeTree<T>;

    fn source_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn target_tree(&self) -> &Self::Tree {
        &self.target_tree
    }

    fn domain(&self) -> &<Self::Tree as SingleTree>::Domain {
        &self.domain
    }
}

unsafe impl<T: RlstScalar + Float + Default> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: RlstScalar + Float + Default> Sync for SingleNodeFmmTree<T> {}
