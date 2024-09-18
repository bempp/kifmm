//! Should eventually store all parallel traits

/// Interface for ghost exchange implementations
pub trait GhostExchange {
    /// Exchange V list data, must be done at runtime as it relies on node existence
    fn v_list_exchange(&mut self);

    /// Exchange U list data, can be done during pre-computation
    fn u_list_exchange(&mut self);

    /// Gather root multipoles from local source trees at nominated node
    fn gather_global_fmm_at_root(&mut self);

    /// Scatter root locals back to local target trees
    fn scatter_global_fmm_from_root(&mut self);
}
