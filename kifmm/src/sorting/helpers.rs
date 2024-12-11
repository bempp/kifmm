//! Sorting helper functions


/// Find indices that sort a slice
pub fn argsort<T: Ord>(slice: &[T]) -> Vec<usize> {
    let mut indices = (0..slice.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &slice[i]);
    indices
}