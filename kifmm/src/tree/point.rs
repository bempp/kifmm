//! Implementation of traits for handling, and sorting, containers of point data.
use crate::tree::types::Point;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use num::Float;
use rlst::RlstScalar;

impl<T> PartialEq for Point<T>
where
    T: RlstScalar + Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.encoded_key == other.encoded_key
    }
}

impl<T> Eq for Point<T> where T: RlstScalar + Float {}

impl<T> Ord for Point<T>
where
    T: RlstScalar + Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.encoded_key.cmp(&other.encoded_key)
    }
}

impl<T> PartialOrd for Point<T>
where
    T: RlstScalar + Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // less_than(&self.morton, &other.morton)
        Some(self.encoded_key.cmp(&other.encoded_key))
    }
}

impl<T> Hash for Point<T>
where
    T: RlstScalar + Float,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.encoded_key.hash(state);
    }
}

#[cfg(feature = "mpi")]
mod mpi_point {
    use super::Point;

    use crate::{tree::types::MortonKey, RlstScalarFloat};
    use memoffset::offset_of;
    use mpi::{
        datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
        Address,
    };
    use num::Float;

    unsafe impl<T> Equivalence for Point<T>
    where
        T: RlstScalar + Float + Equivalence,
    {
        type Out = UserDatatype;
        fn equivalent_datatype() -> Self::Out {
            UserDatatype::structured(
                &[1, 1, 1, 1],
                &[
                    offset_of!(Point<T>, coordinate) as Address,
                    offset_of!(Point<T>, global_index) as Address,
                    offset_of!(Point<T>, base_key) as Address,
                    offset_of!(Point<T>, encoded_key) as Address,
                ],
                &[
                    UncommittedUserDatatype::contiguous(3, &T::equivalent_datatype()).as_ref(),
                    UncommittedUserDatatype::contiguous(1, &usize::equivalent_datatype()).as_ref(),
                    UncommittedUserDatatype::structured(
                        &[1, 1],
                        &[
                            offset_of!(MortonKey<T>, anchor) as Address,
                            offset_of!(MortonKey<T>, morton) as Address,
                        ],
                        &[
                            UncommittedUserDatatype::contiguous(3, &u64::equivalent_datatype())
                                .as_ref(),
                            UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                                .as_ref(),
                        ],
                    )
                    .as_ref(),
                    UncommittedUserDatatype::structured(
                        &[1, 1],
                        &[
                            offset_of!(MortonKey<T>, anchor) as Address,
                            offset_of!(MortonKey<T>, morton) as Address,
                        ],
                        &[
                            UncommittedUserDatatype::contiguous(3, &u64::equivalent_datatype())
                                .as_ref(),
                            UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                                .as_ref(),
                        ],
                    )
                    .as_ref(),
                ],
            )
        }
    }
}

#[allow(unused_imports)]
#[cfg(feature = "mpi")]
pub use mpi_point::*;

#[cfg(test)]
mod test {
    use crate::tree::constants::DEEPEST_LEVEL;
    use crate::tree::helpers::points_fixture;
    use crate::tree::types::{Domain, MortonKey, Point};
    use rlst::RawAccess;

    #[test]
    pub fn test_ordering() {
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);

        let n_points = 1000;
        let coords = points_fixture(n_points, None, None, None);
        let mut points = Vec::new();

        for i in 0..n_points {
            let p = [
                coords.data()[i],
                coords.data()[i + n_points],
                coords.data()[i + 2 * n_points],
            ];
            points.push(Point {
                coordinate: p,
                base_key: MortonKey::from_point(&p, &domain, DEEPEST_LEVEL),
                encoded_key: MortonKey::from_point(&p, &domain, DEEPEST_LEVEL),
                global_index: i,
            })
        }

        points.sort();

        for i in 0..(points.len() - 1) {
            let a = &points[i];
            let b = &points[i + 1];
            assert!(a <= b);
        }
    }
}
