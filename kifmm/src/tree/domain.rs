//! Constructor for a single node Domain.
use crate::{traits::tree::Domain as DomainTrait, tree::types::Domain};
use itertools::Itertools;
#[allow(unused_imports)]
#[cfg(feature = "mpi")]
pub use mpi_domain::*;
use num::Float;
use rlst::RlstScalar;

impl<T> Domain<T>
where
    T: RlstScalar + Float,
{
    /// Compute the domain defined by a set of points on a local node. When defined by a set of points
    /// The domain adds a small threshold such that no points lie on the actual edge of the domain to
    /// ensure correct Morton encoding.
    ///
    /// # Arguments
    /// * `points` - A slice of point coordinates, expected in column major order  [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
    pub fn from_local_points(coordinates: &[T]) -> Domain<T> {
        let xs = coordinates.iter().step_by(3).cloned().collect_vec();
        let ys = coordinates.iter().skip(1).step_by(3).cloned().collect_vec();
        let zs = coordinates.iter().skip(2).step_by(3).cloned().collect_vec();

        let max_x = xs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_y = ys.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_z = zs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let min_x = xs.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_y = ys.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_z = zs.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        // Find maximum dimension, this will define the size of the boxes in the domain
        let diameter_x = Float::abs(*max_x - *min_x);
        let diameter_y = Float::abs(*max_y - *min_y);
        let diameter_z = Float::abs(*max_z - *min_z);

        // Want a cubic box to place everything in
        let diameter = diameter_x.max(diameter_y).max(diameter_z);

        // Increase size of bounding box by 1% along each dimension to capture all points
        let err_fraction = T::from(0.005).unwrap();
        let err = diameter * err_fraction;

        let two = T::from(2.0).unwrap();
        let diameter = [
            diameter + two * err,
            diameter + two * err,
            diameter + two * err,
        ];

        // The origin is defined by the minimum point
        let origin = [*min_x - err, *min_y - err, *min_z - err];

        Domain {
            origin,
            side_length: diameter,
        }
    }

    /// Find the union of two domains such that the returned domain is a superset and contains both sets of corresponding points
    ///
    /// # Arguments
    /// * `other` - Other domain with which to find union
    pub fn union(&self, other: &Self) -> Self {
        // Find minimum origin
        let min_x = self.origin[0].min(other.origin[0]);
        let min_y = self.origin[1].min(other.origin[1]);
        let min_z = self.origin[2].min(other.origin[2]);

        let min_origin = [min_x, min_y, min_z];

        // Find maximum diameter (+max origin)
        let max_x = self.side_length[0].max(other.side_length[0]);
        let max_y = self.side_length[0].max(other.side_length[0]);
        let max_z = self.side_length[0].max(other.side_length[0]);

        let max_diameter = [max_x, max_y, max_z];

        Domain {
            origin: min_origin,
            side_length: max_diameter,
        }
    }

    /// Construct a domain a user specified origin and diameter.
    ///
    /// # Arguments
    /// * `origin` - The point from which to construct a cuboid domain.
    /// * `diameter` - The diameter along each axis of the domain.
    pub fn new(origin: &[T; 3], side_length: &[T; 3]) -> Self {
        Domain {
            origin: *origin,
            side_length: *side_length,
        }
    }
}

impl<T> DomainTrait for Domain<T>
where
    T: RlstScalar + Float,
{
    type Scalar = T;

    fn side_length(&self) -> &[T; 3] {
        &self.side_length
    }

    fn origin(&self) -> &[T; 3] {
        &self.origin
    }
}

#[cfg(feature = "mpi")]
mod mpi_domain {

    use super::{Float, RlstScalar};

    use super::Domain;
    use memoffset::offset_of;
    use mpi::{
        datatype::{UncommittedUserDatatype, UserDatatype},
        traits::{Buffer, BufferMut, Communicator, CommunicatorCollectives, Equivalence},
        Address,
    };

    unsafe impl<T> Equivalence for Domain<T>
    where
        T: RlstScalar + Float + Equivalence,
    {
        type Out = UserDatatype;
        fn equivalent_datatype() -> Self::Out {
            UserDatatype::structured(
                &[1, 1],
                &[
                    offset_of!(Domain<T>, origin) as Address,
                    offset_of!(Domain<T>, side_length) as Address,
                ],
                &[
                    UncommittedUserDatatype::contiguous(3, &T::equivalent_datatype()).as_ref(),
                    UncommittedUserDatatype::contiguous(3, &T::equivalent_datatype()).as_ref(),
                ],
            )
        }
    }

    impl<T> Domain<T>
    where
        [Domain<T>]: BufferMut,
        Vec<Domain<T>>: Buffer,
        T: RlstScalar + Float,
    {
        /// Compute the points domain over all nodes by computing `local' domains on each MPI process, communicating the bounds
        /// globally and using the local domains to create a globally defined domain. Relies on an `all to all` communication.
        ///
        /// # Arguments
        /// * `local_points` - A slice of point coordinates, expected in column major order  [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
        /// * `comm` - An MPI (User) communicator over which the domain is defined.
        pub fn from_global_points<C: Communicator>(local_points: &[T], comm: &C) -> Domain<T> {
            let size = comm.size();

            let local_domain = Domain::<T>::from_local_points(local_points);
            let local_bounds: Vec<Domain<T>> = vec![local_domain; size as usize];
            let zero = T::zero();
            let one = T::one();
            let mut buffer =
                vec![Domain::<T>::new(&[zero, zero, zero], &[one, one, one]); size as usize];

            comm.all_to_all_into(&local_bounds, &mut buffer[..]);

            // Find minimum origin
            let min_x = buffer
                .iter()
                .min_by(|a, b| a.origin[0].partial_cmp(&b.origin[0]).unwrap())
                .unwrap()
                .origin[0];
            let min_y = buffer
                .iter()
                .min_by(|a, b| a.origin[1].partial_cmp(&b.origin[1]).unwrap())
                .unwrap()
                .origin[1];
            let min_z = buffer
                .iter()
                .min_by(|a, b| a.origin[2].partial_cmp(&b.origin[2]).unwrap())
                .unwrap()
                .origin[2];

            let min_origin = [min_x, min_y, min_z];

            // Find maximum diameter (+max origin)
            let max_x = buffer
                .iter()
                .max_by(|a, b| a.side_length[0].partial_cmp(&b.side_length[0]).unwrap())
                .unwrap()
                .side_length[0];
            let max_y = buffer
                .iter()
                .max_by(|a, b| a.side_length[1].partial_cmp(&b.side_length[1]).unwrap())
                .unwrap()
                .side_length[1];
            let max_z = buffer
                .iter()
                .max_by(|a, b| a.side_length[2].partial_cmp(&b.side_length[2]).unwrap())
                .unwrap()
                .side_length[2];

            let max_diameter = [max_x, max_y, max_z];

            Domain {
                origin: min_origin,
                side_length: max_diameter,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tree::helpers::{points_fixture, points_fixture_col, PointsMat};
    use rlst::{RawAccess, Shape};

    fn test_compute_bounds<T>(points: PointsMat<T>)
    where
        T: RlstScalar + Float,
    {
        let domain = Domain::<T>::from_local_points(points.data());

        // Test that the domain remains cubic
        assert!(domain
            .side_length
            .iter()
            .all(|&x| x == domain.side_length[0]));

        // Test that all local points are contained within the local domain
        let n_points = points.shape()[0];
        for i in 0..n_points {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];

            assert!(
                domain.origin[0] <= point[0]
                    && point[0] <= domain.origin[0] + domain.side_length[0]
            );
            assert!(
                domain.origin[1] <= point[1]
                    && point[1] <= domain.origin[1] + domain.side_length[1]
            );
            assert!(
                domain.origin[2] <= point[2]
                    && point[2] <= domain.origin[2] + domain.side_length[2]
            );
        }
    }

    #[test]
    fn test_bounds() {
        let n_points = 10000;

        // Test points in positive octant only
        let points = points_fixture::<f64>(n_points, None, None, None);
        test_compute_bounds::<f64>(points);

        // Test points in positive and negative octants
        let points = points_fixture::<f64>(n_points, Some(-1.), Some(1.), None);
        test_compute_bounds::<f64>(points);

        // Test rectangular distributions of points
        let points = points_fixture_col::<f64>(n_points);
        test_compute_bounds::<f64>(points);
    }
}
