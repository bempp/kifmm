//! Implementation of traits for handling, and sorting, containers of point data.
use crate::tree::types::point::{Point, Points};
use rlst::RlstScalar;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

impl<T> PartialEq for Point<T>
where
    T: RlstScalar<Real = T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.encoded_key == other.encoded_key
    }
}

impl<T> Eq for Point<T> where T: RlstScalar<Real = T> {}

impl<T> Ord for Point<T>
where
    T: RlstScalar<Real = T>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.encoded_key.cmp(&other.encoded_key)
    }
}

impl<T> PartialOrd for Point<T>
where
    T: RlstScalar<Real = T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // less_than(&self.morton, &other.morton)
        Some(self.encoded_key.cmp(&other.encoded_key))
    }
}

impl<T> Hash for Point<T>
where
    T: RlstScalar<Real = T>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.encoded_key.hash(state);
    }
}

impl<T> Points<T>
where
    T: RlstScalar<Real = T>,
{
    /// Create new
    pub fn new() -> Points<T> {
        Points {
            points: Vec::new(),
            index: 0,
        }
    }

    /// Add a point
    pub fn add(&mut self, item: Point<T>) {
        self.points.push(item);
    }

    /// Sort points
    pub fn sort(&mut self) {
        self.points.sort();
    }
}

#[cfg(test)]
mod test {
    use crate::tree::constants::DEEPEST_LEVEL;
    use crate::tree::implementations::helpers::points_fixture;
    use crate::tree::types::{domain::Domain, morton::MortonKey, point::Point};
    use rlst::RawAccess;

    #[test]
    pub fn test_ordering() {
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        let npoints = 1000;
        let coords = points_fixture(npoints, None, None, None);
        let mut points = Vec::new();

        for i in 0..npoints {
            let p = [
                coords.data()[i],
                coords.data()[i + npoints],
                coords.data()[i + 2 * npoints],
            ];
            points.push(Point {
                coordinate: p,
                base_key: MortonKey::from_point(&p, &domain, DEEPEST_LEVEL),
                encoded_key: MortonKey::from_point(&p, &domain, DEEPEST_LEVEL),
                global_idx: i,
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