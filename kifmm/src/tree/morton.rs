//! Implementations of constructors and transformation methods for Morton keys, as well as traits for sorting, and handling containers of Morton keys.
use crate::fmm::helpers::single_node::ncoeffs_kifmm;
use crate::traits::tree::{FmmTreeNode, TreeNode};
use crate::tree::{
    constants::{
        BYTE_DISPLACEMENT, BYTE_MASK, DEEPEST_LEVEL, DIRECTIONS, LEVEL_DISPLACEMENT, LEVEL_MASK,
        LEVEL_SIZE, NINE_BIT_MASK, X_LOOKUP_DECODE, X_LOOKUP_ENCODE, Y_LOOKUP_DECODE,
        Y_LOOKUP_ENCODE, Z_LOOKUP_DECODE, Z_LOOKUP_ENCODE,
    },
    helpers::find_corners,
    types::{Domain, MortonKey, MortonKeys},
};
use itertools::{izip, Itertools};
use num::{Float, ToPrimitive};
use rlst::RlstScalar;
use std::marker::PhantomData;
use std::{
    cmp::Ordering,
    collections::HashSet,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    vec,
};

impl<T> MortonKeys<T>
where
    T: RlstScalar + Float,
{
    /// Instantiate Morton Keys
    pub fn new() -> MortonKeys<T> {
        MortonKeys {
            keys: Vec::new(),
            index: 0,
        }
    }

    /// Add a key
    pub fn add(&mut self, item: MortonKey<T>) {
        self.keys.push(item);
    }

    /// Linearizes a collection of Morton Keys by removing overlaps, prioritizing smaller keys in case of overlaps.
    ///
    /// This function processes a slice of Morton Keys to produce a new, owned vector of keys where any overlapping
    /// keys have been resolved. In scenarios where keys overlap, the smaller (by Morton order) keys are preferred
    /// and retained in the output vector. Requires a copy of the input keys.
    ///
    /// # Arguments
    ///
    /// - `keys` -, A slice of Morton Keys subject to linearization, ensuring uniqueness and non-overlap in the resulting set.
    pub fn linearize_keys(keys: &[MortonKey<T>]) -> Vec<MortonKey<T>> {
        let depth = keys.iter().map(|k| k.level()).max().unwrap();
        let mut key_set: HashSet<MortonKey<_>> = keys.iter().cloned().collect();

        for level in (0..=depth).rev() {
            let work_set: Vec<&MortonKey<_>> =
                keys.iter().filter(|&&k| k.level() == level).collect();

            for work_item in work_set.iter() {
                let mut ancestors = work_item.ancestors();
                ancestors.remove(work_item);
                for ancestor in ancestors.iter() {
                    if key_set.contains(ancestor) {
                        key_set.remove(ancestor);
                    }
                }
            }
        }

        let result: Vec<MortonKey<_>> = key_set.into_iter().collect();
        result
    }

    /// Ensures a 2:1 balance among a set of Morton Keys.
    ///
    /// This function applies a 2:1 balance constraint to a slice of Morton Keys, where neighbouring
    /// nodes are at most twice as large as each other in terms of side length.  It assumes the input
    /// keys form a 'complete' set, meaning there are no gaps in the spatial coverage they represent.
    /// The balancing process may add new keys to meet this criterion, with the resulting, balanced
    /// set of keys returned as a `HashSet`.
    ///
    /// # Arguments
    ///
    /// - `keys` - A slice of Morton Keys to enforce the 2:1 balance upon. The keys should represent a
    /// contiguous space without gaps.
    pub fn balance_keys(keys: &[MortonKey<T>]) -> HashSet<MortonKey<T>> {
        let mut balanced: HashSet<MortonKey<_>> = keys.iter().cloned().collect();
        let deepest_level = keys.iter().map(|key| key.level()).max().unwrap();

        for level in (0..=deepest_level).rev() {
            let work_list = balanced
                .iter()
                .filter(|&key| key.level() == level)
                .cloned()
                .collect_vec(); // each key has its siblings here at deepest level

            for key in work_list.iter() {
                let neighbors = key.neighbors();
                for neighbor in neighbors.iter() {
                    let parent = neighbor.parent();

                    if !balanced.contains(neighbor) && !balanced.contains(&parent) {
                        balanced.insert(parent);
                        if parent.level() > 0 {
                            for sibling in parent.siblings() {
                                balanced.insert(sibling);
                            }
                        }
                    }
                }
            }
        }
        balanced
    }

    /// Fills the spatial region between two Morton Keys with the minimal set of covering boxes.
    ///
    /// This function computes and returns the smallest set of Morton Keys necessary to fully cover
    /// the space between two given keys, `start` and `end`. These keys define the bounds of the region
    /// to be completed.
    ///
    /// # Arguments
    ///
    /// - `start` - The Morton Key defining the beginning of the region to complete.
    /// - `end` - The Morton Key marking the end of the region.
    pub fn complete_region(start: &MortonKey<T>, end: &MortonKey<T>) -> Vec<MortonKey<T>> {
        let mut start_ancestors: HashSet<MortonKey<_>> = start.ancestors();
        let mut end_ancestors: HashSet<MortonKey<_>> = end.ancestors();

        // Remove endpoints from ancestors
        start_ancestors.remove(start);
        end_ancestors.remove(end);

        let mut minimal_tree: Vec<MortonKey<_>> = Vec::new();
        let mut work_list: Vec<MortonKey<_>> =
            start.finest_ancestor(end).children().into_iter().collect();

        while let Some(current_item) = work_list.pop() {
            if (current_item > *start)
                & (current_item < *end)
                & !end_ancestors.contains(&current_item)
            {
                minimal_tree.push(current_item);
            } else if (start_ancestors.contains(&current_item))
                | (end_ancestors.contains(&current_item))
            {
                let mut children = current_item.children();
                work_list.append(&mut children);
            }
        }

        // Sort the minimal tree before returning
        minimal_tree.sort();
        minimal_tree
    }

    /// Complete the region between all elements in an vector of Morton keys that doesn't
    /// necessarily span the domain defined by its least and greatest nodes.
    pub fn complete(&mut self) {
        let a = self.keys.iter().min().unwrap();
        let b = self.keys.iter().max().unwrap();
        let completion = Self::complete_region(a, b);
        let start_val = vec![*a];
        let end_val = vec![*b];
        self.keys = start_val
            .into_iter()
            .chain(completion)
            .chain(end_val)
            .collect();
    }

    /// Wrapper for linearising a container of Morton Keys.
    pub fn linearize(&mut self) {
        self.keys = Self::linearize_keys(&self.keys);
    }

    /// Wrapper for sorting a container of Morton Keys.
    pub fn sort(&mut self) {
        self.keys.sort();
    }

    /// Enforce a 2:1 balance for a vector of Morton keys, and remove any overlaps.
    pub fn balance(&mut self) {
        self.keys = Self::balance_keys(self).into_iter().collect();
    }
}

impl<T> Deref for MortonKeys<T>
where
    T: RlstScalar + Float,
{
    type Target = Vec<MortonKey<T>>;

    fn deref(&self) -> &Self::Target {
        &self.keys
    }
}

impl<T> DerefMut for MortonKeys<T>
where
    T: RlstScalar + Float,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.keys
    }
}

impl<T> Iterator for MortonKeys<T>
where
    T: RlstScalar + Float,
{
    type Item = MortonKey<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.keys.len() {
            return None;
        }

        self.index += 1;
        self.keys.get(self.index).copied()
    }
}

impl<T> FromIterator<MortonKey<T>> for MortonKeys<T>
where
    T: RlstScalar + Float,
{
    fn from_iter<I: IntoIterator<Item = MortonKey<T>>>(iter: I) -> Self {
        let mut c = MortonKeys::new();

        for i in iter {
            c.add(i);
        }
        c
    }
}

impl<T> From<Vec<MortonKey<T>>> for MortonKeys<T>
where
    T: RlstScalar + Float,
{
    fn from(keys: Vec<MortonKey<T>>) -> Self {
        MortonKeys { keys, index: 0 }
    }
}

impl<T> From<HashSet<MortonKey<T>>> for MortonKeys<T>
where
    T: RlstScalar + Float,
{
    fn from(keys: HashSet<MortonKey<T>>) -> Self {
        MortonKeys {
            keys: keys.into_iter().collect_vec(),
            index: 0,
        }
    }
}

impl<T> PartialEq for MortonKey<T>
where
    T: RlstScalar + Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.morton == other.morton
    }
}
impl<T> Eq for MortonKey<T> where T: RlstScalar + Float {}

impl<T> Ord for MortonKey<T>
where
    T: RlstScalar + Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.morton.cmp(&other.morton)
    }
}

impl<T> PartialOrd for MortonKey<T>
where
    T: RlstScalar + Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.morton.cmp(&other.morton))
    }
}

impl<T> Hash for MortonKey<T>
where
    T: RlstScalar + Float,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.morton.hash(state);
    }
}

/// Helper function for decoding keys.
fn decode_key_helper(key: u64, lookup_table: &[u64; 512]) -> u64 {
    const N_LOOPS: u64 = 7; // 8 bytes in 64 bit key
    let mut coord: u64 = 0;

    for index in 0..N_LOOPS {
        coord |= lookup_table[((key >> (index * 9)) & NINE_BIT_MASK) as usize] << (3 * index);
    }

    coord
}

/// Decodes a Morton key to retrieve its spatial anchor point.
///
/// # Arguments
///
/// - `morton` - The Morton key to be decoded.
///
/// # Returns
///
/// An array of three u64 values representing the x, y, and z coordinates of the anchor.
fn decode_key(morton: u64) -> [u64; 3] {
    let key = morton >> LEVEL_DISPLACEMENT;

    let x = decode_key_helper(key, &X_LOOKUP_DECODE);
    let y = decode_key_helper(key, &Y_LOOKUP_DECODE);
    let z = decode_key_helper(key, &Z_LOOKUP_DECODE);

    [x, y, z]
}

/// Map a point to the anchor of the enclosing box.
///
/// Returns the 3 integer coordinates of the enclosing box.
///
/// # Arguments
/// * `point` - The (x, y, z) coordinates of the point to map.
/// * `level` - The level of the tree at which the point will be mapped.
/// * `domain` - The computational domain defined by the point set.
fn point_to_anchor<T: RlstScalar + Float + ToPrimitive>(
    point: &[T; 3],
    level: u64,
    domain: &Domain<T>,
) -> Result<[u64; 3], std::io::Error> {
    // Check if point is in the domain

    let mut tmp = Vec::new();
    for (&p, d, o) in izip!(point, domain.side_length, domain.origin) {
        tmp.push((o <= p) && (p <= o + d));
    }
    let contained = tmp.iter().all(|&x| x);

    match contained {
        true => {
            let mut anchor = [u64::default(); 3];

            let side_length = domain
                .side_length
                .iter()
                .map(|d| *d / T::from(1 << level).unwrap())
                .collect_vec();

            let scaling_factor = 1 << (DEEPEST_LEVEL - level);

            for (a, p, o, s) in izip!(&mut anchor, point, &domain.origin, &side_length) {
                *a = (((*p - *o) / *s).floor()).to_u64().unwrap() * scaling_factor;
            }
            Ok(anchor)
        }
        false => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Point not in Domain",
        )),
    }
}

/// Encode an anchor.
/// Returns the Morton key associated with the given anchor.
///
/// # Arguments
/// * `anchor` - A vector with 3 elements defining the integer coordinates.
/// * `level` - The level of the tree the anchor is encoded to.
pub fn encode_anchor(anchor: &[u64; 3], level: u64) -> u64 {
    let x = anchor[0];
    let y = anchor[1];
    let z = anchor[2];

    let key: u64 = X_LOOKUP_ENCODE[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
        | Y_LOOKUP_ENCODE[((y >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
        | Z_LOOKUP_ENCODE[((z >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

    let key = (key << 24)
        | X_LOOKUP_ENCODE[(x & BYTE_MASK) as usize]
        | Y_LOOKUP_ENCODE[(y & BYTE_MASK) as usize]
        | Z_LOOKUP_ENCODE[(z & BYTE_MASK) as usize];

    let key = key << LEVEL_DISPLACEMENT;
    key | level
}

impl<T> MortonKey<T>
where
    T: RlstScalar + Float,
{
    /// Constructor for Morton key
    pub fn new(anchor: &[u64; 3], morton: u64) -> Self {
        Self {
            anchor: *anchor,
            morton,
            scalar: PhantomData::<T>,
        }
    }

    /// The Morton key corresponding to an octree root node
    pub fn root() -> Self {
        Self::new(&[0, 0, 0], 0)
    }

    /// Construct a `MortonKey` type from a Morton index
    pub fn from_morton(morton: u64) -> Self {
        let anchor = decode_key(morton);
        Self::new(&anchor, morton)
    }

    /// Construct a `MortonKey` type from the anchor at a given level
    pub fn from_anchor(anchor: &[u64; 3], level: u64) -> Self {
        let morton = encode_anchor(anchor, level);
        Self::new(anchor, morton)
    }

    /// Construct a `MortonKey` associated with the box that encloses the point on the deepest level.
    ///
    /// # Arguments
    /// * `point` - Cartesian coordinate for a given point.
    /// * `domain` - Domain associated with a given tree encoding.
    /// * `level` - level of octree on which to find the encoding.
    pub fn from_point(point: &[T; 3], domain: &Domain<T>, level: u64) -> Self {
        let anchor = point_to_anchor(point, level, domain).unwrap();
        MortonKey::from_anchor(&anchor, level)
    }

    /// Find the transfer vector between two Morton keys in component form.
    ///
    /// # Arguments
    /// * `other` - A Morton Key with which to calculate a transfer vector to.
    pub fn find_transfer_vector_components(
        &self,
        &other: &MortonKey<T>,
    ) -> Result<[i64; 3], std::io::Error> {
        // Only valid for keys at level 2 and below
        if self.level() < 2 || other.level() < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Transfer vectors only computed for keys at levels deeper than 2",
            ));
        }

        let level_diff = DEEPEST_LEVEL - self.level();

        let a = decode_key(self.morton);
        let b = decode_key(other.morton);

        // Compute transfer vector
        let mut x = a[0] as i64 - b[0] as i64;
        let mut y = a[1] as i64 - b[1] as i64;
        let mut z = a[2] as i64 - b[2] as i64;

        // Convert to an absolute transfer vector, wrt to key level.
        x /= 2_i64.pow(level_diff as u32);
        y /= 2_i64.pow(level_diff as u32);
        z /= 2_i64.pow(level_diff as u32);

        Ok([x, y, z])
    }

    /// Subroutine for converting components of a transfer vector into a unique, positive, checksum.
    ///
    /// # Arguments
    /// * `components` - A three vector corresponding to a transfer vector.
    pub fn find_transfer_vector_from_components(components: &[i64]) -> usize {
        let mut x = components[0];
        let mut y = components[1];
        let mut z = components[2];

        fn positive_map(num: &mut i64) {
            if *num < 0 {
                *num = 2 * (-1 * *num) + 1;
            } else {
                *num *= 2;
            }
        }

        // Compute checksum via mapping to positive integers.
        positive_map(&mut x);
        positive_map(&mut y);
        positive_map(&mut z);

        let mut checksum = x;
        checksum = (checksum << 16) | y;
        checksum = (checksum << 16) | z;

        checksum as usize
    }

    /// Checksum encoding unique transfer vector between this key, and another.
    /// ie. the vector other->self returned as an unsigned integer.
    ///
    /// # Arguments
    /// * `other` - A Morton Key with which to calculate a transfer vector to.
    pub fn find_transfer_vector(&self, &other: &MortonKey<T>) -> Result<usize, std::io::Error> {
        let tmp = self.find_transfer_vector_components(&other)?;
        Ok(MortonKey::<T>::find_transfer_vector_from_components(&tmp))
    }

    /// The physical diameter of a box specified by this Morton Key, calculated with respect to
    /// a Domain. Returns a stack allocated array for the size of the box width in each corresponding
    /// dimension.
    ///
    /// # Arguments
    /// `domain` - The physical domain with which we calculate the diameter with respect to.
    pub fn diameter(&self, domain: &Domain<T>) -> [T; 3] {
        domain.side_length.map(|x| {
            RlstScalar::powf(T::from(0.5).unwrap(), T::from(self.level()).unwrap().re()) * x
        })
    }

    /// The physical centre of a box specified by this Morton Key, calculated with respect to
    /// a Domain. Returns a stack allocated array for the coordinates of the centre,
    ///
    /// # Arguments
    /// * `domain` - The physical domain with which we calculate the centre with respect to.
    pub fn centre(&self, domain: &Domain<T>) -> [T; 3] {
        let mut result = [T::zero(); 3];

        let anchor_coordinate = self.to_coordinates(domain);
        let diameter = self.diameter(domain);
        let two = T::from(2.0).unwrap();

        for (i, (c, d)) in anchor_coordinate.iter().zip(diameter).enumerate() {
            result[i] = *c + d / two;
        }

        result
    }

    /// The anchor corresponding to this key.
    pub fn anchor(&self) -> &[u64; 3] {
        &self.anchor
    }

    /// The Morton Key in index form.
    pub fn morton(&self) -> u64 {
        self.morton
    }

    /// The level of this key.
    pub fn level(&self) -> u64 {
        self.morton & LEVEL_MASK
    }

    /// Return the parent of a Morton Key.
    pub fn parent(&self) -> Self {
        let level = self.level();
        let morton = self.morton >> LEVEL_DISPLACEMENT;

        let parent_level = level - 1;
        let bit_multiplier = DEEPEST_LEVEL - parent_level;

        // Zeros out the last 3 * bit_multiplier bits of the Morton index
        let parent_morton_without_level = (morton >> (3 * bit_multiplier)) << (3 * bit_multiplier);

        let parent_morton = (parent_morton_without_level << LEVEL_DISPLACEMENT) | parent_level;

        MortonKey::from_morton(parent_morton)
    }

    /// Return the first child of a Morton Key.
    pub fn first_child(&self) -> Self {
        Self::new(&self.anchor, 1 + self.morton)
    }

    /// Return the first child of a Morton Key on the deepest level.
    pub fn finest_first_child(&self) -> Self {
        Self::new(&self.anchor, DEEPEST_LEVEL - self.level() + self.morton)
    }

    /// Return the last child of a Morton Key on the deepest level.
    pub fn finest_last_child(&self) -> Self {
        self.last_child(DEEPEST_LEVEL)
    }

    /// Return last child at `level`
    pub fn last_child(&self, level: u64) -> Self {
        if self.level() < level {
            let mut level_diff = level - self.level();
            let mut flc = *self.children().iter().max().unwrap();
            while level_diff > 1 {
                let tmp = flc;
                flc = *tmp.children().iter().max().unwrap();
                level_diff -= 1;
            }

            flc
        } else {
            *self
        }
    }

    /// Return all children of a Morton Key in sorted order.
    pub fn children(&self) -> Vec<MortonKey<T>> {
        let level = self.level();
        let morton = self.morton() >> LEVEL_DISPLACEMENT;

        let mut children_morton: [u64; 8] = [0; 8];
        let mut children = Vec::with_capacity(8);
        let bit_shift = 3 * (DEEPEST_LEVEL - level - 1);
        for (index, item) in children_morton.iter_mut().enumerate() {
            *item = ((morton | (index << bit_shift) as u64) << LEVEL_DISPLACEMENT) | (level + 1);
        }

        for &child_morton in children_morton.iter() {
            children.push(MortonKey::from_morton(child_morton))
        }

        children.sort();
        children
    }

    /// Return all children of the parent of this Morton Key.
    pub fn siblings(&self) -> Vec<MortonKey<T>> {
        self.parent().children()
    }

    /// Check if the key is ancestor of `other`.
    pub fn is_ancestor(&self, other: &MortonKey<T>) -> bool {
        let ancestors = other.ancestors();
        ancestors.contains(self)
    }

    /// Check if key is descendant of another key.
    pub fn is_descendant(&self, other: &MortonKey<T>) -> bool {
        other.is_ancestor(self)
    }

    /// Return set of all ancestors of this Morton Key.
    pub fn ancestors(&self) -> HashSet<MortonKey<T>> {
        let mut ancestors = HashSet::<MortonKey<_>>::new();

        let mut current = *self;

        ancestors.insert(current);

        while current.level() > 0 {
            current = current.parent();
            ancestors.insert(current);
        }

        ancestors
    }

    /// Return set of all ancestors of this Morton Key up to a specified level
    pub fn ancestors_to_level(&self, level: u64) -> HashSet<MortonKey<T>> {
        let mut ancestors = HashSet::<MortonKey<_>>::new();

        let mut current = *self;

        ancestors.insert(current);

        while current.level() > level {
            current = current.parent();
            ancestors.insert(current);
        }

        ancestors
    }

    /// Return descendants `n` levels down from a key.
    ///
    /// # Arguments
    /// * `n` - The level below the key's level to return descendants from.
    pub fn descendants(&self, n: u64) -> Result<Vec<MortonKey<T>>, std::io::Error> {
        let valid: bool = self.level() + n <= DEEPEST_LEVEL;

        match valid {
            false => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Cannot find descendants below level {:?}", DEEPEST_LEVEL),
            )),
            true => {
                let mut descendants = vec![*self];
                for _ in 0..n {
                    let mut tmp = Vec::<MortonKey<_>>::new();
                    for key in descendants {
                        tmp.append(&mut key.children());
                    }
                    descendants = tmp;
                }
                Ok(descendants)
            }
        }
    }

    /// Find the finest ancestor of key and another key.
    ///
    /// # Arguments
    /// * `other` - The key with which we are calculating the shared finest ancestor with respect to.
    pub fn finest_ancestor(&self, other: &MortonKey<T>) -> MortonKey<T> {
        if self == other {
            *other
        } else {
            let my_ancestors = self.ancestors();
            let mut current = *other;
            while !my_ancestors.contains(&current) {
                current = current.parent()
            }
            current
        }
    }

    /// Return the coordinates of the anchor for this Morton Key.
    ///
    /// # Arguments
    /// * `domain` - The domain with which we are calculating with respect to.
    pub fn to_coordinates(&self, domain: &Domain<T>) -> [T; 3] {
        let mut coord: [T; 3] = [T::zero(); 3];

        for (anchor_value, coord_ref, origin_value, diameter_value) in
            izip!(self.anchor, &mut coord, &domain.origin, &domain.side_length)
        {
            *coord_ref = *origin_value
                + *diameter_value * T::from(anchor_value).unwrap() / T::from(LEVEL_SIZE).unwrap();
        }

        coord
    }

    /// Serialized representation of a box associated with a key.
    ///
    /// Returns a vector with 24 entries, associated with the 8 x,y,z coordinates
    /// of the box associated with the key.
    /// If the lower left corner of the box is (0, 0, 0). Then the points are numbered in the
    /// following order.
    /// 1. (0, 0, 0)
    /// 2. (1, 0, 0)
    /// 3. (0, 1, 0)
    /// 4. (1, 1, 0)
    /// 5. (0, 0, 1)
    /// 6. (1, 0, 1)
    /// 7. (0, 1, 1)
    /// 8. (1, 1, 1)
    ///
    /// # Arguments
    /// * `domain` - The domain with which we are calculating with respect to.
    pub fn box_coordinates(&self, domain: &Domain<T>) -> Vec<T> {
        let mut serialized = Vec::<T>::with_capacity(24);
        let level = self.level();
        let step = (1 << (DEEPEST_LEVEL - level)) as u64;

        let anchors = [
            [self.anchor[0], self.anchor[1], self.anchor[2]],
            [step + self.anchor[0], self.anchor[1], self.anchor[2]],
            [self.anchor[0], step + self.anchor[1], self.anchor[2]],
            [step + self.anchor[0], step + self.anchor[1], self.anchor[2]],
            [self.anchor[0], self.anchor[1], step + self.anchor[2]],
            [step + self.anchor[0], self.anchor[1], step + self.anchor[2]],
            [self.anchor[0], step + self.anchor[1], step + self.anchor[2]],
            [
                step + self.anchor[0],
                step + self.anchor[1],
                step + self.anchor[2],
            ],
        ];

        for anchor in anchors.iter() {
            let mut coord = [T::zero(); 3];
            for (&anchor_value, coord_ref, origin_value, diameter_value) in
                izip!(anchor, &mut coord, &domain.origin, &domain.side_length)
            {
                *coord_ref = *origin_value
                    + *diameter_value * T::from(anchor_value).unwrap()
                        / T::from(LEVEL_SIZE).unwrap();
            }

            for component in &coord {
                serialized.push(*component);
            }
        }

        serialized
    }

    /// Find key in a given direction.
    ///
    /// Returns the key obtained by moving direction\[j\] boxes into direction j
    /// starting from the anchor associated with the given key.
    /// Negative steps are possible. If the result is out of bounds,
    /// i.e. anchor\[j\] + direction\[j\] is negative or larger than the number of boxes
    /// across each dimension, `None` is returned. Otherwise, `Some(new_key)` is returned,
    /// where `new_key` is the Morton key after moving into the given direction.
    ///
    /// # Arguments
    /// * `direction` - A vector describing how many boxes we move along each coordinate direction.
    ///               Negative values are possible (meaning that we move backwards).
    pub fn find_key_in_direction(&self, direction: &[i64; 3]) -> Option<MortonKey<T>> {
        let level = self.level();

        let max_number_of_boxes: i64 = 1 << DEEPEST_LEVEL;
        let step_multiplier: i64 = (1 << (DEEPEST_LEVEL - level)) as i64;

        let x: i64 = self.anchor[0] as i64;
        let y: i64 = self.anchor[1] as i64;
        let z: i64 = self.anchor[2] as i64;

        let x = x + step_multiplier * direction[0];
        let y = y + step_multiplier * direction[1];
        let z = z + step_multiplier * direction[2];

        if (x >= 0)
            & (y >= 0)
            & (z >= 0)
            & (x < max_number_of_boxes)
            & (y < max_number_of_boxes)
            & (z < max_number_of_boxes)
        {
            let new_anchor: [u64; 3] = [x as u64, y as u64, z as u64];
            let new_morton = encode_anchor(&new_anchor, level);
            Some(Self::new(&new_anchor, new_morton))
        } else {
            None
        }
    }

    /// Find all neighbors for to a given key. Filter out 'invalid' neighbours that lie outside of the tree domain.
    pub fn neighbors(&self) -> Vec<MortonKey<T>> {
        DIRECTIONS
            .iter()
            .map(|d| self.find_key_in_direction(d))
            .filter(|d| !d.is_none())
            .map(|d| d.unwrap())
            .collect()
    }

    /// Find all neighbors for to a given key, even if they lie outside of the tree domain.
    pub fn all_neighbors(&self) -> Vec<Option<MortonKey<T>>> {
        DIRECTIONS
            .iter()
            .map(|d| self.find_key_in_direction(d))
            .collect()
    }

    /// Check if two keys are adjacent with respect to each other when they are known to on the same tree level.
    pub fn is_adjacent_same_level(&self, other: &MortonKey<T>) -> bool {
        // Calculate distance between centres of each node
        let da = 1 << (DEEPEST_LEVEL - self.level());
        let db = 1 << (DEEPEST_LEVEL - other.level());
        let ra = (da as f64) * 0.5;
        let rb = (db as f64) * 0.5;

        let ca: Vec<f64> = self.anchor.iter().map(|&x| (x as f64) + ra).collect();
        let cb: Vec<f64> = other.anchor.iter().map(|&x| (x as f64) + rb).collect();

        let distance: Vec<f64> = ca.iter().zip(cb.iter()).map(|(a, b)| b - a).collect();

        let min = -ra - rb;
        let max = ra + rb;
        let mut result = true;

        for &d in distance.iter() {
            if d > max || d < min {
                result = false
            }
        }

        result
    }

    /// Check if two keys are adjacent with respect to each other
    pub fn is_adjacent(&self, other: &MortonKey<T>) -> bool {
        let ancestors = self.ancestors();
        let other_ancestors = other.ancestors();

        // If either key overlaps they cannot be adjacent.
        if ancestors.contains(other) || other_ancestors.contains(self) {
            false
        } else {
            self.is_adjacent_same_level(other)
        }
    }
}

impl<T> TreeNode for MortonKey<T>
where
    T: RlstScalar + Float,
{
    type Scalar = T;

    type Nodes = MortonKeys<T>;
    type Domain = Domain<T>;

    fn raw(&self) -> u64 {
        self.morton
    }

    fn children(&self) -> Self::Nodes {
        MortonKeys {
            keys: self.children(),
            index: 0,
        }
    }

    fn parent(&self) -> Self {
        self.parent()
    }

    fn level(&self) -> u64 {
        self.level()
    }

    fn neighbors(&self) -> Self::Nodes {
        MortonKeys {
            keys: self.neighbors(),
            index: 0,
        }
    }

    fn is_adjacent(&self, other: &Self) -> bool {
        self.is_adjacent(other)
    }
}

/// Compute surface grid for a given expansion order used in the kernel independent fast multipole method
/// returns a tuple, the first element is an owned vector of the physical coordinates of the
/// surface grid in row major order [x_1, y_1, z_1,...x_N, y_N, z_N].
/// the second element is a vector of indices corresponding to each of these coordinates.
///
/// # Arguments
/// * `expansion_order` - the expansion order of the fmm
pub fn surface_grid<T: RlstScalar + Float>(expansion_order: usize) -> Vec<T> {
    let dim = 3;
    let n_coeffs = ncoeffs_kifmm(expansion_order);

    // Implicitly in row major order
    let mut surface: Vec<T> = vec![T::zero(); dim * n_coeffs];

    // Bounds of the surface grid
    let lower = 0;
    let upper = expansion_order - 1;

    // Orders the surface grid, implicitly row major
    let mut idx = 0;

    for k in 0..expansion_order {
        for j in 0..expansion_order {
            for i in 0..expansion_order {
                if (i >= lower && j >= lower && (k == lower || k == upper))
                    || (j >= lower && k >= lower && (i == lower || i == upper))
                    || (k >= lower && i >= lower && (j == lower || j == upper))
                {
                    surface[dim * idx] = T::from(i).unwrap();
                    surface[dim * idx + 1] = T::from(j).unwrap();
                    surface[dim * idx + 2] = T::from(k).unwrap();
                    idx += 1;
                }
            }
        }
    }

    // Shift and scale surface so that it's centered at the origin and has side length of 1
    let two = T::from(2.0).unwrap();

    surface.iter_mut().for_each(|point| {
        *point *= two / (T::from(expansion_order).unwrap() - T::one());
    });

    surface.iter_mut().for_each(|point| *point -= T::one());

    surface
}

impl<T: RlstScalar + Float> FmmTreeNode for MortonKey<T> {
    fn convolution_grid(
        &self,
        expansion_order: usize,
        domain: &Self::Domain,
        alpha: Self::Scalar,
        conv_point_corner: &[Self::Scalar],
        conv_point_corner_index: usize,
    ) -> (Vec<Self::Scalar>, Vec<usize>) {
        // Number of convolution points along each axis
        let n = 2 * expansion_order - 1;

        let dim: usize = 3;
        let ncoeffs = n.pow(dim as u32);
        let mut grid = vec![T::zero(); dim * ncoeffs];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let conv_index = i + j * n + k * n * n;
                    grid[dim * conv_index] = T::from(i).unwrap();
                    grid[dim * conv_index + 1] = T::from(j).unwrap();
                    grid[dim * conv_index + 2] = T::from(k).unwrap();
                }
            }
        }

        // Map conv points to indices
        let conv_idxs = grid
            .iter()
            .clone()
            .map(|&x| x.to_usize().unwrap())
            .collect();

        let diameter = self
            .diameter(domain)
            .iter()
            .map(|x| *x * alpha)
            .collect_vec();

        // Shift and scale to embed surface grid inside convolution grid
        // Scale
        grid.iter_mut().for_each(|point| {
            *point *= T::from(1.0).unwrap() / T::from(n - 1).unwrap(); // normalize
            *point *= diameter[0]; // find diameter
            *point *= T::from(2.0).unwrap(); // convolution grid is 2x as large
        });

        // Shift convolution grid to align with a specified corner of surface grid
        let corners = find_corners(&grid);

        let surface_point = [
            corners[dim * conv_point_corner_index],
            corners[dim * conv_point_corner_index + 1],
            corners[dim * conv_point_corner_index + 2],
        ];

        let diff = conv_point_corner
            .iter()
            .zip(surface_point)
            .map(|(a, b)| *a - b)
            .collect_vec();

        grid.chunks_exact_mut(dim).for_each(|coord| {
            coord
                .iter_mut()
                .zip(diff.iter())
                .for_each(|(c, d)| *c += *d)
        });

        (grid, conv_idxs)
    }

    fn scale_surface(
        &self,
        surface: Vec<Self::Scalar>,
        domain: &Domain<Self::Scalar>,
        alpha: Self::Scalar,
    ) -> Vec<Self::Scalar> {
        let dim = 3;
        // Translate box to specified centre, and scale
        let scaled_diameter = self.diameter(domain);
        let dilated_diameter = scaled_diameter.map(|d| d * alpha);

        let mut scaled_surface = vec![T::zero(); surface.len()];

        let centre = self.centre(domain);

        let two = T::from(2.0).unwrap();
        let ncoeffs = surface.len() / 3;
        for i in 0..ncoeffs {
            let idx = i * dim;

            scaled_surface[idx] = (surface[idx] * T::from(dilated_diameter[0] / two).unwrap())
                + T::from(centre[0]).unwrap();
            scaled_surface[idx + 1] = (surface[idx + 1]
                * T::from(dilated_diameter[1] / two).unwrap())
                + T::from(centre[1]).unwrap();
            scaled_surface[idx + 2] = (surface[idx + 2]
                * T::from(dilated_diameter[2] / two).unwrap())
                + T::from(centre[2]).unwrap();
        }

        scaled_surface
    }

    fn surface_grid(
        &self,
        expansion_order: usize,
        domain: &Domain<Self::Scalar>,
        alpha: Self::Scalar,
    ) -> Vec<Self::Scalar> {
        self.scale_surface(surface_grid(expansion_order), domain, alpha)
    }
}

#[cfg(feature = "mpi")]
use memoffset::offset_of;

#[cfg(feature = "mpi")]
use mpi::{
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    Address,
};

#[cfg(feature = "mpi")]
unsafe impl<T: RlstScalar + Float> Equivalence for MortonKey<T> {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(MortonKey<T>, anchor) as Address,
                offset_of!(MortonKey<T>, morton) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(3, &u64::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype()).as_ref(),
            ],
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tree::helpers::points_fixture;
    use rlst::{RawAccess, Shape};
    use std::vec;

    /// Subroutine in less than function, equivalent to comparing floor of log_2(x). Adapted from [3].
    fn most_significant_bit(x: u64, y: u64) -> bool {
        (x < y) & (x < (x ^ y))
    }

    /// Implementation of Algorithm 12 in [1]. to compare the ordering of two **Morton Keys**. If key
    /// `a` is less than key `b`, this function evaluates to true.
    fn less_than<T: RlstScalar + Float>(a: &MortonKey<T>, b: &MortonKey<T>) -> Option<bool> {
        // If anchors match, the one at the coarser level has the lesser Morton id.
        let same_anchor = (a.anchor[0] == b.anchor[0])
            & (a.anchor[1] == b.anchor[1])
            & (a.anchor[2] == b.anchor[2]);

        match same_anchor {
            true => {
                if a.level() < b.level() {
                    Some(true)
                } else {
                    Some(false)
                }
            }
            false => {
                let x = [
                    a.anchor[0] ^ b.anchor[0],
                    a.anchor[1] ^ b.anchor[1],
                    a.anchor[2] ^ b.anchor[2],
                ];

                let mut argmax = 0;

                for dim in 1..3 {
                    if most_significant_bit(x[argmax as usize], x[dim as usize]) {
                        argmax = dim
                    }
                }

                match argmax {
                    0 => {
                        if a.anchor[0] < b.anchor[0] {
                            Some(true)
                        } else {
                            Some(false)
                        }
                    }
                    1 => {
                        if a.anchor[1] < b.anchor[1] {
                            Some(true)
                        } else {
                            Some(false)
                        }
                    }
                    2 => {
                        if a.anchor[2] < b.anchor[2] {
                            Some(true)
                        } else {
                            Some(false)
                        }
                    }
                    _ => None,
                }
            }
        }
    }

    #[test]
    fn test_z_encode_table() {
        for (mut index, actual) in Z_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_y_encode_table() {
        for (mut index, actual) in Y_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 1)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_x_encode_table() {
        for (mut index, actual) in X_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 2)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_z_decode_table() {
        for (index, &actual) in Z_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = (index & 1) as u64;
            expected |= (((index >> 3) & 1) << 1) as u64;
            expected |= (((index >> 6) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_y_decode_table() {
        for (index, &actual) in Y_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = ((index >> 1) & 1) as u64;
            expected |= (((index >> 4) & 1) << 1) as u64;
            expected |= (((index >> 7) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_x_decode_table() {
        for (index, &actual) in X_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = ((index >> 2) & 1) as u64;
            expected |= (((index >> 5) & 1) << 1) as u64;
            expected |= (((index >> 8) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_encoding_decoding() {
        let anchor: [u64; 3] = [65535, 65535, 65535];

        let actual = decode_key(encode_anchor(&anchor, DEEPEST_LEVEL));

        assert_eq!(anchor, actual);
    }

    #[test]
    fn test_siblings() {
        // Test that we get the same siblings for a pair of siblings
        let a = [0, 0, 0];
        let b = [1, 1, 1];

        let a = MortonKey::<f64>::from_anchor(&a, DEEPEST_LEVEL);
        let b = MortonKey::<f64>::from_anchor(&b, DEEPEST_LEVEL);
        let mut sa = a.siblings();
        let mut sb = b.siblings();
        sa.sort();
        sb.sort();

        for (a, b) in sa.iter().zip(sb.iter()) {
            assert_eq!(a, b)
        }
    }

    #[test]
    fn test_sorting() {
        let n_points = 1000;
        let points = points_fixture(n_points, Some(-1.), Some(1.0), None);

        let domain = Domain::<f64>::from_local_points(points.data());

        let mut keys: Vec<MortonKey<_>> = Vec::new();

        for i in 0..points.shape()[0] {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];

            keys.push(MortonKey::from_point(&point, &domain, DEEPEST_LEVEL));
        }

        // Add duplicates to keys, to test ordering in terms of equality
        let mut cpy = keys.to_vec();
        keys.append(&mut cpy);

        // Add duplicates to ensure equality is also sorted
        let mut replica = keys.to_vec();
        keys.append(&mut replica);
        keys.sort();

        // Test that Z order is maintained when sorted
        for i in 0..(keys.len() - 1) {
            let a = keys[i];
            let b = keys[i + 1];
            assert!(less_than(&a, &b).unwrap() | (a == b));
        }
    }

    #[test]
    fn test_find_children() {
        let root = MortonKey::<f64>::root();
        let displacement = 1 << (DEEPEST_LEVEL - root.level() - 1);

        let expected: Vec<MortonKey<f64>> = vec![
            MortonKey::new(&[0, 0, 0], 1),
            MortonKey::new(
                &[displacement, 0, 0],
                0b100000000000000000000000000000000000000000000000000000000000001,
            ),
            MortonKey::new(
                &[0, displacement, 0],
                0b10000000000000000000000000000000000000000000000000000000000001,
            ),
            MortonKey::new(
                &[0, 0, displacement],
                0b1000000000000000000000000000000000000000000000000000000000001,
            ),
            MortonKey::new(
                &[displacement, displacement, 0],
                0b110000000000000000000000000000000000000000000000000000000000001,
            ),
            MortonKey::new(
                &[displacement, 0, displacement],
                0b101000000000000000000000000000000000000000000000000000000000001,
            ),
            MortonKey::new(
                &[0, displacement, displacement],
                0b11000000000000000000000000000000000000000000000000000000000001,
            ),
            MortonKey::new(
                &[displacement, displacement, displacement],
                0b111000000000000000000000000000000000000000000000000000000000001,
            ),
        ];

        let children = root.children();

        for child in &children {
            assert!(expected.contains(child));
        }
    }

    #[test]
    fn test_ancestors() {
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);
        let point = [0.5, 0.5, 0.5];

        let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);

        let mut ancestors: Vec<MortonKey<_>> = key.ancestors().into_iter().collect();
        ancestors.sort();

        // Test that all ancestors found
        for (current_level, &ancestor) in ancestors.iter().enumerate() {
            assert!(ancestor.level() == current_level.try_into().unwrap());
        }

        // Test that the ancestors include the key at the leaf level
        assert!(ancestors.contains(&key));
    }

    #[test]
    pub fn test_finest_ancestor() {
        // Trivial case
        let root = MortonKey::<f64>::root();
        let result = root.finest_ancestor(&root);
        let expected = root;

        assert!(result == expected);

        // Standard case
        let displacement = 1 << (DEEPEST_LEVEL - root.level() - 1);
        let a = MortonKey::<f64>::new(&[0, 0, 0], 16);
        let b = MortonKey::new(
            &[displacement, displacement, displacement],
            0b111000000000000000000000000000000000000000000000000000000000001,
        );
        let result = a.finest_ancestor(&b);
        let expected = MortonKey::new(&[0, 0, 0], 0);
        assert!(result == expected);
    }

    #[test]
    pub fn test_neighbors() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);
        let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);

        // Simple case, at the leaf level
        {
            let mut result = key.neighbors();
            result.sort();

            // Test that we get the expected number of neighbors
            assert!(result.len() == 26);

            // Test that the displacements are correct
            let displacement = 1 << (DEEPEST_LEVEL - key.level()) as i64;
            let anchor = key.anchor;
            let expected: [[i64; 3]; 26] = [
                [-displacement, -displacement, -displacement],
                [-displacement, -displacement, 0],
                [-displacement, -displacement, displacement],
                [-displacement, 0, -displacement],
                [-displacement, 0, 0],
                [-displacement, 0, displacement],
                [-displacement, displacement, -displacement],
                [-displacement, displacement, 0],
                [-displacement, displacement, displacement],
                [0, -displacement, -displacement],
                [0, -displacement, 0],
                [0, -displacement, displacement],
                [0, 0, -displacement],
                [0, 0, displacement],
                [0, displacement, -displacement],
                [0, displacement, 0],
                [0, displacement, displacement],
                [displacement, -displacement, -displacement],
                [displacement, -displacement, 0],
                [displacement, -displacement, displacement],
                [displacement, 0, -displacement],
                [displacement, 0, 0],
                [displacement, 0, displacement],
                [displacement, displacement, -displacement],
                [displacement, displacement, 0],
                [displacement, displacement, displacement],
            ];

            let mut expected: Vec<MortonKey<_>> = expected
                .iter()
                .map(|n| {
                    [
                        (n[0] + (anchor[0] as i64)) as u64,
                        (n[1] + (anchor[1] as i64)) as u64,
                        (n[2] + (anchor[2] as i64)) as u64,
                    ]
                })
                .map(|anchor| MortonKey::from_anchor(&anchor, DEEPEST_LEVEL))
                .collect();
            expected.sort();

            for i in 0..26 {
                assert!(expected[i] == result[i]);
            }

            // Test that they are in Morton order
            for i in 0..25 {
                assert!(expected[i + 1] >= expected[i])
            }
        }

        // More complex case, in the middle of the tree
        {
            let parent = key.parent().parent().parent();
            let result = parent.neighbors();
            // result.sort();

            // Test that we get the expected number of neighbors
            assert!(result.len() == 26);

            // Test that the displacements are correct
            let displacement = 1 << (DEEPEST_LEVEL - parent.level()) as i64;
            let anchor = key.anchor;
            let expected: [[i64; 3]; 26] = [
                [-displacement, -displacement, -displacement],
                [-displacement, -displacement, 0],
                [-displacement, -displacement, displacement],
                [-displacement, 0, -displacement],
                [-displacement, 0, 0],
                [-displacement, 0, displacement],
                [-displacement, displacement, -displacement],
                [-displacement, displacement, 0],
                [-displacement, displacement, displacement],
                [0, -displacement, -displacement],
                [0, -displacement, 0],
                [0, -displacement, displacement],
                [0, 0, -displacement],
                [0, 0, displacement],
                [0, displacement, -displacement],
                [0, displacement, 0],
                [0, displacement, displacement],
                [displacement, -displacement, -displacement],
                [displacement, -displacement, 0],
                [displacement, -displacement, displacement],
                [displacement, 0, -displacement],
                [displacement, 0, 0],
                [displacement, 0, displacement],
                [displacement, displacement, -displacement],
                [displacement, displacement, 0],
                [displacement, displacement, displacement],
            ];

            let mut expected: Vec<MortonKey<_>> = expected
                .iter()
                .map(|n| {
                    [
                        (n[0] + (anchor[0] as i64)) as u64,
                        (n[1] + (anchor[1] as i64)) as u64,
                        (n[2] + (anchor[2] as i64)) as u64,
                    ]
                })
                .map(|anchor| MortonKey::<f64>::from_anchor(&anchor, DEEPEST_LEVEL))
                .map(|key| {
                    let anchor = key.anchor;
                    let morton =
                        ((key.morton >> LEVEL_DISPLACEMENT) << LEVEL_DISPLACEMENT) | parent.level();
                    MortonKey::new(&anchor, morton)
                })
                .collect();
            expected.sort();

            for i in 0..26 {
                assert!(expected[i] == result[i]);
            }

            // Test that they are in Morton order
            for i in 0..25 {
                assert!(result[i + 1] >= result[i])
            }
        }
    }

    #[test]
    pub fn test_morton_keys_iterator() {
        let n_points = 1000;
        let domain = Domain::<f64>::new(&[-1.01, -1.01, -1.01], &[2.0, 2.0, 2.0]);
        let min = Some(-1.01);
        let max = Some(0.99);

        let points = points_fixture(n_points, min, max, None);

        let mut keys = Vec::new();

        for i in 0..points.shape()[0] {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            keys.push(MortonKey::from_point(&point, &domain, DEEPEST_LEVEL))
        }

        let keys = MortonKeys { keys, index: 0 };

        // test that we can call keys as an iterator
        keys.iter().sorted();

        // test that iterator index resets to 0
        assert!(keys.index == 0);
    }

    #[test]
    fn test_linearize_keys() {
        let key = MortonKey::<f64>::new(&[0, 0, 0], 15);
        let ancestors = key.ancestors().into_iter().collect_vec();
        let linearized = MortonKeys::linearize_keys(&ancestors);

        assert_eq!(linearized.len(), 1);
        assert_eq!(linearized[0], key);
    }

    #[test]
    fn test_point_to_anchor() {
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);

        // Test points in the domain
        let point = [0.9999, 0.9999, 0.9999];
        let level = 2;
        let anchor = point_to_anchor(&point, level, &domain);
        let expected = [49152, 49152, 49152];

        for (i, a) in anchor.unwrap().iter().enumerate() {
            assert_eq!(a, &expected[i])
        }

        let domain = Domain::<f64>::new(&[-0.7, -0.6, -0.5], &[1., 1., 1.]);

        let point = [-0.499, -0.499, -0.499];
        let level = 1;
        let anchor = point_to_anchor(&point, level, &domain);
        let expected = [0, 0, 0];

        for (i, a) in anchor.unwrap().iter().enumerate() {
            assert_eq!(a, &expected[i])
        }
    }

    #[test]
    fn test_point_to_anchor_fails() {
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);

        // Test a point not in the domain
        let point = [1.9, 0.9, 0.9];
        let level = 2;
        assert!(point_to_anchor(&point, level, &domain).is_err());
    }

    #[test]
    fn test_point_to_anchor_fails_negative_domain() {
        let domain = Domain::<f64>::new(&[-0.5, -0.5, -0.5], &[1., 1., 1.]);

        // Test a point not in the domain
        let point = [-0.51, -0.5, -0.5];
        let level = 2;
        assert!(point_to_anchor(&point, level, &domain).is_err());
    }

    #[test]
    fn test_encode_anchor() {
        let anchor = [1, 0, 1];
        let level = 1;
        let morton = encode_anchor(&anchor, level);
        let expected = 0b101000000000000001;
        assert_eq!(expected, morton);

        let anchor = [3, 3, 3];
        let level = 2;
        let morton = encode_anchor(&anchor, level);
        let expected = 0b111111000000000000010;
        assert_eq!(expected, morton);
    }

    #[test]
    fn test_find_descendants() {
        let root = MortonKey::<f64>::root();

        let descendants = root.descendants(1).unwrap();
        assert_eq!(descendants.len(), 8);

        // Ensure this also works for other keys in hierarchy
        let key = descendants[0];
        let descendants = key.descendants(2).unwrap();
        assert_eq!(descendants.len(), 64);
    }

    #[test]
    fn test_find_descendants_errors() {
        let root = MortonKey::<f64>::root();
        assert!(root.descendants(17).is_err());
    }

    #[test]
    fn test_complete_region() {
        let a = MortonKey::<f64>::new(&[0, 0, 0], 16);
        let b = MortonKey::<f64>::new(
            &[LEVEL_SIZE - 1, LEVEL_SIZE - 1, LEVEL_SIZE - 1],
            0b111111111111111111111111111111111111111111111111000000000010000,
        );

        let region = MortonKeys::complete_region(&a, &b);

        let fa = a.finest_ancestor(&b);

        let min = *region.iter().min().unwrap();
        let max = *region.iter().max().unwrap();

        // Test that bounds are satisfied
        assert!(a <= min);
        assert!(b >= max);

        // Test that FCA is an ancestor of all nodes in the result
        for node in region.iter() {
            let ancestors = node.ancestors();
            assert!(ancestors.contains(&fa));
        }

        // Test that completed region doesn't contain its bounds
        assert!(!region.contains(&a));
        assert!(!region.contains(&b));

        // Test that the compeleted region doesn't contain any overlaps
        for node in region.iter() {
            let mut ancestors = node.ancestors();
            ancestors.remove(node);
            for ancestor in ancestors.iter() {
                assert!(!region.contains(ancestor))
            }
        }

        // Test that the region is sorted
        for i in 0..region.iter().len() - 1 {
            let a = region[i];
            let b = region[i + 1];

            assert!(a <= b);
        }
    }

    #[test]
    pub fn test_balance() {
        let a = MortonKey::<f64>::from_anchor(&[0, 0, 0], DEEPEST_LEVEL);
        let b = MortonKey::<f64>::from_anchor(&[1, 1, 1], DEEPEST_LEVEL);

        let mut complete = MortonKeys::complete_region(&a, &b);
        let start_val = vec![a];
        let end_val = vec![b];
        complete = start_val
            .into_iter()
            .chain(complete)
            .chain(end_val)
            .collect();
        let mut tree = MortonKeys {
            keys: complete,
            index: 0,
        };

        tree.balance();
        tree.linearize();
        tree.sort();

        // Test for overlaps in balanced tree
        for key in tree.iter() {
            if !tree.iter().contains(key) {
                let mut ancestors = key.ancestors();
                ancestors.remove(key);

                for ancestor in ancestors.iter() {
                    assert!(!tree.keys.contains(ancestor));
                }
            }
        }

        // Test that adjacent keys are 2:1 balanced
        for key in tree.iter() {
            let adjacent_levels: Vec<u64> = tree
                .iter()
                .cloned()
                .filter(|k| key.is_adjacent(k))
                .map(|a| a.level())
                .collect();

            for l in adjacent_levels.iter() {
                assert!(l.abs_diff(key.level()) <= 1);
            }
        }
    }

    #[test]
    fn test_is_adjacent() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain::<f64>::new(&[-0.1, -0.1, -0.1], &[1., 1., 1.]);

        let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);

        let mut ancestors = key.ancestors();
        ancestors.remove(&key);

        // Test that overlapping nodes are not adjacent
        for a in ancestors.iter() {
            assert!(!key.is_adjacent(a))
        }

        // Test that siblings & neighbours are adjacent
        let siblings = key.siblings();
        let neighbors = key.neighbors();

        for s in siblings.iter() {
            if *s != key {
                assert!(key.is_adjacent(s));
            }
        }

        for n in neighbors.iter() {
            assert!(key.is_adjacent(n));
        }

        // Test keys on different levels
        let anchor_a = [0, 0, 0];
        let a = MortonKey::<f64>::from_anchor(&anchor_a, DEEPEST_LEVEL - 1);
        let anchor_b = [2, 2, 2];
        let b = MortonKey::from_anchor(&anchor_b, DEEPEST_LEVEL);
        assert!(a.is_adjacent(&b));
    }

    #[test]
    fn test_encoding_is_always_absolute() {
        let point = [-0.099999, -0.099999, -0.099999];
        let domain = Domain::<f64>::new(&[-0.1, -0.1, -0.1], &[1., 1., 1.]);

        let a = MortonKey::from_point(&point, &domain, 1);
        let b = MortonKey::from_point(&point, &domain, 16);
        assert_ne!(a, b);
        assert_eq!(a.anchor, b.anchor);
    }

    #[test]
    fn test_transfer_vector() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);

        // Test scale independence of transfer vectors
        let a = MortonKey::from_point(&point, &domain, 2);
        let other = a.siblings()[2];
        let res_a = a.find_transfer_vector(&other).unwrap();

        let b = MortonKey::from_point(&point, &domain, 16);
        let other = b.siblings()[2];
        let res_b = b.find_transfer_vector(&other).unwrap();

        assert_eq!(res_a, res_b);

        // Test translational invariance of transfer vector
        let a = MortonKey::from_point(&point, &domain, 2);
        let other = a.siblings()[2];
        let res_a = a.find_transfer_vector(&other).unwrap();

        let shifted_point = [0.1, 0.1, 0.1];
        let b = MortonKey::from_point(&shifted_point, &domain, 2);
        let other = b.siblings()[2];
        let res_b = b.find_transfer_vector(&other).unwrap();

        assert_eq!(res_a, res_b);
    }

    #[test]
    fn test_transfer_vector_errors() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);
        let key = MortonKey::from_point(&point, &domain, 1);
        let sibling = key.siblings()[0];
        assert!(key.find_transfer_vector(&sibling).is_err());
    }

    #[test]
    fn test_surface_grid() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);
        let key = MortonKey::from_point(&point, &domain, 0);

        let expansion_order = 2;
        let alpha = 1.;
        let dim = 3;
        let ncoeffs = ncoeffs_kifmm(expansion_order);

        // Test lengths
        let surface = key.surface_grid(expansion_order, &domain, alpha);
        assert_eq!(surface.len(), ncoeffs * dim);

        let surface: Vec<f64> = surface_grid(expansion_order);
        assert_eq!(surface.len(), ncoeffs * dim);

        let mut expected = vec![[0usize; 3]; ncoeffs];
        let lower = 0;
        let upper = expansion_order - 1;
        let mut idx = 0;
        for k in 0..expansion_order {
            for j in 0..expansion_order {
                for i in 0..expansion_order {
                    if (i >= lower && j >= lower && (k == lower || k == upper))
                        || (j >= lower && k >= lower && (i == lower || i == upper))
                        || (k >= lower && i >= lower && (j == lower || j == upper))
                    {
                        expected[idx] = [i, j, k];
                        idx += 1;
                    }
                }
            }
        }

        // Test scaling
        let level = 2;
        let key = MortonKey::from_point(&point, &domain, level);
        let surface = key.surface_grid(expansion_order, &domain, alpha);

        let min_x = surface
            .iter()
            .take(ncoeffs)
            .fold(f64::INFINITY, |a, &b| a.min(b));

        let max_x = surface.iter().take(ncoeffs).fold(0f64, |a, &b| a.max(b));

        let diam_x = max_x - min_x;

        let expected = key.diameter(&domain)[0];
        assert_eq!(diam_x, expected);

        // Test shifting
        let point = [0.1, 0.2, 0.3];
        let level = 2;
        let key = MortonKey::from_point(&point, &domain, level);
        let surface = key.surface_grid(expansion_order, &domain, alpha);
        let expected = key.centre(&domain);

        let xs = surface.iter().step_by(3).cloned().collect_vec();
        let ys = surface.iter().skip(1).step_by(3).cloned().collect_vec();
        let zs = surface.iter().skip(2).step_by(3).cloned().collect_vec();
        let c_x = xs.iter().take(ncoeffs).fold(0f64, |a, b| a + b) / (ncoeffs as f64);
        let c_y = ys.iter().take(ncoeffs).fold(0f64, |a, b| a + b) / (ncoeffs as f64);
        let c_z = zs.iter().take(ncoeffs).fold(0f64, |a, b| a + b) / (ncoeffs as f64);

        let result = vec![c_x, c_y, c_z];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_convolution_grid() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            side_length: [1., 1., 1.],
        };

        let expansion_order = 5;
        let alpha = 1.0;
        let dim = 3;

        let key = MortonKey::from_point(&point, &domain, 0);

        let surface_grid = key.surface_grid(expansion_order, &domain, alpha);

        // Place convolution grid on max corner
        let corners = find_corners(&surface_grid);
        let conv_point_corner_index = 7;

        let conv_point = vec![
            corners[dim * conv_point_corner_index],
            corners[dim * conv_point_corner_index + 1],
            corners[dim * conv_point_corner_index + 2],
        ];

        let (conv_grid, _) = key.convolution_grid(
            expansion_order,
            &domain,
            alpha,
            &conv_point,
            conv_point_corner_index,
        );

        // Test that surface grid is embedded in convolution grid
        let mut surface = Vec::new();
        let nsurf = surface_grid.len() / 3;
        for i in 0..nsurf {
            let idx = i * 3;
            surface.push([
                surface_grid[idx],
                surface_grid[idx + 1],
                surface_grid[idx + 2],
            ])
        }

        let mut convolution = Vec::new();
        let nconv = conv_grid.len() / 3;
        for i in 0..nconv {
            let idx = i * 3;
            convolution.push([conv_grid[idx], conv_grid[idx + 1], conv_grid[idx + 2]])
        }
        assert!(surface.iter().all(|point| convolution.contains(point)));
    }
}
