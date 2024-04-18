use green_kernels::helmholtz_3d::Helmholtz3dKernel;
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::{BlasFieldTranslation, FftFieldTranslation};
use kifmm::SingleNodeBuilder;
use kifmm::traits::fmm::Fmm;
use kifmm::tree::helpers::points_fixture;
use num::One;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rlst::{rlst_dynamic_array2, Array, BaseArray, RawAccessMut, RlstScalar, Shape, VectorContainer, c64};

extern crate blas_src;
extern crate lapack_src;

use std::error::Error;
use std::fs::File;
use std::path::Path;
use csv::Reader;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct Coordinate {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug, Serialize)]
struct Potential <T>
where
    T: RlstScalar + Serialize + for<'de> Deserialize<'de>, // Ensuring T also supports serialization
{
    value: T
}

impl <T> Potential<T>
where
    T: RlstScalar + Serialize + for<'de> Deserialize<'de>, // Ensuring T also supports serialization
{
    fn from(value: T) -> Self {
        Self {
            value
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct GlobalIndex {
    value: usize
}

fn read_csv_into_vectors(filepath: &str) -> Result<Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>, Box<dyn Error>> {

    let file = File::open(filepath)?;
    let mut rdr = Reader::from_reader(file);

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();

    for result in rdr.deserialize() {
        let record: Coordinate = result?;
        xs.push(record.x);
        ys.push(record.y);
        zs.push(record.z);
    }
    let mut tmp = Vec::new();

    let n = xs.len();
    tmp.append(&mut xs);
    tmp.append(&mut ys);
    tmp.append(&mut zs);

    let mut result = rlst_dynamic_array2!(f64, [n, 3]);
    result.data_mut().copy_from_slice(tmp.as_slice());

    Ok(result)
}


fn main () {

    let name = "ball";
    let input_filepath =  format!("{name}.csv");
    let output_filepath = format!("output_{name}.csv");
    let global_index_file = format!("global_indices_{name}.csv");

    let sources = read_csv_into_vectors(&input_filepath).unwrap();
    let targets = read_csv_into_vectors(&input_filepath).unwrap();
    let [nsources, _] = sources.shape();

    let n_crit = Some(200);
    let expansion_order = 9;
    let sparse = true;

    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    // let tmp = vec![c64::one(); nsources * nvecs];

    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    // let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let kernel = Laplace3dKernel::new();

    let fmm = SingleNodeBuilder::new()
        .tree(&sources, &targets, n_crit, sparse)
        .unwrap()
        .parameters(
            &charges,
            expansion_order,
            kernel,
            EvalType::Value,
            FftFieldTranslation::new(),
        )
        .unwrap()
        .build()
        .unwrap();

    let s = std::time::Instant::now();
    fmm.evaluate();
    println!("{:?} npoints {:?} time {:?}", fmm.tree().source_tree.depth, sources.shape(), s.elapsed());

    // Open a file in write mode
    let file = File::create(output_filepath).unwrap();
    let mut wtr = csv::Writer::from_writer(file);

    for val in fmm.potentials {
        wtr.serialize(Potential::from(val.abs())).unwrap();
    }

    wtr.flush().unwrap();

    // Create a CSV writer from the file
    let file = File::create(global_index_file).unwrap();
    let mut wtr = csv::Writer::from_writer(file);

    for (_i, &global_idx) in fmm.tree.target_tree.global_indices.iter().enumerate() {
        wtr.serialize(GlobalIndex { value: global_idx }).unwrap();
    }

    // Ensure all data is flushed to the file
    wtr.flush().unwrap();

}