use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::field::types::{BlasFieldTranslationKiFmm, FftFieldTranslationKiFmm};
use kifmm::fmm::types::KiFmmBuilderSingleNode;
use kifmm::traits::fmm::Fmm;
use kifmm::tree::implementations::helpers::points_fixture;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rlst::{rlst_dynamic_array2, Array, BaseArray, RawAccessMut, Shape, VectorContainer};

extern crate blas_src;
extern crate lapack_src;

use std::error::Error;
use std::fs::File;
use csv::Reader;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct Record {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct Potential {
    value: f64
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
        let record: Record = result?;
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

    let input_filepath = "king.csv";
    let output_filepath = "output_king.csv";
    let global_index_file = "global_indices_king.csv";


    let sources = read_csv_into_vectors(input_filepath).unwrap();
    let targets = read_csv_into_vectors(input_filepath).unwrap();
    let [nsources, _] = sources.shape();


    let n_crit = Some(50);
    let expansion_order = 7;
    let sparse = true;

    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];

    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    // let mut rng = StdRng::seed_from_u64(0);
    // charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_fft = KiFmmBuilderSingleNode::new()
        .tree(&sources, &targets, n_crit, sparse)
        .unwrap()
        .parameters(
            &charges,
            expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            FftFieldTranslationKiFmm::new(),
        )
        .unwrap()
        .build()
        .unwrap();
    fmm_fft.evaluate();

    // Open a file in write mode
    let file = File::create(output_filepath).unwrap();
    let mut wtr = csv::Writer::from_writer(file);

    for val in fmm_fft.potentials {
        // let &val = &fmm_fft.potentials[global_idx];
        wtr.serialize(Potential { value: val }).unwrap();
    }

    wtr.flush().unwrap();

    // Create a CSV writer from the file
    let file = File::create(global_index_file).unwrap();
    let mut wtr = csv::Writer::from_writer(file);

    for (i, &global_idx) in fmm_fft.tree.target_tree.global_indices.iter().enumerate() {
        wtr.serialize(GlobalIndex { value: global_idx }).unwrap();
    }

    wtr.flush().unwrap();

    // Ensure all data is flushed to the file
}