use std::vec::Vec;
use std::error::Error;

use rusty_machine;
use rusty_machine::learning::toolkit::cost_fn::CostFunc;
use rusty_machine::learning::toolkit::cost_fn::MeanSqError;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;

use ml_utils::datasets::get_boston_records_from_file;
use ml_utils::sup_metrics::r_squared_score;
use ml_utils::datasets::BostonHousing;


pub fn run() -> Result<(), Box<dyn Error>> {
    let fl = "data/housing.csv";
    let data = get_boston_records_from_file(&fl);

    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size: usize = test_size.round() as usize;
    let (test_data, train_data): (&[BostonHousing], &[BostonHousing]) = data.split_at(test_size);
    let train_size: usize = train_data.len();
    let test_size: usize = test_data.len();

    let boston_y_train: Vec<f64> = train_data.iter().map(|r| r.into_targets()).collect();
    let boston_x_train: Vec<f64> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    
    let boston_x_test: Vec<f64> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_test: Vec<f64> = test_data.iter().map(|r| r.into_targets()).collect();

    let boston_x_train: Matrix<f64> = Matrix::new(train_size, 13, boston_x_train);
    let boston_y_train: Vector<f64> = Vector::new(boston_y_train);
    let boston_x_test: Matrix<f64> = Matrix::new(test_size, 13, boston_x_test);
    let boston_y_test: Matrix<f64> = Matrix::new(test_size, 1, boston_y_test);

    let mut lin_model: LinRegressor = LinRegressor::default();
    lin_model.train(&boston_x_train, &boston_y_train)?;

    let predictions: Vector<f64> = lin_model.predict(&boston_x_test).unwrap();
    let predictions: Matrix<f64> = Matrix::new(test_size, 1, predictions);
    let acc: f64 = {
        let outputs: &Matrix<f64> = &predictions;
        let targets: &Matrix<f64> = &boston_y_test;
        // MeanSqError divides the actual mean squared error by two.
        -2f64 * MeanSqError::cost(outputs, targets)
    };

    println!("linear regression error: {:?}", acc);
    println!("linear regression R2 score: {:?}", r_squared_score(
    &boston_y_test.data(), &predictions.data()));
    
    Ok(())

}
