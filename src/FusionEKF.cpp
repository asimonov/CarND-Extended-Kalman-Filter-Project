#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // ekf_ is already created. need first measurement to have everything else to call ekf.Init

  // initializing matrices

  // laser measurement function
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  // laser measurement covariance
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  // radar measurement function
  //Hj_ = MatrixXd(3, 4); // Jacobian, depends on the state, calculated later
  // radar measurement covariance
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.0225,  0.,     0.,
              0.,      0.0225, 0.,
              0.,      0.,     0.0225;

}

FusionEKF::~FusionEKF() {}


void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;

    /**
      * Initialize the state ekf_.x_ with the first measurement.
    */
    ekf_.x_ = VectorXd(4);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      ekf_.x_ << ro * cos(phi), ro * sin(phi), 0.0, 0.0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0.0, 0.0;
    }

    /**
      * Create the covariance matrix.
    */
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1000., 0.,    0.,    0.,
               0.,    1000., 0.,    0.,
               0.,    0.,    1000., 0.,
               0.,    0.,    0.,    1000.;

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;     //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  cout << "dt = " << dt << endl;

  /**
     * Update the state transition matrix F according to the new elapsed time.
   */
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1., 0., dt, 0.,
             0., 1., 0., dt,
             0., 0., 1., 0.,
             0., 0., 0., 1.;
  /**
     * Update the process noise covariance matrix.
   */
  float dtsq = dt*dt;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << (dtsq/4.)*noise_ax_, 0,                   (dt/2.)*noise_ax_, 0,
          0,                   (dtsq/4.)*noise_ay_, 0,                 (dt/2.)*noise_ay_,
          (dt/2.)*noise_ax_,   0,                   noise_ax_,         0,
          0,                   (dt/2.)*noise_ay_,   0,                 noise_ay_;
  ekf_.Q_ *= dtsq;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates

    //measurement covariance
    ekf_.R_ = R_radar_;
    //measurement matrix
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);

  } else {
    // Laser updates

    //measurement covariance
    ekf_.R_ = R_laser_;
    //measurement matrix
    ekf_.H_ = H_laser_;

  }

  ekf_.Update(measurement_pack.raw_measurements_);

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
