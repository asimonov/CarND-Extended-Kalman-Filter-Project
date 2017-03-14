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

  // initializing matrices

  // laser measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1., 0., 0., 0.,
              0., 1., 0., 0.;
  // laser measurement covariance
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0, // from lectures
              0,      0.0225;

  // radar measurement function has to use Jacobian, depends on the state, calculated later
  // radar measurement covariance
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.0225,  0.,     0., //from lectures
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
      * Assume zero velocity to start with. Take first measurement x,y as position at the start.
    */
    ekf_.x_ = VectorXd(4);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      float ro = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      ekf_.x_ << ro * cos(phi), ro * sin(phi), 0.0, 0.0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1), 0.0, 0.0;
    }

    /**
      * Create the covariance matrix.
    */
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1000., 0.,    0.,    0., // we do not know where pedestrian is
               0.,    1000., 0.,    0.,
               0.,    0.,    0.001,    0., // but we assume pedestrian does not move. for now
               0.,    0.,    0.,    0.001;

    previous_timestamp_ = measurement_pack.timestamp_;

    ekf_.F_ = MatrixXd(4, 4); // to be initialized at each update
    ekf_.Q_ = MatrixXd(4, 4);

    is_initialized_ = true;

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;

    // done initializing, no need to predict or update
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;     //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  cout << "dt = " << dt << endl;

  if (dt > 0.00001) {
    // only predict if processing non-simultaneous measurements
    /**
       * Update the state transition matrix F according to the new elapsed time.
     */
    ekf_.F_ << 1., 0., dt, 0.,
            0., 1., 0., dt,
            0., 0., 1., 0.,
            0., 0., 0., 1.;
    /**
       * Update the process noise covariance matrix.
     */
    float dtsq = dt * dt;
    ekf_.Q_ << (dtsq / 4.) * noise_ax_, 0., (dt / 2.) * noise_ax_, 0.,
            0., (dtsq / 4.) * noise_ay_, 0., (dt / 2.) * noise_ay_,
            (dt / 2.) * noise_ax_, 0., noise_ax_, 0.,
            0., (dt / 2.) * noise_ay_, 0., noise_ay_;
    ekf_.Q_ *= dtsq;

    ekf_.Predict(); // Kalman Filter prediction step

    cout << "predict done" << endl;
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
  }

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar update

    //set measurement covariance
    ekf_.R_ = R_radar_;
    //set measurement matrix to be Jacobian of non-linear h(x)
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);

    // Extended Kalman Filter measurement update
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    cout << "radar update done" << endl;

  } else {
    // Laser update

    //set measurement covariance
    ekf_.R_ = R_laser_;
    //set measurement matrix
    ekf_.H_ = H_laser_;

    // Kalman Filter measurement update
    ekf_.Update(measurement_pack.raw_measurements_);
    cout << "laser update done" << endl;
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
