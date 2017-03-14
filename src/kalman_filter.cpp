#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state. standard KF equations. assume U==0
  */
  // KF Prediction step
  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  // KF Measurement update step
  VectorXd y = z - H_*x_;
  MatrixXd S = H_*P_*H_.transpose() + R_;
  MatrixXd K = P_*H_.transpose()*S.inverse();
  // new state
  x_ = x_ + K*y;
  MatrixXd I = MatrixXd::Identity(P_.diagonalSize(), P_.diagonalSize());
  P_ = (I - K*H_)*P_;
}


void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations, but non-linear radar measurement h(x)
  */
  float rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  float phi = 0.0;
  float rhorate = 0.0;
  if (std::abs(x_(0))>1e-6 && std::abs(x_(1))>1e-6) {
    phi = atan2(x_(1), x_(0));
    rhorate = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
  }
  VectorXd hx(3);
  hx << rho, phi, rhorate;

  // KF Measurement update step
  VectorXd y = z - hx;
  if (y(1) > M_PI)
    y(1) = y(1) - M_PI;
  if (y(1) < -M_PI)
    y(1) = y(1) + M_PI;
  // rest is same as Update() function
  MatrixXd S = H_*P_*H_.transpose() + R_;
  MatrixXd K = P_*H_.transpose()*S.inverse();
  // new state
  x_ = x_ + K*y;
  MatrixXd I = MatrixXd::Identity(P_.diagonalSize(), P_.diagonalSize());
  P_ = (I - K*H_)*P_;
}
