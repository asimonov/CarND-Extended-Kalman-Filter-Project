#include <iostream>
#include "tools.h"

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() == 0 || ground_truth.size() == 0 || estimations.size() != ground_truth.size())
    {
        cout << "CalculateRMSE() - wrong size of input vectors" << endl;
    }

    //accumulate squared residuals
    VectorXd s(4);
    for(int i=0; i < estimations.size(); ++i){
        VectorXd a = estimations[i]-ground_truth[i];
        VectorXd b = a.array()*a.array();
        s += b;
    }

    //calculate the mean
    s /= float(estimations.size());

    //calculate the squared root
    rmse = s.array().sqrt();

    //return the result
    return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //check division by zero
    float sumsq = px*px + py*py;
    if (sumsq <= 1e-6)
    {
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    }
    //compute the Jacobian matrix
    else
    {
        float sqsumsq = sqrt(sumsq);
        Hj << px/sqsumsq,                       py/sqsumsq,                       0.,         0.,
              -py/sumsq,                        px/sumsq,                         0.,         0.,
              py*(vx*py-vy*px)/(sumsq*sqsumsq), px*(vy*px-vx*py)/(sumsq*sqsumsq), px/sqsumsq, py/sqsumsq;
    }

    return Hj;
}
