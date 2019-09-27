#include "kalman_filter.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;


// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

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
       x_ = F_ * x_ ;
       MatrixXd Ft = F_.transpose();
       P_ = F_ * P_ * Ft + Q_;
       //std::cout<<"Completed Prediction step"<<std::endl;
 }
void KalmanFilter::Update(const VectorXd &z) {
  	VectorXd y = z - H_ * x_;
	MatrixXd Ht = H_.transpose();
        MatrixXd s = H_ * P_ * Ht + R_;
	MatrixXd si = s.inverse();
        MatrixXd k = P_ * Ht * si;
        x_ = x_ + k*y;
        P_ = (MatrixXd::Identity(4,4) - k*H_)*P_;
 
}
void KalmanFilter::UpdateEKF(const VectorXd &z) {
        //cout<< "In radar update"<<endl;
        double px = x_[0];
        double py = x_[1];
        double vx = x_[2];
        double vy = x_[3];
        double ro = sqrt(px*px+py*py);
	double theta = atan2(py,px);
	double ro_dot = (px*vx + py*vy) / sqrt(px*px+py*py);
	VectorXd hx = VectorXd(3);
	hx << ro , theta, ro_dot;
	VectorXd y = z - hx;
	while(y(1) < -M_PI || y(1) > M_PI){
		if(y(1) < -M_PI){
			y(1)+= M_PI;		
		}else {
			y(1)-= M_PI;		
		}
	}
	MatrixXd Ht = H_.transpose();
        MatrixXd s = H_ * P_ * Ht + R_;
	MatrixXd si = s.inverse();
        MatrixXd k = P_ * Ht * si;
	x_ = x_ + k*y;
        P_ = (MatrixXd::Identity(4,4) - k*H_)*P_;
}
