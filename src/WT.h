#ifndef WT_H
#define WT_H
#include<iostream>
#include<Eigen/Dense>
#include<complex>
#include<math.h>
using namespace std;
using namespace Eigen;

#define _USE_MATH_DEFINES
VectorXd func(VectorXd t)
{
    return cos(2*M_PI*3*(t.array()))*exp(-M_PI*(t.array())*(t.array()));
}

class wavelet{
private:
    //VectorXd _psi;
    double scale_factor;
    MatrixXd spect;
    VectorXd f_x;
    VectorXd x;
    double Cg;
    int n_samples;
public:
    wavelet(VectorXd input_x,VectorXd input_f_x,int samples)
        :spect(8,40),
         x(input_x),
         f_x(input_f_x),
         Cg(0),
         n_samples(samples)
    {}
public:
    VectorXd psi(VectorXd t)
    {
        VectorXd identity_vector = VectorXd::Ones(t.size());

        return (identity_vector.array()-(t.array())*(t.array()))*exp(-t.array()*t.array()/2);
    }
    VectorXd f_psi(double a, double b)
    {
        return psi((x.array()-b)/a);
    }
    MatrixXd wavelet_transform()
    {  
        for(int i=0;i<8;i++)
            for(int j=0;j<40;j++)
            {
                auto s = (double(i)/10)+0.2;
                auto tau = (double(j)/40*n_samples);
                auto _psi = f_psi(s,tau);
                scale_factor = 1/sqrt(s);
                auto T = ((f_x.array())*(_psi.array())*scale_factor).sum();
                Cg+=T*T;
                spect(i,j) = T;
            }
        return spect;
    }
    VectorXd inverse_transform()
    {
        VectorXd sum=VectorXd::Zero(n_samples);
        for(int i=0;i<8;i++)
            for(int j=0;j<40;j++)
            {
                auto s = (double(i)/10)+0.2;
                auto tau = (double(j)/40*n_samples);
                auto _psi = f_psi(s,tau);
                sum+=(_psi*spect(i,j)/(s*s))/Cg;
            }
        return sum;
    }
    double get_Cg()
    {
        return Cg;
    }

};
#endif

