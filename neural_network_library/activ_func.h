#ifndef ACTIV_FUNCH
#define ACTIV_FUNCH
#include<iostream>
#include<Eigen/Dense>
#include<memory>
#include "neuronbase.h"
#include<autodiff/reverse/var.hpp>
#include<autodiff/reverse/var/eigen.hpp>
using namespace autodiff;
using namespace std;
using namespace Eigen;

/*class ReLU_new
{
private:
    int m_current_neuron_num;

public:
    ReLU_new(int current_neuron_num)
        :m_current_neuron_num(current_neuron_num)
    {}

    VectorXvar h(VectorXvar x)
    {
        VectorXvar vector_after_activ(m_current_neuron_num);
        for(int i=0;i<m_current_neuron_num;i++)                         //need modifications here
        {
             vector_after_activ(i) = max(0,x(i));
        }
        return vector_after_activ;
    }

};*/
class Tanh
{
private:
    int m_current_neuron_num;
public:
    Tanh(int current_neuron_num)
        :m_current_neuron_num(current_neuron_num)
    {}
    VectorXvar h(VectorXvar x)
    {
        return (x.array()).tanh();
    }
};
class linear
{
private:
    int m_current_neuron_num;
public:
    linear(int current_neuron_num)
        :m_current_neuron_num(current_neuron_num)
    {}
    VectorXvar h(VectorXvar x)
    {
        return x;
    }
};
class Softmax
{
private:
    int m_current_neuron_num;
public:
    Softmax(int current_neuron_num)
        :m_current_neuron_num(current_neuron_num)
    {}
    VectorXvar h(VectorXvar x)
    {
        return exp((x.array()-x.maxCoeff()))/(exp((x.array()-x.maxCoeff())).sum());
    }
};
#endif
