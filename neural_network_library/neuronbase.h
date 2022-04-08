#ifndef NEURONBASE_H
#define NEURONBASE_H

#include<iostream>
#include<Eigen/Dense>
#include<memory>
#include<autodiff/reverse/var.hpp>
#include<autodiff/reverse/var/eigen.hpp>
using namespace autodiff;
using namespace std;
using namespace Eigen;

class NeuronBase{
private:
    int m_num_neurons;
    bool m_activation_layer_flag;
    //VectorXvar current_layer_vector;
public:
    NeuronBase(int num_neurons)
        :m_num_neurons(num_neurons)//,
         //current_layer_vector(num_neurons)
        {}
public:
    virtual var activation(VectorXvar,bool)=0;
    int get_num_neurons(){return m_num_neurons;}
    //VectorXvar get_current_layer_vector(){return current_layer_vector;}
    //void set_current_layer_vector(VectorXvar x){current_layer_vector = x;}


};

#endif
