#ifndef FULLYCONH
#define FULLYCONH

#include<iostream>
#include<Eigen/Dense>
#include<memory>
#include "neuronbase.h"
#include "activ_func.h"
#include<autodiff/reverse/var.hpp>
#include<autodiff/reverse/var/eigen.hpp>
using namespace autodiff;
using namespace std;
using namespace Eigen;

template<class T>
class FullConnect:public NeuronBase,public T
{
private:
    shared_ptr<NeuronBase> layer_pointer;
    //MatrixXvar weight;
    MatrixXd weight;
    //VectorXvar bias;
    VectorXd bias;
    VectorXd dlossdy;
    MatrixXvar weight_gradient;
    VectorXd bias_gradient;


public:
    FullConnect(int num_neurons, shared_ptr<NeuronBase> _layer_pointer)                        //construction func is used to initialize a layer
        :NeuronBase(num_neurons),
         layer_pointer(_layer_pointer),
         T(num_neurons)

    {

         weight = MatrixXd::Random(get_next_layer_neurons_address()->get_num_neurons(), num_neurons);
         bias = VectorXd::Random(get_next_layer_neurons_address()->get_num_neurons());

         cout<<"A layer is built, it has "<<num_neurons<<" neurons"<<endl;
    }

public:

    var activation(VectorXvar x,bool flag)                           //this func is used to activate a layer of neurons
    {


        double learning_rate = 1e-4;

        MatrixXvar weight_var = weight;

        VectorXvar y = weight_var*T::h(x)+bias;

        var loss_func = layer_pointer->activation(y,1);

        if(flag == 1)
            single_layer_backward(loss_func,y,x,learning_rate);

        return loss_func;
    }

    void calculate_gradient(var loss_func, VectorXvar y,VectorXvar x)                                    //backward propagation of the single layer
    {
        dlossdy = gradient(loss_func,y);
        weight_gradient = dlossdy*x.transpose();
        bias_gradient = dlossdy;

    }
    void single_layer_backward(var loss_func,VectorXvar y,VectorXvar x,double learning_rate)
    {
        calculate_gradient(loss_func, y, x);
        update_para(learning_rate);
    }
    void update_para(double learning_rate)
    {

           weight = weight - weight_gradient.cast<double>()*learning_rate;
           bias = bias - bias_gradient.cast<double>()*learning_rate;
    }
    shared_ptr<NeuronBase> get_next_layer_neurons_address()
    {
        return layer_pointer;
    }



};

#endif
