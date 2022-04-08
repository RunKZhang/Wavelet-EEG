#ifndef MSE_H
#define MSE_H


#include<iostream>
#include<Eigen/Dense>
#include<memory>
#include "neuronbase.h"
#include<autodiff/reverse/var.hpp>
#include<autodiff/reverse/var/eigen.hpp>
using namespace autodiff;
using namespace std;
using namespace Eigen;

template<class T>
class Mse:public NeuronBase,public T
{
private:
    VectorXvar m_output_label;
    var loss_value;
    Index max_Row_y,max_Row_label;
    double m_acc;
public:
    Mse(int num_neurons)                        //construction func is used to initialize a layer
        :NeuronBase(num_neurons),
         T(num_neurons)
    {

        cout<<"Output layer is built, it has "<<num_neurons<<" neurons"<<endl;

    }
public:
    var activation(VectorXvar y,bool flag)                           //this func is used to activate a layer of neurons
    {
        //set_current_layer_vector(T::h(y));
        //loss_value = calculate_cross_entropy(get_current_layer_vector());
        loss_value = calculate_cross_entropy(T::h(y));

        calculate_accuracy(y);
        return loss_value;
    }
    void set_output_label(VectorXvar output_label)
    {
        m_output_label = output_label;
    }

    VectorXvar get_output_label()
    {
        return m_output_label;
    }

    var calculate_mse(VectorXvar y)
    {
        return ((y-m_output_label).cwiseProduct(y-m_output_label)).sum()/get_num_neurons();
    }
    var calculate_cross_entropy(VectorXvar y)
    {
        double delta = 1e-7;
        //cout<<"output:"<<y<<endl;
        //cout<<"label:"<<m_output_label<<endl;
        return -((m_output_label.array())*(log(y.array()+delta))).sum();

    }
    var get_loss_value()
    {
        return loss_value;
    }
    void calculate_accuracy(VectorXvar y_in)
    {
        T::h(y_in).maxCoeff(&max_Row_y);
        m_output_label.maxCoeff(&max_Row_label);
        if(max_Row_y == max_Row_label)
            m_acc+=1;
    }
    void set_acc()
    {
        m_acc = 0;
    }
    double get_acc()
    {
        return m_acc;
    }

};

#endif
