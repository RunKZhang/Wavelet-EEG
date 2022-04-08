#include<iostream>
#include "fully_connected.h"
#include "mse.h"
#include<Eigen/Dense>
#include<autodiff/reverse/var.hpp>
#include<autodiff/reverse/var/eigen.hpp>
#include "test_nn.h"
#include "activ_func.h"
#include<fstream>
#include<vector>
#include "process_dataset.h"
using namespace std;
using namespace autodiff;

int main()
{
    auto p_layer_output =make_shared<Mse<Softmax>>(3);
    //auto p_layer3 =make_shared<FullConnect<Tanh>>(64,dynamic_pointer_cast<NeuronBase>(p_layer_output));
    auto p_layer2 =make_shared<FullConnect<Tanh>>(64,dynamic_pointer_cast<NeuronBase>(p_layer_output));
    auto p_layer1 =make_shared<FullConnect<linear>>(4,dynamic_pointer_cast<NeuronBase>(p_layer2));


    VectorXd v_new(1);
    VectorXd v_out_new(1);

    VectorXvar v_test_relu(5);
    v_test_relu<<0.2,0.2,0.5,0.05,0.05;
    VectorXvar v_test_relu_label(5);
    v_test_relu_label<<0,0,1,0,0;
    //p_layer_output ->set_output_label(v_test_relu_label);
    //p_layer1->activation(v_test_relu);

    //test_proportion_func(p_layer_output,p_layer1,v_new,v_out_new);

    vector<vector<string>> outer_content;
    outer_content = get_dataset();
    MatrixXd input_matrix = transfer_dataset(outer_content);
    MatrixXd label_matrix = set_one_hot_label(outer_content);
    double total_loss;
    double accuracy=0;
    int i = 0;
    p_layer_output->set_acc();
    for(int j=0;j<14900;j++)
    {


        i = (i+23)%149;
        p_layer_output ->set_output_label(label_matrix.col(i));
        p_layer1 -> activation(input_matrix.col(i),1);

        if(j%149 == 0)
        {
            accuracy = p_layer_output->get_acc()/149*100;
            cout<<"epoch:"<<(j/149)<<" The accuracy is:"<<accuracy<<"%"<<endl;
            p_layer_output->set_acc();
        }
    }

    return 0;
}
