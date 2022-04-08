#include<iostream>
#include<Eigen/Dense>
#include<complex>
#include "WT.h"
#include "process_data.h"
#include <activ_func.h>
#include "fully_connected.h"
#include "mse.h"
#include<autodiff/reverse/var.hpp>
#include<autodiff/reverse/var/eigen.hpp>
using namespace std;
using namespace Eigen;


int main()
{
    auto p_layer_output =make_shared<Mse<Softmax>>(2);
    //auto p_layer3 =make_shared<FullConnect<Tanh>>(64,dynamic_pointer_cast<NeuronBase>(p_layer_output));
    auto p_layer2 =make_shared<FullConnect<Tanh>>(64,dynamic_pointer_cast<NeuronBase>(p_layer_output));
    auto p_layer1 =make_shared<FullConnect<linear>>(320,dynamic_pointer_cast<NeuronBase>(p_layer2));
    vector<vector<string>> outer_content;
    outer_content = get_new_dataset();
    auto new_dataset = transfer_dataset(outer_content);
    auto recti_data = rectifier(new_dataset);
    auto normal_data = normalize(recti_data);

    vector<vector<string>> test_dataset;
    test_dataset = get_test_dataset();
    auto new_test_dataset = transfer_test_dataset(test_dataset);
    auto recti_test_data = rectifier(new_test_dataset);
    auto normal_test_data = normalize(recti_test_data);
    cout<<normal_test_data.rows()<<" "<<normal_test_data.cols()<<endl;

    remake_train_test(normal_data,normal_test_data);

    VectorXd time=VectorXd::LinSpaced(200,0,199);                   //Use wavelet transform
    wavelet wav1(time,normal_data.block(0,0,200,1),200);
    wavelet wav2(time,normal_data.block(0,1,200,1),200);
    MatrixXd temp1 = wav1.wavelet_transform();
    MatrixXd temp2 = wav2.wavelet_transform();
    temp1.resize(320,1);
    temp2.resize(320,1);
    MatrixXd input_feature = append(temp1,temp2);
    make_input_feature(input_feature,normal_data);
    cout<<input_feature.rows()<<" "<<input_feature.cols()<<endl;

    VectorXd label_temp1(2);                                            //Build labels;
    VectorXd label_temp2(2);
    label_temp1<<0,1;
    label_temp2<<0,1;
    MatrixXd output_label = append(label_temp1,label_temp2);
    make_output_label(output_label,normal_data);
    cout<<output_label.rows()<<" "<<output_label.cols()<<endl;

    VectorXd time_series1 = normal_data.block(0,0,200,1);
    VectorXd time_series2 = normal_data.block(0,1,200,1);
    MatrixXd input_time_series = append(time_series1,time_series2);
    make_time_series_feature(input_time_series,normal_data);
    cout<<input_time_series.rows()<<" "<<input_time_series.cols()<<endl;

    wavelet test_wav1(time,normal_test_data.block(0,0,200,1),200);
    wavelet test_wav2(time,normal_test_data.block(0,1,200,1),200);
    MatrixXd test_temp1 = wav1.wavelet_transform();
    MatrixXd test_temp2 = wav2.wavelet_transform();
    test_temp1.resize(320,1);
    test_temp2.resize(320,1);
    MatrixXd test_feature = append(test_temp1,test_temp2);
    make_input_feature(test_feature,normal_test_data);
    cout<<test_feature.rows()<<" "<<test_feature.cols()<<endl;

    VectorXd test_label_temp1(2);                                            //Build labels;
    VectorXd test_label_temp2(2);
    test_label_temp1<<0,1;
    test_label_temp2<<0,1;
    MatrixXd test_output_label = append(test_label_temp1,test_label_temp2);
    make_test_output_label(test_output_label,normal_data);
    cout<<test_output_label.rows()<<" "<<test_output_label.cols()<<endl;

    double accuracy=0;
    int i = 0;
    p_layer_output->set_acc();

    for(int j=0;j<11300;j++)
    {


        i = (i+23)%113;
        p_layer_output ->set_output_label(output_label.col(i));
        p_layer1 -> activation(input_feature.col(i),1);

        if(j%113 == 0)
        {
            accuracy = p_layer_output->get_acc()/113*100;
            cout<<"epoch:"<<(j/113)<<" The accuracy is:"<<accuracy<<"%"<<"The loss is:"<<p_layer_output->get_loss_value()<<endl;
            p_layer_output->set_acc();

            double test_accuracy = 0;
            for(int i=0;i<72;i++)
            {
                p_layer_output ->set_output_label(test_output_label.col(i));
                p_layer1 -> activation(test_feature.col(i),0);
            }
            test_accuracy = p_layer_output ->get_acc()/72*100;
            cout<<"The accuracy is:"<<test_accuracy<<endl;

            p_layer_output->set_acc();
        }
    }




    return 0;
}
