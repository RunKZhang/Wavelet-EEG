#ifndef PROCESS
#define PROcESS
#include<iostream>
#include<fstream>
#include<vector>
#include<Eigen/Dense>
#include "WT.h"
using namespace std;
using namespace Eigen;

vector<vector<string>> get_new_dataset()
{
    ifstream inFile;
    inFile.open("../data/rebuilt_eye_movement.csv",ios::in);
    if(inFile)
    {
        cout<<"yes"<<endl;

    }
    else
    {
        cout<<"does not exist"<<endl;
    }
    vector<string> row;
    vector<vector<string>> content;
    string line, word;
    if(inFile.is_open())
    {
        while(getline(inFile,line))
        {
            row.clear();
            stringstream str(line);
            while(getline(str,word,','))
              {
                row.push_back(word);

               }
            content.push_back(row);
        }
    }
    inFile.close();
    return content;
}
vector<vector<string>> get_test_dataset()
{
    ifstream inFile;
    inFile.open("../data/test_dataset.csv",ios::in);
    if(inFile)
    {
        cout<<"yes"<<endl;

    }
    else
    {
        cout<<"does not exist"<<endl;
    }
    vector<string> row;
    vector<vector<string>> content;
    string line, word;
    if(inFile.is_open())
    {
        while(getline(inFile,line))
        {
            row.clear();
            stringstream str(line);
            while(getline(str,word,','))
              {
                row.push_back(word);

               }
            content.push_back(row);
        }
    }
    inFile.close();
    return content;
}

MatrixXd transfer_test_dataset(vector<vector<string>> &content)
{
    MatrixXd content_matrix (2800,6);
    for(int i=1;i<2801;i++)
        for(int j=0;j<6;j++)
            content_matrix(i-1,j) = stod(content[i][j]);
    return content_matrix;
}
MatrixXd transfer_dataset(vector<vector<string>> &content)
{
    MatrixXd content_matrix (2800,9);
    for(int i=1;i<2801;i++)
        for(int j=0;j<9;j++)
            content_matrix(i-1,j) = stod(content[i][j]);
    return content_matrix;
}
VectorXd pick_wave(MatrixXd original_data, int i, int j, int len)
{
    return original_data.block(i,j,1,len).transpose();
}
MatrixXd rectifier(MatrixXd mat)
{
    Index max_index,min_index;
    MatrixXd new_mat=mat;
    for(int i=0;i<mat.cols()-1;i++)
        {
            if(mat.col(i).maxCoeff(&max_index)>5000)
            {
                new_mat(max_index,i)= (mat.col(i)).mean();
            }
            if(mat.col(i).minCoeff(&min_index)<3850)
            {
                new_mat(min_index,i)=(mat.col(i)).mean();
            }

        }
    return new_mat;
}
MatrixXd normalize(MatrixXd mat)
{
    MatrixXd new_mat(mat.rows(),mat.cols());
    for(int i=0;i<mat.cols()-1;i++)
    {
        new_mat.col(i)=mat.col(i).array()-mat.col(i).mean();
    }
    if(mat.cols()>6)
        new_mat.col(8) = mat.col(8);
    return new_mat;
}
MatrixXd append(MatrixXd a, VectorXd b)
{
    MatrixXd c(a.rows(),a.cols()+1);
    c<<a,b;
    return c;
}
MatrixXd append(VectorXd a, VectorXd b)
{
    MatrixXd c(a.size(),2);
    c<<a,b;
    return c;
}
void make_input_feature(MatrixXd &input_feature, MatrixXd mat)
{
    VectorXd time=VectorXd::LinSpaced(200,0,199);

    for(int i=0;i<14;i++)
    {
        for(int j=0;j<mat.cols()-1;j++)
        {
             wavelet wav(time,mat.block(i*200,j,200,1),200);
             MatrixXd temp = wav.wavelet_transform();
             temp.resize(320,1);
             input_feature = append(input_feature,temp);
        }
    }
}
void make_time_series_feature(MatrixXd &input_feature, MatrixXd mat)
{
    //VectorXd time=VectorXd::LinSpaced(200,0,199);

    for(int i=0;i<14;i++)
    {
        for(int j=0;j<mat.cols()-1;j++)
        {
             //wavelet wav(time,mat.block(i*200,j,200,1),200);
             VectorXd temp = mat.block(i*200,j,200,1);
             input_feature = append(input_feature,temp);
        }
    }
}
void make_output_label(MatrixXd &output_label, MatrixXd mat)
{
    VectorXd inner_10(2);
    VectorXd inner_01(2);
    inner_01<<0,1;
    inner_10<<1,0;
    for(int i=90;i<=2800;i+=200)
    {
       if(mat(i,8)==0 && mat(i+20,8)==1)
        {
            for(int j=0;j<8;j++)
            {
                output_label = append(output_label,inner_01);
            }
        }
        else
        {
           for(int j=0;j<8;j++)
           {
                output_label = append(output_label,inner_10);
           }
        }
    }
}
void make_test_output_label(MatrixXd &output_label, MatrixXd mat)
{
    VectorXd inner_10(2);
    VectorXd inner_01(2);
    inner_01<<0,1;
    inner_10<<1,0;
    for(int i=90;i<=2800;i+=200)
    {
       if(mat(i,8)==0 && mat(i+20,8)==1)
        {
            for(int j=0;j<5;j++)
            {
                output_label = append(output_label,inner_01);
            }
        }
        else
        {
           for(int j=0;j<5;j++)
           {
                output_label = append(output_label,inner_10);
           }
        }
    }
}
MatrixXd set_onset_length(vector<vector<string>> &content)
{
    MatrixXd onlength (2129,2);
    for(int i=0;i<2129;i++)
    {
        onlength(i,0) = stod(content[i+1][2]);
        onlength(i,1) = stod(content[i+1][9]);
    }
    return onlength;
}
void remake_train_test(MatrixXd &mat_train, MatrixXd &mat_test)
{
    for(int i=0;i<6;i+=2)
    {
        auto temp = mat_train.col(i);
        mat_train.col(i) = mat_test.col(i);
        mat_test.col(i) = temp;
        cout<<temp.rows()<<" "<<temp.cols()<<endl;
    }
}

MatrixXd set_one_hot_lexical(vector<vector<string>> &content)
{
    MatrixXd one_hot_matrix (2129,2);

    for(int i=1;i<=2129;i++)
    {
        if(content[i][12]=="0")
            one_hot_matrix.row(i-1)<<1,0;
        else
            one_hot_matrix.row(i-1)<<0,1;
    }
    one_hot_matrix.transposeInPlace();
    return one_hot_matrix;
}
int show_row_length(vector<vector<string>> &content)
{
    return content.size();
}
long show_datasetcolumn_length(vector<vector<string>> &content)
{
    return content[61].size();
}
int show_descriptioncolumn_length(vector<vector<string>> &content)
{
    return content[2129].size();
}
#endif
