#include<iostream>
#include<fstream>
#include<vector>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;


vector<vector<string>> get_dataset()
{
    ifstream inFile;
    inFile.open("../data/Iris.csv",ios::in);
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

void show_dataset(vector<vector<string>> content)
{
    for(int i=0;i<content.size();i++)
    {
        for(int j=0;j<content[i].size();j++)
            cout<<content[i][j]<<" ";
        cout<<endl;
    }

}
MatrixXd transfer_dataset(vector<vector<string>> content)
{
    MatrixXd content_matrix (150,4);
    for(int i=0;i<150;i++)
        for(int j=1;j<5;j++)
            content_matrix(i,j-1) = stod(content[i+1][j]);
    content_matrix.transposeInPlace();
    return content_matrix;
}
MatrixXd set_one_hot_label(vector<vector<string>> content)
{
    MatrixXd one_hot_matrix (150,3);

    for(int i=1;i<=150;i++)
    {
        if(content[i][5]=="Iris-setosa")
            one_hot_matrix.row(i-1)<<1,0,0;
        else
            if(content[i][5]=="Iris-versicolor")
                one_hot_matrix.row(i-1)<<0,1,0;
            else
                one_hot_matrix.row(i-1)<<0,0,1;
    }
    one_hot_matrix.transposeInPlace();
    return one_hot_matrix;
}
