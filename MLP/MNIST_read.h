#pragma once
#include<vector>
#include<string>

using namespace std;

extern int read_int(ifstream& in);
extern void read_mnist(vector<vector<unsigned char>>& images, vector<unsigned char>& labels, string image_file, string label_file);
extern void convert_to_double(vector<vector<unsigned char>>& images, vector<vector<double>>& dest_train);
extern vector<double> generate_onehot(int n);
extern void convert_to_onehot(vector<unsigned char>& labels, vector<vector<double>>& dest_label);
extern void normalization(vector<vector<double>>& labels);
extern void convert_to_int(vector<unsigned char>& labels, vector<int>& dest_label);
