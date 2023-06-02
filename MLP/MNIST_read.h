#pragma once
#include<vector>
#include<string>

using namespace std;

extern int read_int(ifstream& in);
extern void read_mnist(vector<vector<unsigned char>>& images, vector<unsigned char>& labels, string image_file, string label_file);
extern void convert_to_double(vector<vector<unsigned char>>& images, vector<vector<double>>& dest_train);
extern vector<int> generate_onehot(int n);
extern void convert_to_onehot(vector<unsigned char>& labels, vector<vector<int>>& dest);
extern void normalization(vector<vector<double>>& labels);
