#pragma once
#include<vector>
#include<string>
#include<iostream>

using namespace std;


extern vector<double> matrix_multi(vector<vector<double>> weight, vector<double> input);
extern vector<double> matrix_sum(vector<double> wx, vector<double> b);
extern vector<double> softmax(vector<double> e);
extern vector<double> sigmoid(vector<double> e);
extern double cross_entropy(vector<double> y, vector<double> a);
extern vector<double> matrix_sub(vector<double> output, vector<double> y);
extern vector<double> matrix_hadamard_product(vector<double> m1, vector<double> m2);
extern vector<vector<double>> matrix_scalar_multi(vector<vector<double>> m, double a);
extern vector<vector<double>> matrix_sub(vector<vector<double>> a, vector<vector<double>> b);
extern vector<double> matrix_scalar_multi(vector<double> v, double a);
extern vector<vector<double>> matrix_multi(vector<double> v1, vector<double> v2);
extern vector<vector<double>> matrix_transpose(vector<vector<double>> matrix);
extern vector<vector<double>> Xavier_initialization(int n, int m);
extern vector<double> Xavier_initialization(int n);
extern void matrix_swap(vector<vector<double>>& m, int i, int k);
extern void shuffle(vector<vector<double>>& m1, vector<vector<double>>& m2);
extern vector<vector<double>> get_batch(vector<vector<double>> X_train, int i, int batch_size);
extern int arg_max(vector<double> vec);
extern vector<double> relu(vector<double> e);
extern vector<double> greater_than_zero(vector<double> e);
extern vector<vector<double>> Kaiming_initialization(int n, int m);
extern vector<double> Kaiming_initialization(int n);
extern vector<double> divided_by_1k(vector<double> e);
