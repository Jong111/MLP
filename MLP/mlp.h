#pragma once
#include<iostream>
#include<vector>
#include<string>
#include<memory>
#include "tools.h"
#include<assert.h>

using namespace std;

class Layer {
public:
	vector<double> input;
	vector<double> output;
	vector<vector<double>> weight;
	vector<double> bias;
	// e�Ǹ�����ļ�Ȩ��
	vector<double> e;
	int inputDim;
	int outputDim;
	string activation;
	vector<vector<double>> weight_gradient;
	vector<double> bias_gradient;
	virtual vector<double> forward(vector<double> input_data) = 0;
	virtual vector<double> backward(vector<double> input_gradient) = 0;
	virtual void update() = 0;
};


class LinearLayer : public Layer {
public:
	// ���캯��
	LinearLayer(int input_dim, int output_dim, string active_func) {
		assert(active_func == "sigmoid" || active_func == "softmax" || active_func == "relu");
		inputDim = input_dim;
		outputDim = output_dim;
		activation = active_func;
		input = vector<double>();
		output = vector<double>();
		if (activation == "sigmoid" || activation == "softmax" || activation == "relu") {
			// ����Xavier initialization����ʼ��weight����
			weight = Xavier_initialization(output_dim, input_dim);
			// ����Xavier initialization����ʼ��b����
			bias = Xavier_initialization(output_dim);
		}
		//else if(activation == "relu") {
		//	// ����Kaiming initialization����ʼ��weight����
		//	weight = Kaiming_initialization(output_dim, input_dim);
		//	// ����Kaiming initialization����ʼ��b����
		//	bias = Kaiming_initialization(output_dim);
		//}
		e = vector<double>();
		weight_gradient = vector<vector<double>>();
		bias_gradient = vector<double>();
	}

	// ��������
	~LinearLayer() {
		input.clear();
		output.clear();
		weight.clear();
		bias.clear();
		e.clear();
		weight_gradient.clear();
		bias_gradient.clear();
	}

	vector<double> forward(vector<double> input_data) {
		input = input_data;
		vector<double> wx = matrix_multi(weight, input);
		e = matrix_sum(wx, bias);
		return e;
	}

	vector<double> backward(vector<double> input_gradient) {
		// ����ò��Ȩ�ؾ����ƫ���������ݶȣ����洢��weight_gradient��bias_gradient������
		weight_gradient = matrix_multi(input_gradient, input);
		bias_gradient = input_gradient;
		// ����ò㴫�ݸ���һ����ݶȣ�������
		vector<double> output_gradient = matrix_multi(matrix_transpose(weight), input_gradient);
		return output_gradient;
	}

	void update() {
		// ����ѧϰ����0.01
		double learning_rate = 0.01;
		// �����ݶ��½���������Ȩ�ؾ����ƫ������
		weight = matrix_sub(weight, matrix_scalar_multi(weight_gradient, learning_rate));
		bias = matrix_sub(bias, matrix_scalar_multi(bias_gradient, learning_rate));
	}


};


class ActivationLayer : public Layer {
public:
	// ���캯��
	ActivationLayer(string activate_func) {
		activation = activate_func;
		input = vector<double>();
		output = vector<double>();
		e = vector<double>();
	}

	// ��������
	~ActivationLayer() {
		input.clear();
		output.clear();
		e.clear();
	}

	vector<double> forward(vector<double> input_data) {
		e = input_data;
		assert(activation == "sigmoid" || activation == "softmax" || activation == "relu");
		if (activation == "sigmoid") {
			output = sigmoid(e);
		}
		else if (activation == "relu") {
			output = relu(e);
		}
		else if (activation == "softmax") {
			output = softmax(e);
		}
		return output;
	}

	vector<double> backward(vector<double> input_gradient) {
		// ����һ��n*1ά��ȫ1����
		vector<double> one_matrix(output.size());
		for (int i = 0; i < one_matrix.size(); i++) {
			one_matrix[i] = 1;
		}
		// ����ò�ļ���ֵ��������һ��ļ�Ȩ������֮����ݶȣ�������
		if (activation == "sigmoid") {
			vector<double> output_gradient = matrix_hadamard_product(input_gradient, matrix_hadamard_product(output, matrix_sub(one_matrix, output)));
			return output_gradient;
		}
		else if (activation == "relu") {
			vector<double> relu_matrix = greater_than_zero(e);
			vector<double> output_gradient = matrix_hadamard_product(input_gradient, relu_matrix);
			return output_gradient;
		}
		else if (activation == "softmax") {
			return input_gradient;
		}
	}

	void update() {
		// ʲô������
	}

};


class LossLayer : public Layer {
public:
	// vector<double> y;

	// ���캯��
	LossLayer() {
		// y = vector<double>();
		output = vector<double>();
	}

	// ��������
	~LossLayer() {
		// y.clear();
		output.clear();
	}

	vector<double> forward(vector<double> input_data) {
		output = input_data;
		return output;
		//// ʹ�ý�������ʧ
		//double loss = cross_entropy(y, output);
		//vector<double> res(1);
		//res[0] = loss;
		//return res;
	}

	vector<double> backward(vector<double> y) {
		// ��ʹ�ý�������ʧ������softmax���������ô��ʧL�ͼ�Ȩ������֮����ݶȾ���Ԥ��ֵ��ȥ��ʵֵ
		vector<double> output_gradient = matrix_sub(output, y);
		return output_gradient;
	}

	void update() {
		// ʲô������
	}
};


class Network {
public:
	vector<double>input;
	vector<Layer*> layers;
	double loss;
	vector<double> gradient;
	vector<double> y;

	// ���캯��
	Network() {
		input = vector<double>();
		layers = vector<Layer*>();
		loss = 0.0;
		gradient = vector<double>();
	}

	// ��������
	~Network() {
		input.clear();
		/*for (int i = 0; i < layers.size(); i++) {
			delete layers[i];
		}
		layers.clear();*/
		gradient.clear();
	}

	void add_layer(Layer* layer) {
		// ���������ӵ�layers������
		layers.push_back(layer);
	}

	void forward() {
		for (int i = 0; i < layers.size(); i++) {
			input = layers[i]->forward(input);
		}
	}

	double calculate_loss(vector<double> input_data, vector<double> y_true) {
		// ʹ�ý�������ʧ
		double Loss = cross_entropy(y_true, input_data);
		return Loss;
	}

	void backward() {
		for (int i = layers.size() - 1; i >= 0; i--) {
			// ��������һ�㣬����ʵ��ǩ��Ϊ�������ݸ�backward��������������ֵ��ֵ��gradient����
			if (i == layers.size() - 1) {
				gradient = layers[i]->backward(y);
			}
			// ���򣬽���һ����ݶ���Ϊ�������ݸ�backward��������������ֵ��ֵ��gradient����
			else {
				gradient = layers[i]->backward(gradient);
			}
			// ����update���������¸ò�Ĳ���
			layers[i]->update();
		}
	}

	void train(vector<vector<double>> X_train, vector<vector<double>> y_train, int epochs, int batch_size) {
		for (int epoch = 0; epoch < epochs; epoch++) {
			// ����ѵ�����ݵ�˳��
			shuffle(X_train, y_train);
			for (int i = 0; i < X_train.size(); i += batch_size) {
				// ��ȡ��ǰ���ε��������ݺ���ʵ��ǩ
				vector<vector<double>> X_batch = get_batch(X_train, i, batch_size);
				vector<vector<double>> y_batch = get_batch(y_train, i, batch_size);
				for (int j = 0; j < X_batch.size(); j++) {
					input = X_batch[j];
					y = y_batch[j];
					forward();
					loss = calculate_loss(input, y);
					// cout << loss << endl;
					backward();
				}
			}
			// ��ӡ��ǰѵ�����ڵ���ʧ������ֵ
			cout << "Epoch " << epoch + 1 << ", loss: " << loss << endl;
		}
	}

	double eval(vector<vector<double>> X_test, vector<int> y_test) {
		assert(X_test.size() == y_test.size());
		// res�Ǳ���Ԥ���ǩ��vector
		vector<int> res(X_test.size());
		for (int i = 0; i < X_test.size(); i++) {
			input = X_test[i];
			forward();
			vector<double> output = layers[layers.size() - 1]->output;
			int predicted_val = arg_max(output);
			res[i] = predicted_val;
		}
		// ׼ȷ��
		double acc = 0.0;
		// Ԥ����ȷ�ĸ���
		double correct_num = 0.0;
		for (int i = 0; i < y_test.size(); i++) {
			if (res[i] == y_test[i]) {
				correct_num += 1;
			}
		}
		acc = correct_num / y_test.size();
		return acc;
	}

};
