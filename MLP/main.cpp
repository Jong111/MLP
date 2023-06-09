#include<iostream>
#include<vector>
#include<string>
#include<assert.h>
#include<cmath>
#include<random>
#include"tools.h"
#include"mlp.h"
#include"MNIST_read.h"
#include <fstream>

using namespace std;

int main() {
	vector<vector<unsigned char>> train_images; // ¥Ê¥¢—µ¡∑ºØÕº∆¨
    vector<unsigned char> train_labels; // ¥Ê¥¢—µ¡∑ºØ±Í«©
    vector<vector<unsigned char>> test_images; // ¥Ê¥¢≤‚ ‘ºØÕº∆¨
    vector<unsigned char> test_labels; // ¥Ê¥¢≤‚ ‘ºØ±Í«©

    read_mnist(train_images, train_labels, "D:\\mnist\\data\\train-images-idx3-ubyte", "D:\\mnist\\data\\train-labels-idx1-ubyte"); 
    read_mnist(test_images, test_labels, "D:\\mnist\\data\\t10k-images-idx3-ubyte", "D:\\mnist\\data\\t10k-labels-idx1-ubyte"); 

    vector<vector<double>> dest_train_images(train_images.size(), vector<double>(train_images[0].size()));
    vector<vector<double>> dest_train_label(train_labels.size());
    vector<int> dest_train_label_int(train_labels.size());
    convert_to_int(train_labels, dest_train_label_int);
    convert_to_double(train_images, dest_train_images);
    convert_to_onehot(train_labels, dest_train_label);
    normalization(dest_train_images);

    vector<vector<double>> dest_test_images(test_images.size(), vector<double>(test_images[0].size()));
    vector<int> dest_test_label_int(test_labels.size());
    convert_to_int(test_labels, dest_test_label_int);
    convert_to_double(test_images, dest_test_images);
    normalization(dest_test_images);

    // 
    string active_func = "relu";
    LinearLayer lin1(784, 20, active_func);
    LinearLayer* plin1 = &lin1;
    ActivationLayer act1(active_func);
    ActivationLayer* pact1 = &act1;
    LinearLayer lin2(20, 20, active_func);
    LinearLayer* plin2 = &lin2;
    ActivationLayer act2(active_func);
    ActivationLayer* pact2 = &act2;
    LinearLayer lin3(20, 10, "softmax");
    LinearLayer* plin3 = &lin3;
    ActivationLayer act3("softmax");
    ActivationLayer* pact3 = &act3;
    LossLayer loss; 
    LossLayer* ploss = &loss;


    Network net;
    net.add_layer(plin1);
    net.add_layer(pact1);
    net.add_layer(plin2);
    net.add_layer(pact2);
    net.add_layer(plin3);
    net.add_layer(pact3);
    net.add_layer(ploss);

    int epochs = 2;
    int batch_size = 32;

    net.train(dest_train_images, dest_train_label, epochs, batch_size);
    double train_acc = net.eval(dest_train_images, dest_train_label_int);
    double test_acc = net.eval(dest_test_images, dest_test_label_int);
    cout << "train acc: " << train_acc << endl;
    cout << "test acc: " << test_acc << endl;

    return 0;
}