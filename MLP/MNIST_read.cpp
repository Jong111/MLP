// 假设MNIST数据集已经下载到本地，并且有四个文件：
// train-images-idx3-ubyte: 训练集图片，大小为60000*28*28
// train-labels-idx1-ubyte: 训练集标签，大小为60000
// t10k-images-idx3-ubyte: 测试集图片，大小为10000*28*28
// t10k-labels-idx1-ubyte: 测试集标签，大小为10000

#include <iostream>
#include <fstream>
#include <vector>
#include<assert.h>
#include "MNIST_read.h"

using namespace std;

// 一个辅助函数，用于读取大端字节序的整数
int read_int(ifstream& in) {
    unsigned char bytes[4];
    in.read((char*)bytes, 4);
    int result = (int)bytes[0] << 24 | (int)bytes[1] << 16 | (int)bytes[2] << 8 | (int)bytes[3];
    return result;
}

// 用于读取MNIST数据集，并将像素值和标签分别存储到两个vector中
void read_mnist(vector<vector<unsigned char>>& images, vector<unsigned char>& labels, string image_file, string label_file) {
    ifstream image_in(image_file, ios::binary); 
    ifstream label_in(label_file, ios::binary); 
    if (image_in.is_open() && label_in.is_open()) {
        int magic_number = read_int(image_in); // 读取图片文件的魔数，应该是2051
        int num_images = read_int(image_in); // 读取图片的数量
        int num_rows = read_int(image_in); // 读取图片的行数，应该是28
        int num_cols = read_int(image_in); // 读取图片的列数，应该是28
        magic_number = read_int(label_in); // 读取标签文件的魔数，应该是2049
        int num_labels = read_int(label_in); // 读取标签的数量，应该和图片的数量相同
        images.resize(num_images); 
        labels.resize(num_labels); 
        for (int i = 0; i < num_images; i++) {
            images[i].resize(num_rows * num_cols); 
            image_in.read((char*)&images[i][0], num_rows * num_cols); // 读取每个图片的像素值，存储到对应的vector中
            label_in.read((char*)&labels[i], 1); // 读取每个标签的值，存储到对应的vector中
        }
        image_in.close(); 
        label_in.close(); 
    }
    else {
        cout << "Error: cannot open files." << endl; 
    }
}


// 用于将images中的像素值（为unsigned char类型）转换为double，并将结果保存在dest_train中
void convert_to_double(vector<vector<unsigned char>>& images, vector<vector<double>>& dest_images) {
    assert(dest_images.size() == images.size());
    for (int i = 0; i < images.size(); i++) {
        assert(dest_images[i].size() == images[i].size());
        for (int j = 0; j < images[i].size(); j++) {
            dest_images[i][j] = static_cast<double>(images[i][j]);
        }
    }
}


// 用于将一个int型整数（范围为0-9）转换为onehot编码，并将结果保存在一个vector中
vector<double> generate_onehot(int n) {
    // 因为是手写数字识别任务，所以这里默认vector的大小为10
    vector<double>res(10);
    for (int i = 0; i < res.size(); i++) {
        if (i == n) {
            res[i] = 1.0;
        }
        else {
            res[i] = 0.0;
        }
    }
    return res;
}


// 用于将labels中的标签值（为unsigned char类型）转换为独热编码
// 然后将独热编码保存在一个新的二维数组dest中，注意，传入的dest_label的维度为labels.size()*num_classes
// 在手写数字识别任务中，num_classes为10
void convert_to_onehot(vector<unsigned char>& labels, vector<vector<double>>& dest_label) {
    assert(dest_label.size() == labels.size());
    for (int i = 0; i < labels.size(); i++) {
        labels[i] = static_cast<int>(labels[i]);
        vector<double> onehot = generate_onehot(labels[i]);
        dest_label[i] = onehot;
    }
}


// 用于将labels中的unsigned char型数字转换为int型
void convert_to_int(vector<unsigned char>& labels, vector<int>& dest_label) {
    assert(labels.size() == dest_label.size());
    for (int i = 0; i < dest_label.size(); i++) {
        dest_label[i] = static_cast<int>(labels[i]);
    }
}


// 用于对像素值归一化
void normalization(vector<vector<double>>& images) {
    for (int i = 0; i < images.size(); i++) {
        for (int j = 0; j < images[i].size(); j++) {
            images[i][j] /= 255.0;
        }
    }
}


//// 主函数，用于测试上述函数
//int main() {
//    vector<vector<unsigned char>> train_images; // 存储训练集图片的vector
//    vector<unsigned char> train_labels; // 存储训练集标签的vector
//    vector<vector<unsigned char>> test_images; // 存储测试集图片的vector
//    vector<unsigned char> test_labels; // 存储测试集标签的vector
//
//    read_mnist(train_images, train_labels, "D:\\mnist\\data\\train-images-idx3-ubyte", "D:\\mnist\\data\\train-labels-idx1-ubyte"); // 读取训练集数据
//    read_mnist(test_images, test_labels, "D:\\mnist\\data\\t10k-images-idx3-ubyte", "D:\\mnist\\data\\t10k-labels-idx1-ubyte"); // 读取测试集数据
//
//    vector<vector<double>> dest_train_images(train_images.size(), vector<double>(train_images[0].size()));
//    vector<vector<double>> dest_label(train_labels.size());
//    convert_to_double(train_images, dest_train_images);
//    convert_to_onehot(train_labels, dest_label);
//    normalization(dest_train_images);
//
//    //cout << "Train images size: " << train_images.size() << endl; // 输出训练集图片数量，应该是60000
//    //cout << "Train labels size: " << train_labels.size() << endl; // 输出训练集标签数量，应该是60000
//    //cout << "Test images size: " << test_images.size() << endl; // 输出测试集图片数量，应该是10000
//    //cout << "Test labels size: " << test_labels.size() << endl; // 输出测试集标签数量，应该是10000
//    int tt = 0;
//    return 0;
//}
