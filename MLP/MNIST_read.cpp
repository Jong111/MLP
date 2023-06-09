// ����MNIST���ݼ��Ѿ����ص����أ��������ĸ��ļ���
// train-images-idx3-ubyte: ѵ����ͼƬ����СΪ60000*28*28
// train-labels-idx1-ubyte: ѵ������ǩ����СΪ60000
// t10k-images-idx3-ubyte: ���Լ�ͼƬ����СΪ10000*28*28
// t10k-labels-idx1-ubyte: ���Լ���ǩ����СΪ10000

#include <iostream>
#include <fstream>
#include <vector>
#include<assert.h>
#include "MNIST_read.h"

using namespace std;

// һ���������������ڶ�ȡ����ֽ��������
int read_int(ifstream& in) {
    unsigned char bytes[4];
    in.read((char*)bytes, 4);
    int result = (int)bytes[0] << 24 | (int)bytes[1] << 16 | (int)bytes[2] << 8 | (int)bytes[3];
    return result;
}

// ���ڶ�ȡMNIST���ݼ�����������ֵ�ͱ�ǩ�ֱ�洢������vector��
void read_mnist(vector<vector<unsigned char>>& images, vector<unsigned char>& labels, string image_file, string label_file) {
    ifstream image_in(image_file, ios::binary); 
    ifstream label_in(label_file, ios::binary); 
    if (image_in.is_open() && label_in.is_open()) {
        int magic_number = read_int(image_in); // ��ȡͼƬ�ļ���ħ����Ӧ����2051
        int num_images = read_int(image_in); // ��ȡͼƬ������
        int num_rows = read_int(image_in); // ��ȡͼƬ��������Ӧ����28
        int num_cols = read_int(image_in); // ��ȡͼƬ��������Ӧ����28
        magic_number = read_int(label_in); // ��ȡ��ǩ�ļ���ħ����Ӧ����2049
        int num_labels = read_int(label_in); // ��ȡ��ǩ��������Ӧ�ú�ͼƬ��������ͬ
        images.resize(num_images); 
        labels.resize(num_labels); 
        for (int i = 0; i < num_images; i++) {
            images[i].resize(num_rows * num_cols); 
            image_in.read((char*)&images[i][0], num_rows * num_cols); // ��ȡÿ��ͼƬ������ֵ���洢����Ӧ��vector��
            label_in.read((char*)&labels[i], 1); // ��ȡÿ����ǩ��ֵ���洢����Ӧ��vector��
        }
        image_in.close(); 
        label_in.close(); 
    }
    else {
        cout << "Error: cannot open files." << endl; 
    }
}


// ���ڽ�images�е�����ֵ��Ϊunsigned char���ͣ�ת��Ϊdouble���������������dest_train��
void convert_to_double(vector<vector<unsigned char>>& images, vector<vector<double>>& dest_images) {
    assert(dest_images.size() == images.size());
    for (int i = 0; i < images.size(); i++) {
        assert(dest_images[i].size() == images[i].size());
        for (int j = 0; j < images[i].size(); j++) {
            dest_images[i][j] = static_cast<double>(images[i][j]);
        }
    }
}


// ���ڽ�һ��int����������ΧΪ0-9��ת��Ϊonehot���룬�������������һ��vector��
vector<double> generate_onehot(int n) {
    // ��Ϊ����д����ʶ��������������Ĭ��vector�Ĵ�СΪ10
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


// ���ڽ�labels�еı�ǩֵ��Ϊunsigned char���ͣ�ת��Ϊ���ȱ���
// Ȼ�󽫶��ȱ��뱣����һ���µĶ�ά����dest�У�ע�⣬�����dest_label��ά��Ϊlabels.size()*num_classes
// ����д����ʶ�������У�num_classesΪ10
void convert_to_onehot(vector<unsigned char>& labels, vector<vector<double>>& dest_label) {
    assert(dest_label.size() == labels.size());
    for (int i = 0; i < labels.size(); i++) {
        labels[i] = static_cast<int>(labels[i]);
        vector<double> onehot = generate_onehot(labels[i]);
        dest_label[i] = onehot;
    }
}


// ���ڽ�labels�е�unsigned char������ת��Ϊint��
void convert_to_int(vector<unsigned char>& labels, vector<int>& dest_label) {
    assert(labels.size() == dest_label.size());
    for (int i = 0; i < dest_label.size(); i++) {
        dest_label[i] = static_cast<int>(labels[i]);
    }
}


// ���ڶ�����ֵ��һ��
void normalization(vector<vector<double>>& images) {
    for (int i = 0; i < images.size(); i++) {
        for (int j = 0; j < images[i].size(); j++) {
            images[i][j] /= 255.0;
        }
    }
}


//// �����������ڲ�����������
//int main() {
//    vector<vector<unsigned char>> train_images; // �洢ѵ����ͼƬ��vector
//    vector<unsigned char> train_labels; // �洢ѵ������ǩ��vector
//    vector<vector<unsigned char>> test_images; // �洢���Լ�ͼƬ��vector
//    vector<unsigned char> test_labels; // �洢���Լ���ǩ��vector
//
//    read_mnist(train_images, train_labels, "D:\\mnist\\data\\train-images-idx3-ubyte", "D:\\mnist\\data\\train-labels-idx1-ubyte"); // ��ȡѵ��������
//    read_mnist(test_images, test_labels, "D:\\mnist\\data\\t10k-images-idx3-ubyte", "D:\\mnist\\data\\t10k-labels-idx1-ubyte"); // ��ȡ���Լ�����
//
//    vector<vector<double>> dest_train_images(train_images.size(), vector<double>(train_images[0].size()));
//    vector<vector<double>> dest_label(train_labels.size());
//    convert_to_double(train_images, dest_train_images);
//    convert_to_onehot(train_labels, dest_label);
//    normalization(dest_train_images);
//
//    //cout << "Train images size: " << train_images.size() << endl; // ���ѵ����ͼƬ������Ӧ����60000
//    //cout << "Train labels size: " << train_labels.size() << endl; // ���ѵ������ǩ������Ӧ����60000
//    //cout << "Test images size: " << test_images.size() << endl; // ������Լ�ͼƬ������Ӧ����10000
//    //cout << "Test labels size: " << test_labels.size() << endl; // ������Լ���ǩ������Ӧ����10000
//    int tt = 0;
//    return 0;
//}
