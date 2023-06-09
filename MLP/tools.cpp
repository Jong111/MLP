#include<vector>
#include<string>
#include<iostream>
#include<assert.h>
#include<cmath>
#include<random>
#include<float.h>

using namespace std;


// 用于计算矩阵乘法，但该函数只能计算n*m和m*1维的矩阵相乘
vector<double> matrix_multi(vector<vector<double>> weight, vector<double> input) {
	assert(weight[0].size() == input.size());
	vector<double> res(weight.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = 0;
		for (int j = 0; j < weight[i].size(); j++) {
			res[i] += weight[i][j] * input[j];
		}
	}
	return res;
}


// 用于计算矩阵乘法，但该函数只能计算n*1和1*m维的矩阵相乘
vector<vector<double>> matrix_multi(vector<double> v1, vector<double> v2) {
	vector<vector<double>>res(v1.size(), vector<double>(v2.size()));
	for (int i = 0; i < res.size(); i++) {
		for (int j = 0; j < res[i].size(); j++) {
			res[i][j] = v1[i] * v2[j];
		}
	}
	return res;
}


// 用于将一个n*m维的矩阵转置为一个m*n维的矩阵
vector<vector<double>> matrix_transpose(vector<vector<double>> matrix) {
	int n = matrix.size();
	int m = matrix[0].size();
	vector<vector<double>> res(m, vector<double>(n));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			res[i][j] = matrix[j][i];
		}
	}
	return res;
}


// 用于对w进行Xavier initialization
vector<vector<double>> Xavier_initialization(int n, int m) {
	assert(m > 0 && n > 0);
	normal_distribution<double> dist(0.0, sqrt(6.0 / (n+m)));
	default_random_engine generator;
	vector<vector<double>>res(n, vector<double>(m));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			double number = dist(generator);
			res[i][j] = number;
		}
	}
	cout << "Initialize success" << endl;
	return res;
}


// 用于对b进行Xavier initialization
vector<double> Xavier_initialization(int n) {
	assert(n > 0);
	vector<double> res(n);
	for (int i = 0; i < n; i++) {
		res[i] = 0;
	}
	return res;
}


// 用于对w进行kaiming initialization
vector<vector<double>> Kaiming_initialization(int n, int m) {
	assert(n > 0 && m > 0);
	cout << "hi" << endl;
	normal_distribution<double> dist(0.0, sqrt((1.0 * m) / 2));
	cout << "hi" << endl;
	default_random_engine generator;
	vector<vector<double>>res(n, vector<double>(m));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			double number = dist(generator);
			res[i][j] = number;
		}
	}
	return res;
}


// 用于对b进行kaiming initialization
vector<double> Kaiming_initialization(int n) {
	assert(n > 0);
	vector<double>res(n);
	for (int i = 0; i < n; i++) {
		res[i] = 0;
	}
	return res;
}


// 用于矩阵加法，但该函数只能计算n*1和n*1维的矩阵相加
vector<double> matrix_sum(vector<double> wx, vector<double> b) {
	assert(wx.size() == b.size());
	vector<double> res(wx.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = wx[i] + b[i];
	}
	return res;
}


// 用于实现sigmoid函数，该函数接受一个n*1维向量e作为参数，返回一个新的向量res
// 其中res的元素是e中对应位置的元素sigmoid后的结果
vector<double> sigmoid(vector<double> e) {
	vector<double> res(e.size());
	for (int i = 0; i < e.size(); i++) {
		double v = 1.0 / (1 + exp(-e[i]));
		res[i] = v;
	}
	return res;
 }


// 用于实现relu函数，该函数接受一个n*1维向量e作为参数，返回一个新的向量res
// 其中res的元素是e中对应位置的元素relu后的结果
vector<double> relu(vector<double> e) {
	vector<double> res(e.size());
	for (int i = 0; i < res.size(); i++) {
		if (e[i] > 0) {
			res[i] = e[i];
		}
		else {
			res[i] = 0;
		}
	}
	return res;
}


// 用于实现softmax函数，该函数接受一个n*1维向量e作为参数，返回一个新的向量res
// 其中res=softmax(e)
vector<double> softmax(vector<double> e) {
	vector<double> res(e.size());
	double sum = 0.0;
	for (int i = 0; i < e.size(); i++) {
		sum += exp(e[i]);
	}
	for (int i = 0; i < res.size(); i++) {
		res[i] = exp(e[i]) / sum;
	}
	return res;
}


// 用于判断一个n*1维向量 e 中的每个元素是否大于0，返回一个结果向量 res
// 如果 e[i]>0 则 res[i] = 1，否则 res[i] = 0
vector<double> greater_than_zero(vector<double> e) {
	vector<double> res(e.size());
	for (int i = 0; i < e.size(); i++) {
		if (e[i] > 0) {
			res[i] = 1;
		}
		else {
			res[i] = 0;
		}
	}
	return res;
}


// 接受一个参数 e 返回一个向量res，其中 res[i] = e[i] / 1000
vector<double> divided_by_1k(vector<double> e) {
	vector<double> res(e.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = (e[i] * 1.0) / 1000;
	}
	return res;
}


// 用于计算交叉熵损失，函数接受两个同等大小的向量作为参数，返回它们的交叉熵
double cross_entropy(vector<double> y, vector<double> a) {
	assert(y.size() == a.size());
	double sum = 0;
	for (int i = 0; i < y.size(); i++) {
		sum += (y[i] * log(a[i]));
	}
	return -1 * sum;
}


// 用于计算两个向量之差，函数接受两个向量output和y，返回output和y对应位置之差
vector<double> matrix_sub(vector<double> output, vector<double> y) {
	assert(output.size() == y.size());
	vector<double> res(output.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = output[i] - y[i];
	}
	return res;
}


// 用于计算两个矩阵之差，函数接受两个矩阵a和b，返回a和b对应位置之差
vector<vector<double>> matrix_sub(vector<vector<double>> a, vector<vector<double>> b) {
	assert(a.size() == b.size());
	vector<vector<double>> res(a.size(), vector<double>(a[0].size()));
	for (int i = 0; i < res.size(); i++) {
		assert(a[i].size() == b[i].size());
		for (int j = 0; j < a[i].size(); j++) {
			res[i][j] = a[i][j] - b[i][j];
		}
	}
	return res;
}


// 用于计算两个矩阵的hadamard积，但该函数只能计算n*1维的矩阵间的hadamard积
vector<double> matrix_hadamard_product(vector<double> m1, vector<double> m2) {
	assert(m1.size() == m2.size());
	vector<double> res(m1.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = m1[i] * m2[i];
	}
	return res;
}


// 用于计算一个矩阵和一个标量的乘积，函数接受一个n*m维矩阵，一个标量，返回乘积结果
vector<vector<double>> matrix_scalar_multi(vector<vector<double>> m, double a) {
	vector<vector<double>> res(m.size(), vector<double>(m[0].size()));
	for (int i = 0; i < m.size(); i++) {
		for (int j = 0; j < m[i].size(); j++) {
			res[i][j] = m[i][j] * a;
		}
	}
	return res;
}


// 用于计算一个向量和一个标量的乘积，函数接受一个n*1维向量，一个标量，返回乘积结果
vector<double> matrix_scalar_multi(vector<double> v, double a) {
	vector<double> res(v.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = v[i] * a;
	}
	return res;
}


// 用于交换矩阵的两行，参数m表示一个n*m维矩阵，参数 i 和 k 表示交换m1的第 i 行和第 k 行
void matrix_swap(vector<vector<double>>&m, int i, int k) {
	assert(i >= 0 && i < m.size() && k >= 0 && k < m.size());
	vector<double> tmp = m[k];
	m[k] = m[i];
	m[i] = tmp;
}


// 用于对两个矩阵随机打乱其行的顺序，接受两个n*m维的矩阵m1和m2作为参数，无返回值
// 但要保证，若a1和a2分别是m1和m2中的第 i 行，则打乱两个矩阵的行的顺序后a1和a2必须都在m1和m2中的第 k 行
void shuffle(vector<vector<double>>&m1, vector<vector<double>>&m2) {
	assert(m1.size() == m2.size());
	int n = m1.size();
	// used用于记录一行是否已经被交换过，若第 i 行还未被交换，则used[i]=0，反之used[i]=1
	vector<int> used(n);
	for (int i = 0; i < n; i++) {
		used[i] = 0;
	}
	uniform_int_distribution<int> int_dist(0, n - 1);
	default_random_engine generator;
	for (int i = 0; i < n; i++) {
		if (used[i] == 1) {
			continue;
		}
		else {
			// 第 i 行和第 k 行交换 
			int k = int_dist(generator);
			used[i] = 1;
			used[k] = 1;
			matrix_swap(m1, i, k);
			matrix_swap(m2, i, k);
		}
	}
}


// 用于获取数据集X_train中以 i 为起点，长度为batch_size的子数据集
// 如果X_train的剩余的样本数不足batch_size，则直接返回X_train中所有剩余的样本
vector<vector<double>> get_batch(vector<vector<double>> X_train, int i, int batch_size) {
	assert(i < X_train.size());
	vector<vector<double>> res;
	for (int k = i; k < i + batch_size; k++) {
		if (k >= X_train.size()) {
			break;
		}
		vector<double> v = X_train[k];
		res.push_back(v);
	}
	return res;
}


// 用于获取一个向量中最大元素的下标，若该向量中有多个相等的最大值，则返回第一个最大值的下标
int arg_max(vector<double> vec) {
	double max = -DBL_MAX;
	int res = -1;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] > max) {
			res = i;
			max = vec[i];
		}
	}
	return res;
}


//int main() {
//	vector<double> a(4);
//	vector<double> b(3);
//	for (int i = 0; i < 4; i++) {
//		double tmp;
//		cin >> tmp;
//		a[i] = tmp;
//	}
//	/*for (int i = 0; i < 3; i++) {
//		double tmp;
//		cin >> tmp;
//		b[i] = tmp;
//	}*/
//	/*vector<vector<double>> e(3, vector<double>(2));
//	for (int i = 0; i < 3; i++) {
//		for (int j = 0; j < 2; j++) {
//			double tmp;
//			cin >> tmp;
//			e[i][j] = tmp;
//		}
//	}*/
//
//	/*vector<vector<double>> f(6, vector<double>(2));
//	for (int i = 0; i < 6; i++) {
//		for (int j = 0; j < 2; j++) {
//			double tmp;
//			cin >> tmp;
//			f[i][j] = tmp;
//		}
//	}
//	vector <vector<double>> d = get_batch(f, 1, 3);*/
//	vector<double> res = divided_by_1k(a);
//	int ttttt = 0;
//	return 0;
//}