#include<vector>
#include<string>
#include<iostream>
#include<assert.h>
#include<cmath>
#include<random>
#include<float.h>

using namespace std;


// ���ڼ������˷������ú���ֻ�ܼ���n*m��m*1ά�ľ������
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


// ���ڼ������˷������ú���ֻ�ܼ���n*1��1*mά�ľ������
vector<vector<double>> matrix_multi(vector<double> v1, vector<double> v2) {
	vector<vector<double>>res(v1.size(), vector<double>(v2.size()));
	for (int i = 0; i < res.size(); i++) {
		for (int j = 0; j < res[i].size(); j++) {
			res[i][j] = v1[i] * v2[j];
		}
	}
	return res;
}


// ���ڽ�һ��n*mά�ľ���ת��Ϊһ��m*nά�ľ���
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


// ���ڶ�w����Xavier initialization
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


// ���ڶ�b����Xavier initialization
vector<double> Xavier_initialization(int n) {
	assert(n > 0);
	vector<double> res(n);
	for (int i = 0; i < n; i++) {
		res[i] = 0;
	}
	return res;
}


// ���ڶ�w����kaiming initialization
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


// ���ڶ�b����kaiming initialization
vector<double> Kaiming_initialization(int n) {
	assert(n > 0);
	vector<double>res(n);
	for (int i = 0; i < n; i++) {
		res[i] = 0;
	}
	return res;
}


// ���ھ���ӷ������ú���ֻ�ܼ���n*1��n*1ά�ľ������
vector<double> matrix_sum(vector<double> wx, vector<double> b) {
	assert(wx.size() == b.size());
	vector<double> res(wx.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = wx[i] + b[i];
	}
	return res;
}


// ����ʵ��sigmoid�������ú�������һ��n*1ά����e��Ϊ����������һ���µ�����res
// ����res��Ԫ����e�ж�Ӧλ�õ�Ԫ��sigmoid��Ľ��
vector<double> sigmoid(vector<double> e) {
	vector<double> res(e.size());
	for (int i = 0; i < e.size(); i++) {
		double v = 1.0 / (1 + exp(-e[i]));
		res[i] = v;
	}
	return res;
 }


// ����ʵ��relu�������ú�������һ��n*1ά����e��Ϊ����������һ���µ�����res
// ����res��Ԫ����e�ж�Ӧλ�õ�Ԫ��relu��Ľ��
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


// ����ʵ��softmax�������ú�������һ��n*1ά����e��Ϊ����������һ���µ�����res
// ����res=softmax(e)
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


// �����ж�һ��n*1ά���� e �е�ÿ��Ԫ���Ƿ����0������һ��������� res
// ��� e[i]>0 �� res[i] = 1������ res[i] = 0
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


// ����һ������ e ����һ������res������ res[i] = e[i] / 1000
vector<double> divided_by_1k(vector<double> e) {
	vector<double> res(e.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = (e[i] * 1.0) / 1000;
	}
	return res;
}


// ���ڼ��㽻������ʧ��������������ͬ�ȴ�С��������Ϊ�������������ǵĽ�����
double cross_entropy(vector<double> y, vector<double> a) {
	assert(y.size() == a.size());
	double sum = 0;
	for (int i = 0; i < y.size(); i++) {
		sum += (y[i] * log(a[i]));
	}
	return -1 * sum;
}


// ���ڼ�����������֮�����������������output��y������output��y��Ӧλ��֮��
vector<double> matrix_sub(vector<double> output, vector<double> y) {
	assert(output.size() == y.size());
	vector<double> res(output.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = output[i] - y[i];
	}
	return res;
}


// ���ڼ�����������֮�����������������a��b������a��b��Ӧλ��֮��
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


// ���ڼ������������hadamard�������ú���ֻ�ܼ���n*1ά�ľ�����hadamard��
vector<double> matrix_hadamard_product(vector<double> m1, vector<double> m2) {
	assert(m1.size() == m2.size());
	vector<double> res(m1.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = m1[i] * m2[i];
	}
	return res;
}


// ���ڼ���һ�������һ�������ĳ˻�����������һ��n*mά����һ�����������س˻����
vector<vector<double>> matrix_scalar_multi(vector<vector<double>> m, double a) {
	vector<vector<double>> res(m.size(), vector<double>(m[0].size()));
	for (int i = 0; i < m.size(); i++) {
		for (int j = 0; j < m[i].size(); j++) {
			res[i][j] = m[i][j] * a;
		}
	}
	return res;
}


// ���ڼ���һ��������һ�������ĳ˻�����������һ��n*1ά������һ�����������س˻����
vector<double> matrix_scalar_multi(vector<double> v, double a) {
	vector<double> res(v.size());
	for (int i = 0; i < res.size(); i++) {
		res[i] = v[i] * a;
	}
	return res;
}


// ���ڽ�����������У�����m��ʾһ��n*mά���󣬲��� i �� k ��ʾ����m1�ĵ� i �к͵� k ��
void matrix_swap(vector<vector<double>>&m, int i, int k) {
	assert(i >= 0 && i < m.size() && k >= 0 && k < m.size());
	vector<double> tmp = m[k];
	m[k] = m[i];
	m[i] = tmp;
}


// ���ڶ�������������������е�˳�򣬽�������n*mά�ľ���m1��m2��Ϊ�������޷���ֵ
// ��Ҫ��֤����a1��a2�ֱ���m1��m2�еĵ� i �У����������������е�˳���a1��a2���붼��m1��m2�еĵ� k ��
void shuffle(vector<vector<double>>&m1, vector<vector<double>>&m2) {
	assert(m1.size() == m2.size());
	int n = m1.size();
	// used���ڼ�¼һ���Ƿ��Ѿ��������������� i �л�δ����������used[i]=0����֮used[i]=1
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
			// �� i �к͵� k �н��� 
			int k = int_dist(generator);
			used[i] = 1;
			used[k] = 1;
			matrix_swap(m1, i, k);
			matrix_swap(m2, i, k);
		}
	}
}


// ���ڻ�ȡ���ݼ�X_train���� i Ϊ��㣬����Ϊbatch_size�������ݼ�
// ���X_train��ʣ�������������batch_size����ֱ�ӷ���X_train������ʣ�������
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


// ���ڻ�ȡһ�����������Ԫ�ص��±꣬�����������ж����ȵ����ֵ���򷵻ص�һ�����ֵ���±�
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