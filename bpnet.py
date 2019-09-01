import math
import random
#反向神经网络
#
'''
该函数既可以读入数据，也可以读入结果
读入训练/测试数据或结果，返回数组
'''
def readdata_(filename):
	data = []
	with open(filename,'r') as f:
		line = f.readline()
		while line:
			eachline = line.split()
			read_data = [x for x in eachline[:]]
			data.append(read_data)
			line = f.readline()
	return data

random.seed(0)

'''
产生随机数
'''
def rand(lownum,highnum):
	return (b-a)*random.random()+a

''' 
激励函数选择 sigmoid
'''
def sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

'''
构造权值矩阵
'''
def makeweights(x,y,fill = 0.0):
	return [[fill]*x for _ in range(y)]

'''
计算两个一维列表对应元素的乘积
'''
def multi_2list(l1,l2,bias):
	len_l = len(l1)
	mul = 0
	for i in range(len_l):
		mul = mul + l1[i] * l2[i]
	mul = mul + bias
    return mul


'''
计算两个一维列表的点乘
'''
def one_list_dotmul(l1,l2):
	res = []
	for i in range(len(l1)):
		res.append(l1[i]*l2[i])
	return res

'''
计算两个一维列表的矩阵乘
'''
def one_list_matmul(l1,l2):
	res = []
	for i in range(len(l1)):
		row = []
		for j in range(len(l2)):
			row.append(l1[i] * l2[j])
		res.append(row)
	return res


'''
计算w*x + b，并存储sigmoid结果
'''
def W_mul_x(weight,traindatarow,bias):
	I = len(weight)
	J = len(weight[0])
	a = []
	for i in range(I):
		res = multi_2list(weight[i],traindatarow,bias[i])
		sig_res = sigmoid(res)
		a.append(sig_res)
	return a

'''
计算两个二维列表相加
'''
def two_mat_add(mat1,mat2):
	a = len(mat1)
	b = len(mat1[0])
	res = []
	for i in range(a):
		row = []
		for j in range(b):
			row.append(mat1[i][j] + mat2[i][j])
		res.append(row)
	return res

'''
计算两个二维列表相减
'''
def two_mat_minus(mat1,mat2):
	a = len(mat1)
	b = len(mat1[0])
	res = []
	for i in range(a):
		row = []
		for j in range(b):
			row.append(mat1[i][j] - mat2[i][j])
		res.append(row)
	return res

'''
数乘二维列表
'''
def num_mul_mat(num,l):
	a = len(l)
	b = len(l[0])
	res = []
	for i in range(a):
		row = []
		for j in range(b):
			row.append(l[i][j] * num)
		res.append(row)
	return res

'''
返回列表
'''
def G(a):
	l = []
	for i in a:
		l.append(i * (1 - i))
	return l

'''
二维列表转置
'''
def transpose_list(l):
	a = len(l)
	b = len(l[0])
	row = []
	l_n = []
	for i in range(b):
		for j in range(a):
			row.append(l[j][i])
		l_n.append(row)
		row = []
	return l_n

'''
计算W*error
'''
def W_error(weight,error):
	a = len(weight)
	b = len(weight[0])
	res = []
	for i in range(a):
		mul = 0
		for j in range(b):
			mul = mul + weight[i][j] * error[j]
		res.append(mul)
	return res


class BPNN:
	def _init_(self,layer_num,everylayer_cell_num):   # layer_num->神经网络层数  everylayer_cell_num->列表，每层的神经元数目
		self.weight = {}              #权重,下标从0开始
		self.bias = {}                #偏差，下标从0开始
		self.error = {}               #误差
		self.delta = {}       
		self.a = {}         
		self.D = {}
 		self.d = {}
		self.layer_num = layer_num
		self.everylayer_cell_num = everylayer_cell_num   #列表
		self.firstbpturn = 1          #第一次实施反向传播，应该把delta置0
 		for i in range(layer_num - 1):
 			self.weight[i] = makeweights(everylayer_cell_num[i],everylayer_cell_num[i+1],rand(-1,1))
 			self.bias[i] = [rand(-0.2,0.2) for _ in range(everylayer_cell_num[i+1])]
 	
 	def propagation(self,traindatarow,y):       #前向传播，y为训练集的结果
 		self.a = {}                                #存储经过sigmoid后的结果a
 		sig_res_list = []
 		for i in range(self.layer_num):          
 			if i == 0:                        #第一层神经网络，利用输入数据计算结果
 				self.a[i] = traindatarow           # a[0] = x
 				sig_res_list = W_mul_x(self.weight[i],traindatarow,bias[i])      #经过sigmoid之后的结果列表
 			else:                             #二以上层的神经网络，利用a[]作为输入数据
 				self.a[i] = sig_res_list
 				sig_res_list = W_mul_x(self.weight[i],sig_res_list,bias[i])

 		self.error[layer_num - 1] = [sig_res_list[j] - y[j] for j in range(len(y))]                #神经网络输出结果与训练集结果的误差
 		for i in range(layer_num-2,0,-1):
 			w = transpose_list(self.weight(i))
 			a_g = G(self.a[i])
 			w_e = W_error(w,self.error[i+1])
 			self.error[i] = one_list_dotmul(a_g,w_e)

 	def backprop(self,lanbda,m):          #lanbda->正则化系数，epsilon->学习速率
 		self.D = {}
 		self.d = {}
 		if self.firstbpturn is 1:               #第一次实施bp，置delta为0
 			self.firstbpturn = 0
 			for i in range(self.layer_num-1,0,-1):  #以error上标为准
 				self.delta[i-1] = [[0.0] * self.everylayer_cell_num[i-1] for _ in range(self.everylayer_cell_num[i])]
 		else:
 			for i in range(self.layer_num-1,0,-1):  #以error上标为准
 				a_error = one_list_matmul(self.error[i],self.a[i-1])
 				temp1 = two_mat_add(a_error,self.delta[i-1])
 				self.delta[i-1] = temp1
 				#计算D
 				m_1_delta = num_mul_mat((1.0 / m),self.delta[i-1])
 				m_lanbda_weight = num_mul_mat(float(lanbda / m),self.weight[i-1])
 				self.D[i-1] = two_mat_add(m_1_delta,m_lanbda_weight)
 				self.d[i-1] = m_1_delta

 	def train(self,lanbda,epsilon,traintimes,layer_num):   #每次从训练数据中读入一行进行训练
 		#读入训练数据及结果
 		traindata = readdata_('train.txt')
 		trainres = readdata_('result.txt')
 		len_traindata = len(traindata)
 		for t in traintimes:
 			for round_ in range(len_traindata):       #round_->样本批数
 				propagation(traindata[round_],trainres[round_])
 				backprop(lanbda,len_traindata)
 				for i in range(layer_num-1)
 					tempD = num_mul_mat(epsilon,self.D[i])
 					self.weight[i] = two_mat_minus(self.weight[i],tempD)
 					tempd = num_mul_mat(epsilon,self.d[i])
 					self.bias = num_mul_mat(epsilon,self.d[i])

