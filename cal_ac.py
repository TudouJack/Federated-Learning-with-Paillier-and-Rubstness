import sys
import numpy as np
from syft.frameworks.torch.tensors.interpreters.paillier import PaillierTensor
from phe.paillier import EncryptedNumber
import torch

# 自编码器输入、输出都是模型
# input:(1,784),输出predict:(1,784)
# 现在的模型为(784,10)，所以一个模型输入10次，还原为原来的模型，并且计算err

# 计算模型压缩重构错误率,返回一个浮点数

def get_err(input_mat):
    # 重载自编码器模型
    models = []
    for i in range(len(input_mat)):
        models.append(torch.load('ae_param_%d.pth' % i))

    # 自编码器运行
    err_l = []

    for i in range(len(input_mat)):
        deep_cop = np.array(list(input_mat[i]))
        model_slice = deep_cop.ravel()
        print(sys.getsizeof(model_slice[0].ciphertext()))
        print(model_slice[0].ciphertext())
        for j in range(len(model_slice)):
            model_slice[j] = np.float64(model_slice[j].ciphertext())
        extr = torch.from_numpy(deep_cop.astype(np.float32))
        pre_e, pre_d = models[i](extr)
        err = torch.norm(pre_d-extr, 2)
        err_l.append(err.detach())
    # 两矩阵对应位置差值的平方矩阵
    return np.mean(np.array(err_l))


# 计算异常度，返回一个np数组
def get_abnormal_score(err_list):
    err_list_np = np.array(err_list)
    min_err = np.min(err_list_np)

    abnormal_score = []
    for i in range(len(err_list_np)):
        # 计算单个异常度
        an = (1 + err_list_np[i]) / (1 + min_err)
        abnormal_score.append(an)

    return np.array(abnormal_score)


# 返回异常参与方np数组，因为参与方进程号表示，所以需要家加上1
# 形参：异常度np数组

def get_abnormal_list(abnormal_score, result):
    threshold = np.mean(abnormal_score)
    abnormal_list = []

    # 测试阶段，不判断是否异常，输出所有异常度np数组
    for i in range(len(abnormal_score)):
        flag = abnormal_score[i] - threshold
        if flag > 0.5:  # 根据数据来看，异常值基本位于1.0-1.1之间
            # abnormal_list.append(i + 1)
            result[i] = False
    #
    # print(abnormal_list)
    return np.array(abnormal_list)


# 计算信任度，返回一个np数组
# 形参：训练集长度列表，超参数，异常度列表
def get_credit_score(L, abnormal_score):
    length = len(abnormal_score)
    credit_score = []
    son = []
    mon = 0
    for i in range(length):
        # 计算分子
        son_sin = 1 * pow(abnormal_score[i], -L)
        # 收集分子数组
        son.append(son_sin)
        # 累加计算分母
        mon = mon + son_sin
    for i in range(length):
        # 计算信用度并加入数组
        credit_score.append(son[i] / mon)

    return np.array(credit_score)


def new_get_credit_score(L, abnormal_score, result):
    length = len(abnormal_score)
    credit_score = []
    son = []
    mon = 0
    for i in range(length):
        # 计算分子
        son_sin = 1 * pow(abnormal_score[i], -L)
        # 收集分子数组
        son.append(son_sin)
        # 累加计算分母
        mon = mon + son_sin
    for i in range(length):
        # 计算信用度并加入数组
        credit_score.append(son[i] / mon)
    return np.array(credit_score)

