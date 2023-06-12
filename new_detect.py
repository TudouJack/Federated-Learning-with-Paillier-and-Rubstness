import torch
import time
from sklearn.cluster import DBSCAN
import sys
from k_means_constrained import KMeansConstrained


def l1_operate(params):
    l1_list = []
    for x in params:
        l1_list.append(torch.norm(x, 1))

    # L1范数归一化
    l1_sum = sum(l1_list)
    for k in range(len(l1_list)):
        l1_list[k] = l1_list[k] / l1_sum

    return l1_list


def distance_compute(local_params, n_params):
    print("各个参与方计算其余参与方的聚合结果")
    t1 = time.time()
    others_params = []
    t_list = []
    weight_list = []
    for i in range(len(local_params)):
        temp = []
        weight_list.append(l1_operate(local_params[i]))
        for p in range(len(local_params[0])):
            local_params[i][p].requires_grad_(False)
            temp.append(local_params[i][p].cpu().numpy())
        t_list.append(temp)
    t_a_list = []

    for p in range(len(local_params[0])):
        t_a_list.append(n_params[p].cpu().numpy())
    for remote_index in range(len(local_params)):
        one_worker_params = []
        for tensor in range(len(local_params[0])):
            one_worker_params.append(
                ( len(local_params) * t_a_list[tensor] - t_list[remote_index][tensor]) * (1 / (len(local_params) - 1)) )
        others_params.append(one_worker_params)

    t1_f = time.time()
    # file.write("其余参与方聚合结果计算完成，总耗时 %.8f 秒，每个参与方计算耗时 %.8f秒\n" % (t1_f - t1, (t1_f - t1) / len(local_params)))
    print("其余参与方聚合结果计算完成，总耗时 %.4f 秒，每个参与方计算耗时 %.4f秒" % (t1_f - t1, (t1_f - t1) / len(local_params)))

    # 参与方向服务器上传计算结果

    print("服务器计算模型参数的距离向量")
    for remote_index in range(len(local_params)):
        for tensor in range(len(local_params[0])):
            others_params[remote_index][tensor] = torch.from_numpy(others_params[remote_index][tensor]).cuda()
            local_params[remote_index][tensor].requires_grad_()

    distances = []

    for tensor in range(len(local_params[0])):
        distemp = []
        for remote_index in range(len(local_params)):
            distance = torch.dist(others_params[remote_index][tensor], local_params[remote_index][tensor], p=2)
            distemp.append(distance.item())
        distances.append(torch.tensor(distemp))
    storage = sys.getsizeof(distances[0][0]) + sys.getsizeof(distances[1][0])
    t2 = time.time()
    # file.write("距离向量计算完成，总耗时 %.8f 秒，每个参与方耗时 %.8f秒\n" % (t2 - t1_f, (t2 - t1_f) / len(local_params)))
    print("距离向量计算完成，总耗时 %.4f 秒，每个参与方耗时 %.4f秒" % (t2 - t1_f, (t2 - t1_f) / len(local_params)))
    # file.write("上传的距离向量的数据大小：%d Byte\n" % storage)
    print("上传的距离向量的数据大小：%d Byte" % storage)  # 72Byte * 2
    # print(distances)
    return distances, weight_list


def detect_3sig(distances, result, p):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    print("运行基于sig原则的检测算法")
    # file.write("运行基于sig原则的检测算法\n")
    num_t, num_f = 0, 0
    t = time.time()
    for i in range(len(distances)):
        print("参数 %d 统计信息：" % (i + 1))
        print("①距离向量：", distances[i])
        avg_distance = torch.mean(distances[i])
        print("②平均距离：%.4f" % avg_distance.item())
        sig = torch.std(distances[i])
        print("③标准差：%.4f" % sig.item())
        left = avg_distance - p * sig
        right = avg_distance + p * sig
        print("④置信区间：[%.2f, %.2f]" % (left.item(), right.item()))
        for j in range(len(distances[i])):
            if abs(distances[i][j] - avg_distance) < p * sig:
                result[j] = result[j] and True
            else:
                result[j] = result[j] and False
    t_f = time.time()
    # file.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    print("异常检测总运行时间： %.4f秒， 每个参与方用时： %.4f秒" % (t_f - t, (t_f - t) / len(distances[0])))
    # file.write(str(result))

    print("True: ", result.count(True), "False: ", result.count(False))


def detect_dbscan(distances, result, p):
    if len(distances[0]) != len(result):
        print("错误，长度不等")

    # file.write("运行基于聚类的检测算法\n")
    print("运行基于聚类的检测算法")
    num_t, num_f = 0, 0
    t = time.time()
    for i in range(len(distances)):
        print("参数 %d 统计信息：" % (i + 1))
        print("①距离向量：", distances[i])
        avg_distance = torch.mean(distances[i])
        print("②平均距离：%.4f" % avg_distance.item())
        dis_x = distances[i][:].numpy()
        dis_y = dis_x.reshape(-1, 1)
        db = DBSCAN(eps=p * avg_distance.item(), min_samples=len(dis_x) * 0.5).fit(dis_y)
        temp = list(db.labels_)
        for j in range(len(temp)):
            if temp[j] == -1:
                result[j] = result[j] and False
            else:
                result[j] = result[j] and True
    t_f = time.time()
    # file.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    print("异常检测总运行时间： %.4f秒， 每个参与方用时： %.4f秒" % (t_f - t, (t_f - t) / len(distances[0])))
    # file.write(str(result))
    print("True: ", result.count(True), "False: ", result.count(False))


def weight_m_model_create(param, m_nodes):
    left, right = -0.5, 0.5  #一个攻击者，3.0, 89%, 4.0 11%
    model_t = []
    weight_list = l1_operate(param)
    max_value = max(weight_list)  # 求列表最大值
    max_idx = weight_list.index(max_value)  # 求最大值对应索引
    print("参数L1范数最大index：", max_idx)
    for i in range(m_nodes):
        p_t = []
        for p in range(len(param)):
            if p == max_idx:
                t = torch.empty_like(param[p])
                torch.nn.init.uniform_(t, a=left, b=right)
                p_t.append(t.requires_grad_())
            else:
                t = param[p].clone().detach()
                p_t.append(t.requires_grad_())
        model_t.append(p_t)
    print("特定参数随机异常模型构造完成，异常模型参数范围：(%.2f, %.2f)" % (left, right))
    return model_t


def m_model_create(param, m_nodes):
    left, right = -1.0, 1.0  #一个攻 击者，3.0, 89%, 4.0 11%
    model_t = []
    for i in range(m_nodes):
        p_t = []
        for p in param:
            t = torch.empty_like(p)
            torch.nn.init.uniform_(t, a=left, b=right)
            p_t.append(t.requires_grad_())
        model_t.append(p_t)
    print("全参数随机异常模型构造完成，异常模型参数范围：(%.2f, %.2f)" % (left, right))
    return model_t


def weight_detect_dbscan(distances, result, p, weight):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    print("运行基于聚类的检测算法（带权）")
    # file.write("运行基于sig原则的检测算法\n")
    num_t, num_f = 0, 0
    t = time.time()
    weight_distance = torch.zeros(len(result))
    for i in range(len(result)):
        for j in range(len(distances)):
            temp_d = distances[j][i].detach() * weight[i][j].detach()
            weight_distance[i] += temp_d

    dis_x = weight_distance[:].numpy()
    dis_y = dis_x.reshape(-1, 1)
    avg_distance = torch.mean(weight_distance)
    print("距离向量：", weight_distance)
    print("平均距离：%.4f" % avg_distance.item())
    db = DBSCAN(eps=p * avg_distance.item(), min_samples=len(dis_x) * 0.5).fit(dis_y)
    temp = list(db.labels_)
    for j in range(len(temp)):
        if temp[j] == -1:
            result[j] = result[j] and False
        else:
            result[j] = result[j] and True
    t_f = time.time()
    # file.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    print("异常检测总运行时间： %.4f秒" % (t_f - t))
    # file.write(str(result))
    print("True: ", result.count(True), "False: ", result.count(False))


def weight_detect_3sig(distances, result, p, weight):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    # file.write("运行基于聚类的检测算法\n")
    print("运行基于sig原则的检测算法（带权）")
    num_t, num_f = 0, 0
    t = time.time()
    weight_distance = torch.zeros(len(result))
    for i in range(len(result)):
        for j in range(len(distances)):
            temp_d = distances[j][i].detach() * weight[i][j].detach()
            weight_distance[i] += temp_d
    avg_distance = torch.mean(weight_distance)
    sig = torch.std(weight_distance)
    print("标准差：%.4f" % sig.item())
    left = avg_distance - p * sig
    right = avg_distance + p * sig
    print("置信区间：[%.2f, %.2f]" % (left.item(), right.item()))
    for j in range(len(weight_distance)):
        if abs(weight_distance[j] - avg_distance) < p * sig:
            result[j] = result[j] and True
        else:
            result[j] = result[j] and False
    t_f = time.time()
    # file.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    print("异常检测总运行时间： %.4f秒，" % (t_f - t))
    # file.write(str(result))
    print("True: ", result.count(True), "False: ", result.count(False))


def weight_constrained_k_means(distances, result, weight):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    # file.write("运行基于聚类的检测算法\n")
    print("运行基于约束k-means原则的检测算法（带权）")
    t_s = time.time()
    weight_distance = torch.zeros((len(result), 1))
    for i in range(len(result)):
        for j in range(len(distances)):
            temp_d = distances[j][i].detach() * weight[i][j].detach()
            weight_distance[i][0] += temp_d
    clf = KMeansConstrained(n_clusters=2, size_min=1)  # 构造聚合模型
    predict = clf.fit_predict(weight_distance)  # 分类
    list_predict = list(predict)
    # print(list_predict)
    if list_predict.count(1) < list_predict.count(0):  # 这里是由于分类结果1还是0不确定，结合True肯定比False多，做此变化
        for i in range(len(list_predict)):
            list_predict[i] = abs(list_predict[i] - 1)
    # print(list_predict)
    for j in range((len(result))):  # 预测结果
        result[j] = result[j] and list_predict[j]
    t_f = time.time()
    # file.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    print("异常检测总运行时间： %.4f秒，" % (t_f - t_s))
    # file.write(str(result))
    print("True: ", result.count(True), "False: ", result.count(False))
