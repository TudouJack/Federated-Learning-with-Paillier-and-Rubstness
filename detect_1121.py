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


def new_weight(params, local_params):
    weight_list = []
    for i in range(len(local_params)):
        weight_list.append(l1_operate(params))
    return weight_list


def distance_compute(local_params, new_params, m_nodes, percent=0.2):
    # print("各个参与方计算其余参与方的聚合结果")
    # t1 = time.time()
    others_params = []
    t_list = []
    weight_list = []
    for i in range(len(local_params)):   # 按照节点遍历
        temp = []
        weight_list.append(l1_operate(local_params[i]))   # 返回的是各节点的一个列表该列表中含各层的L1范数
        # 即weight_list是一个二维向量，第一维是不同节点，第二维是不同层
        for p in range(len(local_params[0])):
            local_params[i][p].requires_grad_(False)
            temp.append(local_params[i][p].cpu().numpy())
        t_list.append(temp)
    t_a_list = []

    for p in range(len(local_params[0])):
        t_a_list.append(new_params[p].cpu().numpy())
    for remote_index in range(len(local_params)):
        one_worker_params = []
        for tensor in range(len(local_params[0])):
            one_worker_params.append(
                ( len(local_params) * t_a_list[tensor] - t_list[remote_index][tensor]) * (1 / (len(local_params) - 1)) )   # 其余参与方聚合结果
        others_params.append(one_worker_params)

    # t1_f = time.time()
    # log_f.write("其余参与方聚合结果计算完成，总耗时 %.8f 秒，每个参与方计算耗时 %.8f秒\n" % (t1_f - t1, (t1_f - t1) / len(local_params)))
    # print("其余参与方聚合结果计算完成，总耗时 %.4f 秒，每个参与方计算耗时 %.4f秒" % (t1_f - t1, (t1_f - t1) / len(local_params)))

    # 参与方向服务器上传计算结果

    # print("客户端计算模型参数的距离向量")
    for remote_index in range(len(local_params)):
        for tensor in range(len(local_params[0])):
            others_params[remote_index][tensor] = torch.from_numpy(others_params[remote_index][tensor]).cuda()
            local_params[remote_index][tensor].requires_grad_()

    distances = []

    for tensor in range(len(local_params[0])):   # 各层进行遍历
        distemp = []   # 各节点
        for remote_index in range(len(local_params)):   # 各节点进行遍历
            distance = torch.dist(others_params[remote_index][tensor], local_params[remote_index][tensor], p=2)   # p2距离计算函数
            distemp.append(distance.item())   # 将distance张量转化成浮点数
        distances.append(torch.tensor(distemp))   # 第一维度是层，第二维度是节点

    ##############################新攻击方式###########################
    new_weight_list = l1_operate(new_params)
    weight_dict = dict()
    for k, v in enumerate(new_weight_list):
        weight_dict[k] = v
    weight_dict = sorted(weight_dict.items(), key=lambda s: s[1], reverse=True)
    print(weight_dict)
    for i in range(int(len(weight_dict) * percent)):
        for j in range(len(local_params) - m_nodes, len(local_params)):
            distances[weight_dict[i][0]][j] = 0

    ##############################新攻击方式###########################

    return distances, weight_list


def detect_3sig(distances, result, p, log_f):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    log_f.write("运行detect_3sig检测算法\n")
    print("运行detect_3sig检测算法")
    num_t, num_f = 0, 0
    t = time.time()
    for i in range(len(distances)):
        # print("参数 %d 统计信息：" % (i + 1))
        # print("①距离向量：", distances[i])
        avg_distance = torch.mean(distances[i])
        # print("②平均距离：%.4f" % avg_distance.item())
        sig = torch.std(distances[i])
        # print("③标准差：%.4f" % sig.item())
        left = avg_distance - p * sig
        right = avg_distance + p * sig
        # print("④置信区间：[%.2f, %.2f]" % (left.item(), right.item()))
        for j in range(len(distances[i])):
            if abs(distances[i][j] - avg_distance) < p * sig:
                result[j] = result[j] and True
            else:
                result[j] = result[j] and False
    t_f = time.time()
    # log_f.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    # print("异常检测总运行时间： %.4f秒， 每个参与方用时： %.4f秒" % (t_f - t, (t_f - t) / len(distances[0])))
    # log_f.write(str(result))
    log_f.write("各参与方检测结果: {}\n".format(str(result)))
    # print("True: ", result.count(True), "False: ", result.count(False))


def new_detect_dbscan(distances, result, p, log_f):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    log_f.write("运行detect_dbscan检测算法\n")
    print("运行weight_detect_dbscan检测算法")
    # num_t, num_f = 0, 0
    # t = time.time()
    weight_distance = torch.zeros(len(result))
    for i in range(len(result)):
        for j in range(len(distances)):
            temp_d = distances[j][i].detach()
            weight_distance[i] += temp_d
    print("distance:", weight_distance)
    log_f.write(str(weight_distance))
    dis_x = weight_distance[:].numpy()
    dis_y = dis_x.reshape(-1, 1)
    avg_distance = torch.mean(weight_distance)
    # print("距离向量：", weight_distance)
    # print("平均距离：%.4f" % avg_distance.item())
    db = DBSCAN(eps=p * avg_distance.item(), min_samples=len(dis_x) * 0.5).fit(dis_y)
    temp = list(db.labels_)
    for j in range(len(temp)):
        if temp[j] == -1:
            result[j] = result[j] and False
        else:
            result[j] = result[j] and True
    # t_f = time.time()
    # log_f.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    # print("异常检测总运行时间： %.4f秒" % (t_f - t))
    log_f.write("各参与方检测结果: {}\n".format(str(result)))
    # print("True: ", result.count(True), "False: ", result.count(False))


def detect_dbscan(distances, result, p, log_f):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    log_f.write("运行detect_dbscan检测算法\n")
    print("运行detect_dbscan检测算法")
    num_t, num_f = 0, 0
    t = time.time()
    for i in range(len(distances)):
        # print("参数 %d 统计信息：" % (i + 1))
        # print("①距离向量：", distances[i])
        avg_distance = torch.mean(distances[i])
        # print("②平均距离：%.4f" % avg_distance.item())
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
    # log_f.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    # print("异常检测总运行时间： %.4f秒， 每个参与方用时： %.4f秒" % (t_f - t, (t_f - t) / len(distances[0])))
    # log_f.write(str(result))
    log_f.write("各参与方检测结果: {}\n".format(str(result)))
    # print("True: ", result.count(True), "False: ", result.count(False))


def constrained_k_means(distances, result, p, log_f):
    # print("distances:")
    # print(distances)
    # print("weight:")
    # print(weight)
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    log_f.write("运行constrained_k_means检测算法\n")
    print("运行constrained_k_means检测算法")
    # t_s = time.time()
    weight_distance = torch.zeros((len(result), 1))
    for i in range (len(result)):
        for j in range(len(distances)):
            temp_d = distances[j][i].detach()
            weight_distance[i][0] += temp_d
    print("distance:", weight_distance)
    log_f.write(str(weight_distance))
    clf = KMeansConstrained(n_clusters=3, size_min=1,tol=0.1)    # 构造聚合模型
    predict = clf.fit_predict(weight_distance)    # 分类
    list_predict = list(predict)
    print(list_predict)
    
    # if list_predict.count(1) < list_predict.count(0):    # 这里是由于分类结果1还是0不确定，结合True肯定比False多，做此变化
    #     for i in range(len(list_predict)):
    #         list_predict[i] = abs(list_predict[i]-1)
    n = int(len(list_predict) * p)  # 3
    p_0 = list_predict.count(0)
    p_1 = list_predict.count(1)
    p_2 = len(list_predict) - p_1 - p_0
    print(p_0, p_1, p_2)
    print(n)
    if p_0 > n:
        for i in range(len(list_predict)):
            if list_predict[i] == 0:
                list_predict[i] = 3

    if p_1 <= n:
        for i in range(len(list_predict)):
            if list_predict[i] == 1:
                list_predict[i] = 0

    if p_2 <= n:
        for i in range(len(list_predict)):
            if list_predict[i] == 2:
                list_predict[i] = 0

    print(list_predict)
    for j in range((len(result))):    # 预测结果
        result[j] = result[j] and bool(list_predict[j])
    print(result)
    # t_f = time.time()
    # log_f.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t_s, (t_f - t_s) / len(distances[0])))
    # print("异常检测总运行时间： %.4f秒，" % (t_f - t_s))
    log_f.write("各参与方检测结果: {}\n".format(str(result)))
    # print("True: ", result.count(True), "False: ", result.count(False))


def weight_m_model_create(param, m_nodes, log_f, epoch, percent=0.2):
    # print(epoch)
    log_f.write("运行weight_m_model_create恶意方生成算法\n")
    print("运行weight_m_model_create恶意方生成算法")
    left, right = -1.0, 1.0  #一个攻击者，3.0, 89%, 4.0 11%
    model_t = []
    weight_list = l1_operate(param)
    # print("全局模型weight:", weight_list)
    weight_dict = dict()
    for k, v in enumerate(weight_list):
        weight_dict[k] = v
    weight_dict = sorted(weight_dict.items(), key=lambda s: s[1], reverse=True)
    max_list = []
    for i in range(int(len(weight_dict) * percent)):
        max_list.append(weight_dict[i][0])

    if epoch == 0:
        for i in range(m_nodes):
            p_t = []        
            # 恶意参与方全随机
            for p in param:
                t = torch.empty_like(p)
                torch.nn.init.uniform_(t, a=left, b=right)
                p_t.append(t.requires_grad_())
            # print(p_t)
            model_t.append(p_t)
    else:
        for i in range(m_nodes):
            p_t = []
            for p in range(len(param)):
                if p in max_list:
                    t = torch.empty_like(param[p])
                    torch.nn.init.uniform_(t, a=left, b=right)
                    p_t.append(t.requires_grad_())
                else:
                    t = param[p].detach().clone()
                    p_t.append(t.requires_grad_())
            model_t.append(p_t)
    print("随机异常模型构造完成，异常模型参数范围：(%.2f, %.2f)" % (left, right))
    return model_t


def m_model_create(param, m_nodes, log_f):
    left, right = -1.0, 1.0  #一个攻击者，3.0, 89%, 4.0 11%
    model_t = []
    log_f.write("运行m_model_create恶意方生成算法\n")
    print("运行m_model_create恶意方生成算法")
    for i in range(m_nodes):
        p_t = []        
        # 恶意参与方全随机
        for p in param:
            t = torch.empty_like(p)
            torch.nn.init.uniform_(t, a=left, b=right)
            p_t.append(t.requires_grad_())
            
        # print(p_t)
        model_t.append(p_t)
    # log_f.write("随机异常模型构造完成，异常模型参数范围：(%.2f, %.2f)" % (left, right))
    # print("随机异常模型构造完成，异常模型参数范围：(%.2f, %.2f)" % (left, right))
    return model_t


def weight_detect_dbscan(distances, result, p, weight, log_f):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    log_f.write("运行weight_detect_dbscan检测算法\n")
    print("运行weight_detect_dbscan检测算法")
    # num_t, num_f = 0, 0
    # t = time.time()
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
    # t_f = time.time()
    # log_f.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    # print("异常检测总运行时间： %.4f秒" % (t_f - t))
    print(result)
    log_f.write("各参与方检测结果: {}\n".format(str(result)))
    # print("True: ", result.count(True), "False: ", result.count(False))


def weight_detect_3sig(distances, result, p, weight, log_f):
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    log_f.write("运行weight_detect_3sig检测算法\n")
    print("运行weight_detect_3sig检测算法")
    # num_t, num_f = 0, 0
    # t = time.time()
    weight_distance = torch.zeros(len(result))
    for i in range(len(result)):
        for j in range(len(distances)):
            temp_d = distances[j][i].detach() * weight[i][j].detach()
            weight_distance[i] += temp_d
    avg_distance = torch.mean(weight_distance)
    sig = torch.std(weight_distance)
    # print("标准差：%.4f" % sig.item())
    left = avg_distance - p * sig
    right = avg_distance + p * sig
    # print("置信区间：[%.2f, %.2f]" % (left.item(), right.item()))
    for j in range(len(weight_distance)):
        if abs(weight_distance[j] - avg_distance) < p * sig:
            result[j] = result[j] and True
        else:
            result[j] = result[j] and False
    # t_f = time.time()
    # log_f.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t, (t_f - t) / len(distances[0])))
    # print("异常检测总运行时间： %.4f秒，" % (t_f - t))
    log_f.write("各参与方检测结果: {}\n".format(str(result)))
    # print("True: ", result.count(True), "False: ", result.count(False))


def weight_constrained_k_means(distances, result, p, weight, log_f):
    # print("distances:")
    # print(distances)
    # print("weight:")
    # print(weight)
    if len(distances[0]) != len(result):
        print("错误，长度不等")
    log_f.write("运行weight_constrained_k_means检测算法\n")
    print("运行weight_constrained_k_means检测算法")
    # t_s = time.time()
    weight_distance = torch.zeros((len(result), 1))
    for i in range (len(result)):
        for j in range(len(distances)):
            temp_d = distances[j][i].detach() * weight[i][j].detach()
            weight_distance[i][0] += temp_d
    print("weight_distance:", weight_distance)
    log_f.write(str(weight_distance))
    clf = KMeansConstrained(n_clusters=2, size_min=1, tol=0.1)    # 构造聚合模型
    predict = clf.fit_predict(weight_distance)    # 分类
    list_predict = list(predict)
    print(list_predict)

    # if list_predict.count(1) < list_predict.count(0):    # 这里是由于分类结果1还是0不确定，结合True肯定比False多，做此变化
    #     for i in range(len(list_predict)):
    #         list_predict[i] = abs(list_predict[i]-1)
    n = len(list_predict) * p
    p_0 = list_predict.count(0)
    p_1 = list_predict.count(1)
    p_2 = len(list_predict) - p_1 - p_0

    if p_0 > n:
        for i in range(len(list_predict)):
            if list_predict[i] == 0:
                list_predict[i] = 3

    if p_1 <= n:
        for i in range(len(list_predict)):
            if list_predict[i] == 1:
                list_predict[i] = 0

    if p_2 <= n:
        for i in range(len(list_predict)):
            if list_predict[i] == 2:
                list_predict[i] = 0

    print(list_predict)
    # print(list_predict)
    for j in range((len(result))):    # 预测结果
        result[j] = result[j] and bool(list_predict[j])
    print(result)
    # t_f = time.time()
    # log_f.write("异常检测总运行时间： %.8f秒， 每个参与方用时： %.8f秒\n" % (t_f - t_s, (t_f - t_s) / len(distances[0])))
    # print("异常检测总运行时间： %.4f秒，" % (t_f - t_s))
    log_f.write("各参与方检测结果: {}\n".format(str(result)))
    # print("True: ", result.count(True), "False: ", result.count(False))
