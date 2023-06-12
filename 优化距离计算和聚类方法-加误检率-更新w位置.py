import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from final_detect import distance_compute, weight_detect_3sig, m_model_create, weight_detect_dbscan, weight_constrained_k_means, detect_3sig, detect_dbscan, constrained_k_means, weight_m_model_create
from torchvision import datasets, transforms
import syft as sy
import time
import random
# import cal_ac
if __name__ == '__main__':



    class Arguments():    # 参数设置
        def __init__(self):
            self.batch_size = 512
            self.test_batch_size = 1024
            self.epochs = 30
            self.lr = 0.01
            self.log_interval = 10
            self.momentum = 0.5
            self.cuda = True


    args = Arguments()    # 实例化参数设置

    class Net(nn.Module):    # 定义网络，需要继承基类nn.Mouele
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1, bias=False)
            self.conv2 = nn.Conv2d(20, 50, 5, 1, bias=False)
            self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=False)
            self.fc2 = nn.Linear(500, 10, bias=False)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
        # train dataset 90%
        # 20epoch test dataset 91%
        # lr = 0.01 bs = 512
        # keylen = 256

    key_len = 256    # 秘钥长度
    print("秘钥长度", key_len)
    pub, pri = sy.keygen(n_length=key_len)  # 假设参与方和服务器已经实现了对密钥的生成和分发，其中服务器和参与方拥有公钥，参与方拥有彼此相同的私钥
    compute_nodes = []    # 计算节点
    result = []    # 结果列表（好坏模型的区分）
    m_nodes = 3   # m是坏模型数量
    n_nodes = 7    # n是好模型数量
    num_nodes = m_nodes + n_nodes  # 参与方数量需广播给各个参与方
    log_f = open("./log/(%d + %d)_%.1f .txt" % (n_nodes, m_nodes, m_nodes/num_nodes), "a")
    log_f.write(time.asctime(time.localtime()) + "\n")   # 记录时间

    # mal_model = []qjmxcs
    # m_model_create(mal_model, m_nodes, (10, 784), (1, 10))  生成异常模型
    models = []    # 模型列表
    optimizers = []    # 优化器列表
    # schedulers = []
    params = []    # 参数列表
    hook = sy.TorchHook(torch)    # 创建一个hook，每个hook可对应多个拥有不同id的VirtualWork
    device = torch.device("cuda" if args.cuda else "cpu")    # 使用 gpu or cpu
    print("系统信息: %d个参与方" % num_nodes)
    log_f.write("参与方数量: %d" % num_nodes + "\n")
    log_f.write("恶意参与方比例: {:.1f}".format(m_nodes/num_nodes) + "\n")
    
    # log_f.write("sigma原则的区间大小: σ" + "\n")
    log_f.write("DBSCAN的半径大小: d"+ "\n")
    # log_f.write("约束k-means分类阈值: 0.3"+ "\n")

    for i in range(n_nodes):  # 初始化好节点
        compute_nodes.append(sy.VirtualWorker(hook, id="client %d" % i))  # 增加worker
        models.append(Net().to(device))  # 给模型列表增加使用gpu的模型
        optimizers.append(optim.SGD(models[i].parameters(), lr=args.lr, momentum=args.momentum))  # 给优化器列表增加随机梯度下降优化器
        # schedulers.append(optim.lr_scheduler.ExponentialLR(optimizers[i], gamma=0.1, last_epoch=-1))
        params.append(list(models[i].parameters()))  # 给参数列表增加参数（只有参数值，没有对应的层名）
    model_shape = []    # 每个模型中各层参数的shape
    for p in params[0]:    # 初始化model_shape
        print(p.shape)
        model_shape.append(tuple(p.shape))

    split_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([  # 加载数据集并进行标准化
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 这是ImageNet数据集的方差和均值
    ]))
    federated_train_loader = sy.FederatedDataLoader(
        split_dataset.federate(compute_nodes), batch_size=args.batch_size, shuffle=True)  # 转化成联邦数据集dataloader形式
    test_dataset =  datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 这是ImageNet数据集的方差和均值
    ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)  # 转化成测试集dataloader形式

    remote_dataset = []   # 定义远程数据集
    for i in range(n_nodes):   # 处理的是好节点
        remote_dataset.append(list())
    remote_dataset = tuple(remote_dataset)   # ([],[],[],[],[])酱紫式儿的

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        # print(compute_nodes.index(data.location))
        remote_dataset[compute_nodes.index(data.location)].append((data, target))   # 向第i个节点中按照batch加入数据

    # print(len(remote_dataset[0]))

    def update(data, target, model, optimizer):   # 模型更新参数过程
        model.train()
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        pred = F.log_softmax(pred, dim=1)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        model.get()
        
        # if batch_idx % args.log_interval == 10:
        #     log_f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
        #                100. * batch_idx / len(federated_train_loader), loss.item()))
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
        #                100. * batch_idx / len(federated_train_loader), loss.item()))
        return model
    

    def train():   # 训练过程（方案核心）
        sum_time_s = time.time()
        print("-------------客户端开始训练-------------")
        train_time_s = time.time()   # 模型参数训练更新开始时间
        global num_nodes, result, m_nodes, params   # 总结点数、结果列表、坏节点数、参数列表
        
        # mal_model_params = m_model_create(params[0], m_nodes, log_f)   # 更新坏节点模型参数,参数全随机
        mal_model_params = weight_m_model_create(params[0], m_nodes, log_f, epoch)   # 更新坏节点模型参数,权重随机
        
        for data_index in range(len(remote_dataset[0]) - 1):  # 为什么要减1捏？最后一个batch数据不完整，执行dropout操作
            # update remote models更新好节点模型参数
            for remote_index in range(n_nodes):
                data, target = remote_dataset[remote_index][data_index]
                models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index])   # 按照节点、batch数据来更新节点本地模型

        # for remote_index in range(n_nodes):
        #     log_f.write("第{}个节点模型参数为：{}".format(remote_index, models[remote_index].parameters()))
        #     print("第{}个节点模型参数为：{}".format(remote_index, models[remote_index].parameters()))

        # encrypted aggregation
        train_time_f = time.time()   # 模型参数训练更新结束时间
        log_f.write("客户端训练时间：{:.8f}秒\n".format(train_time_f-train_time_s))
        print("客户端训练时间：{:.8f}秒".format(train_time_f-train_time_s))   # 开始加密
        print("-------------客户端开始加密-------------")
        encode_time_s = time.time()   # 加密开始时间
        encoding_params = []   # 加密参数，第一维度为不同层，第二维度为不同节点
        for p in range(len(params[0])):   # 按照层遍历
            entemp = []
            for remote_index in range(n_nodes):
                entemp.append(params[remote_index][p].copy().encrypt(protocol="paillier", public_key=pub))
                # entemp.append(params[remote_index][p])
            for remote_index in range(m_nodes):
                entemp.append(mal_model_params[remote_index][p].copy().encrypt(protocol="paillier", public_key=pub))
            encoding_params.append(entemp)   # 每层参数加密后的列表，第一维度为不同层，第二维度为不同节点
        encode_time_f = time.time()   # 加密结束时间
        log_f.write("客户端加密时间：{:.8f}秒\n".format(encode_time_f - encode_time_s))
        print("客户端加密时间：{:.8f}秒".format(encode_time_f - encode_time_s))   # 加密结束，开始聚合
        
        # ######## 自编码（AC）方案begin ########
        # result = [True] * num_nodes
        # err_list = []
        # # 依次调入接受到的权值矩阵并计算重构率
        # for index in range(num_nodes):
        #     encoding_temp_params = []
        #     for param_index in range(len(encoding_params)):
        #         encoding_temp_params.append(encoding_params[param_index][index])
        #     err_list.append(cal_ac.get_err(encoding_temp_params))
        # abnormal_score = cal_ac.get_abnormal_score(err_list)
        # abnormal_list = cal_ac.get_abnormal_list(abnormal_score, result)
        # credit_score = cal_ac.get_credit_score(1, abnormal_score)
        # ######## 自编码（AC）方案end ########
        
        print("-------------服务端与客户端开始聚合与解密-------------")
        new_params = []
        mal_list = [index for index, x in enumerate(result) if bool(x) is False]   # 第一轮都是true，从第二轮开始产生false
        print(mal_list)
        aggregation_time = 0
        decode_time = 0
        for p in range(len(encoding_params)):
            t1 = time.time()
            for ind in range(len(mal_list) - 1, -1, -1):
                # print(mal_list[ind])
                encoding_params[p].pop(mal_list[ind])   # 将坏节点产生的参数弹出
            temp = sum(encoding_params[p])   # 每层的总和
            t2 = time.time()
            new_params.append( (temp.mm(1/len(encoding_params[p]))).decrypt(private_key=pri) )  
            t3 = time.time()
            aggregation_time += (t2 - t1)
            decode_time += (t3 - t2)
            # new_params.append((temp * 1 / num_nodes))
        log_f.write("服务器端聚合时间：{:.8f}秒\n".format(aggregation_time))
        print("服务器端聚合时间：{:.8f}秒".format(aggregation_time))
        log_f.write("客户端解密时间：{:.8f}秒\n".format(decode_time))
        print("客户端解密时间：{:.8f}秒".format(decode_time))
        # print("异常检测开始")
        result = [True] * num_nodes
        result_0 = [True] * n_nodes + [False] * m_nodes
        print("-------------客户端开始距离计算-------------")
        distance_time_s = time.time()
        distance, weight_list = distance_compute(params+mal_model_params, new_params)
        distance_time_f = time.time()
        log_f.write("客户端距离计算时间：{:.8f}秒\n".format(distance_time_f - distance_time_s))
        print("客户端距离计算时间：{:.8f}秒".format(distance_time_f - distance_time_s))
        print("-------------服务端开始异常检测-------------")
        def error_detection_rate(list1, list2):
            sum = 0
            for i in range(len(list1)):
                if list1[i] != list2[i]:
                    sum += 1
            return sum/len(list1)
        detection_time_s = time.time()
        # print("-------------异常检测开始---------------")
        # detect_3sig(distance, result, 1, log_f)
        detect_dbscan(distance, result, 1, log_f)
        # constrained_k_means(distance, result, 0.3, log_f)
        # weight_detect_3sig(distance, result, 1, weight_list, log_f)
        # weight_detect_dbscan(distance, result, 1, weight_list, log_f)
        # weight_constrained_k_means(distance, result, 0.4, weight_list, log_f)
        log_f.write("误检率: {}\n".format(error_detection_rate(result, result_0)))
        print("误检率: {}\n".format(error_detection_rate(result, result_0)))
        # print("-------------异常检测结束---------------")
        detection_time_f = time.time()
        log_f.write("服务器端异常检测时间：{:.8f}秒\n".format(detection_time_f - detection_time_s))
        print("服务器端异常检测时间：{:.8f}秒".format(detection_time_f - detection_time_s))
        sum_time_f = time.time()
        log_f.write("一轮训练结束，整体时间(客户端时间+服务器端时间): {:.8f}秒\n".format(sum_time_f - sum_time_s))
        print("一轮训练结束，整体时间(客户端时间+服务器端时间): {:.8f}秒\n".format(sum_time_f - sum_time_s))
        # cleanup
        test()

        if epoch == 0:
            print("===============模型参数重新初始化==============")
            reagg_params = []
            mal_list_temp = [index for index, x in enumerate(result) if bool(x) is False]
            print("首轮检测：", mal_list_temp)
            print("encoding_params长度:" + str(len(encoding_params[0])))
            for p in range(len(encoding_params)):
                for ind in range(len(mal_list_temp) - 1, -1, -1):
                    encoding_params[p].pop(mal_list_temp[ind])  # 将坏节点产生的参数弹出

            c_id = random.randint(0, len(encoding_params[p]) - 1)
            print("c_id: {}".format(c_id))
            for p in range(len(encoding_params)):
                reagg_params.append(encoding_params[p][c_id].decrypt(private_key=pri))
            # new_params.append((temp * 1 / num_nodes))

            with torch.no_grad():
                for model in params:
                    for param in model:
                        param *= 0
                for remote_index in range(n_nodes):
                    for param_index in range(len(params[0])):
                        params[remote_index][param_index].set_(reagg_params[param_index].to(device))

        else:
            with torch.no_grad():
                for model in params:
                    for param in model:
                        param *= 0
                for remote_index in range(n_nodes):
                    for param_index in range(len(params[0])):
                        params[remote_index][param_index].set_(new_params[param_index].to(device))
        # print("一轮训练结束")


    # def train_test():
    #     print("train_test")
    #     models[0].eval()
    #     train_loss = [0] * n_nodes
    #     correct = [0] * n_nodes
        
    #     for i in range(n_nodes):
    #         x = 0
    #         for data, target in train_loader:
    #             x += 1
    #             if x >= i * len(train_loader.dataset)/n_nodes and x < (i+1) * len(train_loader.dataset)/n_nodes:
    #                 data, target = data.to(device), target.to(device)
    #                 output = models[0](data)
    #                 train_loss[i] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    #                 # pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
    #                 # pred = output.data.max(1, keepdim=True)[1]
    #                 pred = output.argmax(dim=1)
    #                 correct[i] += pred.eq(target.view_as(pred)).sum().item()

    #         train_loss[i] /= (len(train_loader.dataset)/n_nodes)
    #         log_f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
    #             train_loss, correct[i], len(train_loader.dataset)/n_nodes, 100. * correct[i] / (len(train_loader.dataset)/n_nodes)))
    #         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #             train_loss, correct[i], len(train_loader.dataset)/n_nodes, 100. * correct[i] / (len(train_loader.dataset)/n_nodes)))     



    def test():
        print("test")
        models[0].eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = models[0](data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            # pred = output.data.max(1, keepdim=True)[1]
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        log_f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))



    t_s = time.time()

    for epoch in range(args.epochs):

        log_f.write("Epoch %d\n" % (epoch + 1))
        print(f"Epoch {epoch + 1}", time.asctime(time.localtime(time.time())))
        train()
        test()
        # print(time.asctime(time.localtime(time.time())))

    t_f = time.time()
    log_f.write("运行总时间：{:.4f}秒\n".format(t_f - t_s))
    print("运行总时间：{:.4f}秒".format(t_f - t_s))
    print(time.asctime(time.localtime(time.time())) + "\n\n\n\n")

    '''
    if args.save_model:
        print("saving model")
        torch.save(models[0].state_dict(), "./model_save/(%d + %d)_%.2f.pt" % (n_nodes, m_nodes, m_nodes/num_nodes))
    '''
    log_f.close()

