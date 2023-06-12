
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from new_detect import distance_compute, weight_detect_3sig, weight_m_model_create, m_model_create, weight_detect_dbscan, weight_constrained_k_means,detect_3sig
    from torchvision import datasets, transforms
    import syft as sy
    import time
    import cal_ac


    class Arguments():
        def __init__(self):
            self.batch_size = 512
            self.test_batch_size = 1024
            self.epochs = 30
            self.lr = 0.01
            self.log_interval = 50
            self.momentum = 0.5
            self.cuda = True


    args = Arguments()

    class Net(nn.Module):
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

    key_len = 150
    print("秘钥长度", key_len)
    pub, pri = sy.keygen(n_length=key_len)  # 假设参与方和服务器已经实现了对密钥的生成和分发，其中服务器和参与方拥有公钥，参与方拥有彼此相同的私钥
    compute_nodes = []
    result = []
    m_nodes = 4
    n_nodes = 16
    num_nodes = m_nodes + n_nodes  # 参与方数量需广播给各个参与方
    # log_f = open("./log/(%d + %d)_%.1f .txt" % (n_nodes, m_nodes, m_nodes/num_nodes), "a")
    # log_f.write(time.asctime(time.localtime()) + "\n")
    # log_f.write(time.asctime(time.localtime()) + "\n")

    # mal_model = []
    # m_model_create(mal_model, m_nodes, (10, 784), (1, 10))  生成异常模型
    models = []
    optimizers = []
    # schedulers = []
    params = []
    hook = sy.TorchHook(torch)
    device = torch.device("cuda" if args.cuda else "cpu")
    print("系统信息：%d个参与方" % num_nodes)
    # log_f.write("系统信息：%d个参与方\n" % num_nodes)

    for i in range(n_nodes):
        compute_nodes.append(sy.VirtualWorker(hook, id="client %d" % i))
        models.append(Net().to(device))
        optimizers.append(optim.SGD(models[i].parameters(), lr=args.lr, momentum=args.momentum))
        # schedulers.append(optim.lr_scheduler.ExponentialLR(optimizers[i], gamma=0.1, last_epoch=-1))
        params.append(list(models[i].parameters()))
    model_shape = []
    for p in params[0]:
        print(p.shape)
        model_shape.append(p.shape[-1])

    split_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 这是ImageNet数据集的方差和均值
    ]))
    federated_train_loader = sy.FederatedDataLoader(
        split_dataset.federate(compute_nodes), batch_size=args.batch_size, shuffle=True)  # 联邦数据集
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 这是ImageNet数据集的方差和均值
    ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    remote_dataset = []
    for i in range(n_nodes):
        remote_dataset.append(list())
    remote_dataset = tuple(remote_dataset)

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        # print(compute_nodes.index(data.location))
        remote_dataset[compute_nodes.index(data.location)].append((data, target))

    # print(len(remote_dataset[0]))

    def update(data, target, model, optimizer):
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

        '''
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                       100. * batch_idx / len(federated_train_loader), loss.item()))
        '''
        return model


    def train():
        train_time1 = time.time()
        global num_nodes, result, m_nodes, params
        for data_index in range(len(remote_dataset[0]) - 1):
            # update remote models
            for remote_index in range(n_nodes):
                data, target = remote_dataset[remote_index][data_index]
                models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index])
        train_time2 = time.time()
        mal_model_params = weight_m_model_create(params[0], m_nodes)
        #mal_model_params = m_model_create(params[0], m_nodes)
        # encrypted aggregation
        print("本地训练完成，开始加密  %.2f" % (train_time2 - train_time1))
        encode_time1 = time.time()
        encoding_params = []
        for p in range(len(params[0])):
            entemp = []
            for remote_index in range(n_nodes):
                entemp.append(params[remote_index][p].copy().encrypt(protocol="paillier", public_key=pub))
                # entemp.append(params[remote_index][p])
            for remote_index in range(m_nodes):
                entemp.append(mal_model_params[remote_index][p].copy().encrypt(protocol="paillier", public_key=pub))
            encoding_params.append(entemp)
        encode_time2 = time.time()
        print("加密完成，开始聚合并解密更新 %.2f" % (encode_time2 - encode_time1))
        # print(encoding_model)
        new_params = []
        '''
        # AC方案begin
        result = [True] * num_nodes
        err_list = []
        # 依次调入接受到的权值矩阵并计算重构率
        for index in range(num_nodes):
            encoding_temp_params = []
            for param_index in range(len(encoding_params)):
                encoding_temp_params.append(encoding_params[param_index][index])
            err_list.append(cal_ac.get_err(encoding_temp_params))
        abnormal_score = cal_ac.get_abnormal_score(err_list)
        abnormal_list = cal_ac.get_abnormal_list(abnormal_score, result)
        credit_score = cal_ac.get_credit_score(1, abnormal_score)
        # AC方案end
        '''
        mal_list = [index for index, x in enumerate(result) if bool(x) is False]
        print(mal_list)
        test()
        for p in range(len(encoding_params)):
            for ind in range(len(mal_list) - 1, -1, -1):
                encoding_params[p].pop(mal_list[ind])
            temp = sum(encoding_params[p])
            new_params.append((temp.mm(1/len(encoding_params[p]))).decrypt(private_key=pri) )
            # new_params.append((temp * 1 / num_nodes))
        print("异常检测开始")
        result = [True] * num_nodes
        distance, weight_list = distance_compute(params+mal_model_params, new_params)
        weight_detect_3sig(distance, result, 1, weight_list)
        # detect_3sig(distance, result, 1)
        #weight_detect_dbscan(distance, result, 1, weight_list)
        # weight_constrained_k_means(distance, result, weight_list)

        # cleanup
        with torch.no_grad():
            for model in params:
                for param in model:
                    param *= 0

            for remote_index in range(n_nodes):
                for param_index in range(len(params[0])):
                    params[remote_index][param_index].set_(new_params[param_index].to(device))
        test()
        print("一轮训练结束")


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
        # log_f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
            # test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))



    t = time.time()

    for epoch in range(args.epochs):
        # log_f.write("Epoch %d\n" % (epoch + 1))
        print(f"Epoch {epoch + 1}")
        train()
        test()
        print(time.asctime(time.localtime(time.time())))

    total_time = time.time() - t
    # log_f.write('运行总时间：%.4f 秒\n' % total_time)
    print('运行总时间：', round(total_time, 2), '秒')

    '''
    if args.save_model:
        print("saving model")
        torch.save(models[0].state_dict(), "./model_save/(%d + %d)_%.2f.pt" % (n_nodes, m_nodes, m_nodes/num_nodes))
    '''
    # log_f.close()
