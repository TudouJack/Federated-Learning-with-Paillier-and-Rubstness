import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from detect import distance_compute, detect_3sig, m_model_create,detect_dbscan
from torchvision import datasets, transforms
import syft as sy
import time
import sys
import server


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 20
        self.lr = 0.001
        self.log_interval = 50
        self.save_model = True


args = Arguments()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        # self.fc1 = nn.Linear(784, 500)
        # self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return x


if __name__ == '__main__':
    pub, pri = sy.keygen()  # 假设参与方和服务器已经实现了对密钥的生成和分发，其中服务器和参与方拥有公钥，参与方拥有彼此相同的私钥
    compute_nodes = []
    result = []
    m_nodes = 4
    n_nodes = 16
    num_nodes = m_nodes + n_nodes  # 参与方数量需广播给各个参与方
    log_f = open("./log/(%d + %d)_%.1f .txt" % (n_nodes, m_nodes, m_nodes/num_nodes), "a")
    log_f.write(time.asctime(time.localtime()) + "\n")
    # mal_model = []
    # m_model_create(mal_model, m_nodes, (10, 784), (1, 10))  生成异常模型
    models = []
    optimizers = []
    # schedulers = []
    params = []
    hook = sy.TorchHook(torch)
    print("系统信息：%d个参与方" % num_nodes)
    log_f.write("系统信息：%d个参与方\n" % num_nodes)

    for i in range(n_nodes):
        compute_nodes.append(sy.VirtualWorker(hook, id="client %d" % i))
        models.append(Net())
        optimizers.append(optim.SGD(models[i].parameters(), lr=args.lr))
        # schedulers.append(optim.lr_scheduler.ExponentialLR(optimizers[i], gamma=0.1, last_epoch=-1))
        params.append(list(models[i].parameters()))

    split_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 这是ImageNet数据集的方差和均值
    ]))
    federated_train_loader = sy.FederatedDataLoader(
        split_dataset.federate(compute_nodes), batch_size=args.batch_size, shuffle=True)  # 联邦数据集
    test_loader = torch.utils.data.DataLoader(split_dataset, batch_size=args.test_batch_size, shuffle=True)

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
        optimizer.zero_grad()
        pred = model(data)
        pred = F.log_softmax(pred, dim=1)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                       100. * batch_idx / len(federated_train_loader), loss.item()))
        return model


    def train():
        print("开始模型训练")
        train_time = 0
        global num_nodes, result, m_nodes
        for data_index in range(len(remote_dataset[0]) - 1):
            # update remote models
            train_stime = time.time()
            for remote_index in range(n_nodes):
                data, target = remote_dataset[remote_index][data_index]
                models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index])
            train_ftime = time.time()
            train_time += (train_ftime - train_stime)
            # encrypted aggregation
            if data_index == (len(remote_dataset[0]) - 2):
                log_f.write("参与方训练总耗时：%.8f\n" % train_time + "各个参与方训练耗时：%.8f\n" % (train_time / num_nodes))
                print("参与方训练总耗时：%.4f\n" % train_time + "各个参与方训练耗时：%.4f" % (train_time / num_nodes))
                t1 = 0
                t2 = 0
                new_params = list()
                avg_model = list()
                compute_model_list = []  # 未加密的模型参数列表
                for remote_index in range(n_nodes):
                    compute_model_list.append(list())
                    for param_i in range(len(params[0])):
                        compute_model_list[remote_index].append(params[remote_index][param_i].copy().get())
                mal_model = []
                m_model_create(mal_model, m_nodes, (10, 784), (1, 10))  # 生成异常模型
                compute_model_list.extend(mal_model)
                # print(compute_model_list)
                print("开始同态加密")
                en_stime = time.time()
                for param_i in range(len(params[0])):
                    # 模拟各参与方对模型参数加密并上传
                    spdz_params = list()  # 服务器端收集的加密模型列表
                    for remote_index in range(len(compute_model_list)):
                        # (Wrapper)>[PointerTensor | me:44036362214 -> client 0:62892427009]
                        # <class 'torch.nn.parameter.Parameter'>
                        spdz_params.append(
                            compute_model_list[remote_index][param_i].copy().encrypt(protocol="paillier", public_key=pub))
                        print(spdz_params[remote_index].get())
                        if param_i == 0:
                            com_size = compute_model_list[remote_index][param_i].size()[0] * compute_model_list[remote_index][param_i].size()[1]
                        else:
                            com_size = compute_model_list[remote_index][param_i].size()[0]
                    store_space = (sys.getsizeof(spdz_params[0][0]) * com_size +sys.getsizeof(spdz_params[0].storage()))
                    print("各个参与方加密后模型参数 (%d) 列表大小：%d" % (param_i + 1, store_space))
                    log_f.write("各个参与方加密后模型参数 (%d) 列表大小：%d\n" % (param_i + 1, store_space))
                    # 进行服务器端的计算
                    com_stime = time.time()
                    spdz_sum = sum(spdz_params)
                    d_stime = time.time()
                    # 在此处进行异常参与方的处理：每轮聚合恶意参与方都上传异常模型，服务器运行检测算法，根据上一轮结果使异常模型不参与聚合
                    for m_i, x in enumerate(result):  # 此处用的是上一个epoch的结果
                        if x is False:
                            spdz_sum -= spdz_params[m_i]
                    d_ftime = time.time()
                    t2 += (d_ftime - d_stime)
                    avg_p = spdz_sum.mm(1 / num_nodes)
                    avg_model.append(avg_p)
                    com_ftime = time.time()
                    t1 += (com_ftime - com_stime)
                en_ftime = time.time()
                log_f.write("服务器端计算总耗时：%.8f\n" % t1 + "服务器端恶意参与方处理耗时：%.8f\n" % t2)
                log_f.write("本地模型加密运行时间:%.8f秒\n" % (en_ftime - en_stime - t1))
                print("本地模型加密和聚合完成，运行时间:%.4f秒" % (en_ftime - en_stime))

                for param_i in range(len(params[0])):
                    new_params.append(avg_model[param_i].decrypt(private_key=pri))
                # print(new_params)
                de_ftime = time.time()
                log_f.write("全局模型解密完成，各个参与方运行时间:%.8f秒\n" % ((de_ftime - en_ftime)/num_nodes))
                print("全局模型解密完成，运行时间:%.4f秒" % (de_ftime - en_ftime))

                print("异常检测开始")
                result = [True] * num_nodes
                distance = distance_compute(compute_model_list, new_params, log_f)
                # detect_3sig(distance, result, log_f, 3)
                detect_dbscan(distance, result, log_f, 2)

                # cleanup
                with torch.no_grad():
                    for model in params:
                        for param in model:
                            param *= 0

                    for model in models:
                        model.get()

                    for remote_index in range(n_nodes):
                        for param_index in range(len(params[0])):
                            params[remote_index][param_index].set_(new_params[param_index])
            else:
                with torch.no_grad():
                    for model in models:
                        model.get()
        print("一轮训练结束")


    def test():
        print("test")
        models[0].eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            output = models[0](data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # _,pred = torch.max(output,1)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        log_f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


    t = time.time()

    for epoch in range(args.epochs):
        log_f.write("Epoch %d\n" % (epoch + 1))
        print(f"Epoch {epoch + 1}")
        train()
        test()

    total_time = time.time() - t
    log_f.write('运行总时间：%.4f 秒\n' % total_time)
    print('运行总时间：', round(total_time, 2), '秒')

    if args.save_model:
        print("saving model")
        torch.save(models[0].state_dict(), "./model_save/(%d + %d)_%.2f.pt" % (n_nodes, m_nodes, m_nodes/num_nodes))
    log_f.close()
