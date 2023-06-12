import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import embed_train
import image_deal

device = "cuda"
validation_split = 0.15

crossentropyloss = nn.CrossEntropyLoss()
crossentropyloss = crossentropyloss.to(device)


def gen_train_loader():
    dataset_size = len(image_deal.normal_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed()
    np.random.shuffle(indices)
    train_indices = indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(image_deal.normal_dataset, batch_size=128, sampler=train_sampler)
    return train_loader, split


def gen_validation_loader():
    validation_loader = torch.utils.data.DataLoader(image_deal.test_dataset, batch_size=256)
    return validation_loader

# def create_soptimizer(model):
#    optimizer = optim.SGD(model.parameters(), lr=0.01)
#    return optimizer


def update_s(data, target, model, optimizer, layer_index, key, sign, lamda):
    model.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    pred = model(data)
    ce_loss = crossentropyloss(pred, target)
    temp_w = list(model.parameters())[layer_index]
    # print("weight shape", temp_w.shape)  64 3 3 3
    weight = embed_train.weight_deal(temp_w)
    # print("weight ", weight.copy().get())
    x = torch.tensor(key)
    # print("key", x)
    b = torch.tensor(sign, dtype=torch.float32)
    reg_loss = embed_train.compute_loss(x, weight, b, lamda, device)
    loss = reg_loss + ce_loss
    loss.backward()
    optimizer.step()

    # if batch_idx % args.log_interval == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #        epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
    #               100. * batch_idx / len(federated_train_loader), loss.item()))
    return model


def service_embed_operation(model, optimizer, key, sign, layer_index, lamda, num_epochs=10):
    train_loader, split = gen_train_loader()
    print("服务器端数据量：", split)
    model.train()
    for epoch in range(num_epochs):
        print("service epoch ", epoch+1)
        for batch_index, (data, target) in enumerate(train_loader):
            embed_model = update_s(data, target, model, optimizer, layer_index, key, sign, lamda)
        # test_s(model, key, sign, layer_index)
        evaluation_index = embed_train.evaluate_sign(key, model, layer_index, sign)
        print('嵌入成功率:({:.0f}%)'.format(100. * evaluation_index))

        if evaluation_index >= 0.90 and epoch >= 5:  # 提前退出
            print("服务器端训练轮数:", epoch+1)
            embed_test(embed_model)
            break

    return embed_model


def embed_test(embed_model):
    embed_model.eval()
    test_loss = 0
    correct = 0
    validation_loader = gen_validation_loader()
    for data, target in validation_loader:
        data, target = data.to(device), target.to(device)
        output = embed_model(data)
        test_loss += crossentropyloss(output, target).item()  # sum up batch loss
        # pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        # _,pred = torch.max(output,1)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validation_loader.dataset)
    print('service initial model\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    return correct / len(validation_loader.dataset)


def set_embed_model_parameters(models, embed_model):
    for key, value in embed_model.state_dict().items():
        for model in models:
            model.state_dict()[key].copy_(value)


def evaluation_models(models, x, layer_index, sign_b, threshold):
    unconfident_indexs = list()
    for remote_index in range(len(models)):
        extract_index = embed_train.evaluate_sign(x, models[remote_index], layer_index, sign_b)
        if extract_index <= threshold:
            unconfident_indexs.append(remote_index)
    return unconfident_indexs


def top_n(distance_list, distance_t, n):
    for i in range(n):
        ind = distance_list.index(max(distance_list))
        distance_t.append(ind)
        distance_list[ind] = -1


def add_model(dst_model, src_model):
    """Add the parameters of two models.

    Args:
        dst_model (torch.nn.Module): the model to which the src_model will be added.
        src_model (torch.nn.Module): the model to be added to dst_model.
    Returns:
        torch.nn.Module: the resulting model of the addition.
    """

    # src_model.get()
    # print("src_model\n", src_model)

    params1 = src_model.parameters()
    # print(params1)
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                print("param1\n", param1)
                # print(param1.data) tensor([])
                # print(name1) conv1.weight
                # print(dict_params2[name1].shape) torch.Size([20, 1, 5, 5])
                dict_params2[name1].set_(param1.data + dict_params2[name1].data)


    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.

    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.

    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def federated_avg(model_list) -> torch.nn.Module:

    nr_models = len(model_list)
    model = type(model_list[0])().to(device)
    # params = list(model.parameters())
    # for p in params:
        # torch.nn.init.zeros_(p)
    # print(model.parameters)
    for i in range(nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)
    return model


if __name__ == '__main__':
    import math
    loss_func = nn.CrossEntropyLoss()

    m_train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])  # 这是CIFAR10数据集的方差和均值
    ]))

    m_test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])  # 这是CIFAR10数据集的方差和均值
    ]))
    train_indices = indices[:10000]
    # val_indices = indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    m_train_loader = torch.utils.data.DataLoader(m_train_dataset, batch_size=256, sampler=train_sampler, num_workers=4)
    m_test_loader = torch.utils.data.DataLoader(m_test_dataset, batch_size=512, shuffle=True, num_workers=4)
    # print(len(m_train_loader))

    def set_parameters(model_p, modelt_p):
        for param_index in range(len(modelt_p)):
            model_p[param_index].set_(modelt_p[param_index])

    def t_update1(data, target, model, optimizer):
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()

    def t_update2(data, target, model, optimizer):
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        pred = F.log_softmax(pred, dim=1)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()


    def t_test1(model):
        correct = 0
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in m_test_loader:
                # Forward pass
                out = model(x)
                loss = loss_func(out, y)
                predicted = torch.max(out, 1)[1]
                correct += (predicted == y).sum().item()
                test_loss += loss.item()
            print("test loss = {:.2f}, Accuracy={:.6f}".format(test_loss / len(m_test_loader), correct / len(
                m_test_loader) / m_test_loader.batch_size))  # 求验证集的平均损失是多少

    def t_test2(model):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in m_test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # _,pred = torch.max(output,1)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(m_test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(m_test_loader.dataset), 100. * correct / len(m_test_loader.dataset)))


    class NetT1(nn.Module):
        def __init__(self):
            super(NetT1, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
            self.MaxPool = nn.MaxPool2d(2, 2)
            self.AvgPool = nn.AvgPool2d(4, 4)
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))  # (3,32,32) -> (16,32,32)
            x = self.MaxPool(F.relu(self.conv2(x)))  # (16,32,32) -> (32,16,16)
            x = F.relu(self.conv3(x))  # (32,16,16) -> (64,16,16)
            x = self.MaxPool(F.relu(self.conv4(x)))  # (64,16,16) -> (128,8,8)
            x = self.MaxPool(F.relu(self.conv5(x)))  # (128,8,8) -> (256,4,4)
            x = self.AvgPool(x)  # (256,1,1)
            x = x.view(-1, 256)  # (256)
            x = self.fc3(self.fc2(self.fc1(x)))  # (32)
            x = self.fc4(x)  # (10)
            return x


    cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}


    class VGG(nn.Module):
        def __init__(self, net_name):
            super(VGG, self).__init__()

            # 构建网络的卷积层和池化层，最终输出命名features，原因是通常认为经过这些操作的输出为包含图像空间信息的特征层
            self.features = self._make_layers(cfg[net_name])

            # 构建卷积层之后的全连接层以及分类器
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),  # fc1
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),  # fc2
                nn.ReLU(True),
                nn.Linear(512, 10),  # fc3，最终cifar10的输出是10类
            )
            # 初始化权重
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

        def forward(self, x):
            x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
            x = x.view(x.size(0), -1)
            x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
            return x

        def _make_layers(self, cfg):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    # layers += [conv2d, nn.ReLU(inplace=True)]
                    layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                               nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)


    net = VGG('VGG16')

    class NetT2(nn.Module):

        def __init__(self):
            super(NetT2, self).__init__()
            self.feature = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 10)
            )

        def forward(self, x):
            x = self.feature(x)
            output = self.classifier(x)
            return output

    model_t1 = NetT1()
    model_t2 = NetT2()
    o1 = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    o2 = optim.SGD(model_t2.parameters(), lr=0.01, momentum=0.9)

    # for i in list(model_t1.parameters()):
        # print(i.shape)

    for i in range(50):
        print(i+1)
        for batch, (data, target) in enumerate(m_train_loader):
            t_update1(data, target, net, o1)

            # t_update2(data, target, model_t2, o2)

        t_test1(net)
        # t_test2(model_t2)
'''
    model_p1 = list(model_t1.parameters())
    model_p2 = list(model_t2.parameters())
    for i in range(3):
        for batch, (data, target) in enumerate(m_train_loader):
            t_update(data, target, model_t1, o1)
            # t_update(data, target, model_t2, o2)
        t_test(model_t1)
        # t_test(model_t2)
    with torch.no_grad():
        set_parameters(model_p2, model_p1)
    print(list(model_t1.parameters()))
    print(list(model_t2.parameters()))
'''


'''
    def update(data, target, model, optimizer):
        model.train()
        model.send(data.location)
        optimizer.zero_grad()
        pred = model(data)
        # pred = F.log_softmax(pred, dim=1)
        # crossentropyloss = nn.CrossEntropyLoss()
        ce_loss = F.cross_entropy(pred, target)
        if data.location.id == "service":
            temp_w = list(model.parameters())[layer_index]
            # print("weight shape", temp_w.shape)  64 3 3 3
            weight = embed_train.weight_deal(temp_w)
            # print("weight ", weight.copy().get())
            x = torch.tensor(X)
            # print("key", x)
            x = x.send(data.location)
            b = torch.tensor(sign_b, dtype=torch.float32)
            b = b.send(data.location)
            reg_loss = embed_train.compute_loss(x, weight, b, lamda)
            loss = reg_loss + ce_loss
        else:
            loss = ce_loss
        loss.backward()
        optimizer.step()
        model.get()
        x.get()
        b.get()
'''
