import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import os
import pickle
import random
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as Data
import copy

# Hyper Parameters
LR = 0.002
BATCH_SIZE = 32
EPOCH = 100
EACH_TRAIN = 10000
TOTAL_EPOCH = 10
L2_REGULATION = 0.001


print('正在构建网络...')
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(2048, 32)

resnet.cuda()
print('正在读取网络...')
try:
    resnet.load_state_dict(torch.load('./net_param.pkl'))
except Exception as err:
    # print(err)
    print('提取网络时发生错误，重新构建网络。')
print('已构建网络')

for this_epoch in range(TOTAL_EPOCH):
    print(f'开始进行第{this_epoch}轮训练')
    data_list = []
    check_list = []

    print('正在读取数据...')
    read_step = 0
    for root, dirs, files in os.walk('./out'):
        for f in files:
            if read_step not in range(this_epoch * EACH_TRAIN, (this_epoch + 1) * EACH_TRAIN):
                continue

            path = 'out/' + f
            img = cv2.imread(path)
            pos1 = f.find('_')
            pos2 = f.find('.')
            cur_label = int(f[pos1 + 1:pos2])
            if random.random() < 0.95:
                data_list.append((img, cur_label))
            else:
                check_list.append((img, cur_label))

            read_step += 1

    if (len(data_list) == 0):
        print('全部训练集已经完成')
        break
    else:
        print(f'读取了{len(data_list)}个训练数据')

    torch_data = torch.empty((len(data_list), 3, 224, 224), dtype=torch.float)
    torch_label = torch.empty(len(data_list), dtype=torch.long).cuda()

    torch_check_data = torch.empty(
        (len(check_list), 3, 224, 224), dtype=torch.float)
    torch_check_label = torch.empty(
        len(check_list), dtype=torch.long)

    trans_to_tensor = transforms.ToTensor()
    trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    print('数据读取完成！\n正在载入数据...')

    for i in range(len(data_list)):
        cur_img = cv2.cvtColor(data_list[i][0], cv2.COLOR_BGR2RGB)
        # cur_img = cv2.resize(cur_img, (224, 224))
        torch_data[i] = trans_normalize(trans_to_tensor(cur_img))
        torch_label[i] = data_list[i][1]

    for i in range(len(check_list)):
        cur_img = cv2.cvtColor(check_list[i][0], cv2.COLOR_BGR2RGB)
        # cur_img = cv2.resize(cur_img, (224, 224))
        torch_check_data[i] = trans_normalize(trans_to_tensor(cur_img))
        torch_check_label[i] = check_list[i][1]

    print('数据载入完成，删除缓存的内容...')

    del data_list, check_list

    print('将数据载入训练集中...')
    '''
    torch_data = torch.from_numpy(np_data).type(torch.FloatTensor)
    torch_label = torch.from_numpy(np_label).type(torch.LongTensor).cuda()

    torch_check_data = torch.from_numpy(check_data).type(torch.FloatTensor)
    torch_check_label = torch.from_numpy(check_label).type(torch.LongTensor)
    '''

    dataset = Data.TensorDataset(torch_data, torch_label)
    dataloader = Data.DataLoader(dataset, BATCH_SIZE, True)

    chech_dataloader = Data.DataLoader(Data.TensorDataset(
        torch_check_data, torch_check_label), BATCH_SIZE, False)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        resnet.parameters(), LR, weight_decay = L2_REGULATION)

    print('开始训练')

    for epoch in range(EPOCH):
        for step, (batchx, batchy) in enumerate(dataloader):
            resnet.train()
            torch.cuda.empty_cache()
            prediction = resnet(batchx.cuda())
            loss = loss_func(prediction, batchy.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f'Epoch:{epoch},Step:{step},Loss:{loss.item()}')
                print('Start eval...')
                resnet.eval()
                check_pred = torch.empty((torch_check_label.shape[0],32))
                for check_step, (checkx, checky) in enumerate(chech_dataloader):
                    torch.cuda.empty_cache()
                    cur_pred = resnet(checkx.cuda()).cpu()
                    cur_pred = copy.copy(cur_pred)
                    check_pred[check_step*BATCH_SIZE:check_step *
                               BATCH_SIZE+cur_pred.shape[0]] = cur_pred[:]
                    del cur_pred

                check_pred = torch.max(check_pred, 1)[1]
                pred_acc: torch.Tensor = (check_pred == torch_check_label)
                pred_acc = pred_acc.type(torch.FloatTensor).mean()

                print(f'Epoch:{epoch},Step:{step},Acc:{pred_acc}')
            elif step % 5 == 0:
                print(f'Epoch:{epoch},Step:{step},Loss:{loss.item()}')

        resnet.cpu()
        torch.save(resnet.state_dict(), 'net_param.pkl')
        print('网络保存成功！')
        resnet.cuda()

    torch.save(resnet.cpu().state_dict(), 'net_param.pkl')
    print('网络保存成功！')
