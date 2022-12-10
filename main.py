import os, time, datetime, tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_data():
    data = np.array(pd.read_csv('Dataset.csv'))
    x, y = data[:, :4], data[:, 5:6]
    x_scaler = StandardScaler()
    x = x_scaler.fit_transform(x)
    y_class = [list(np.unique(y[:, i])) for i in range(y.shape[1])]
    y_class_num = [len(i) for i in y_class]
    y = np.array(
        [np.array([y_class[0].index(y_[0]), y_class[1].index(y_[1]), y_class[2].index(y_[2])]) for y_ in y])
    x_train, x_test, y_train, y_test = train_test_split(np.expand_dims(x, axis=1), y, test_size=0.2, shuffle=True, random_state=0)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test, y_class, y_class_num


def cal_cm(y_true, y_pred, CLASS_NUM):
    y_true, y_pred = y_true.to('cpu').detach().numpy(), np.argmax(y_pred.to('cpu').detach().numpy(), axis=1)
    y_true, y_pred = y_true.reshape((-1)), y_pred.reshape((-1))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(CLASS_NUM)))
    return cm


class CNN(nn.Module):
    def __init__(self, classes_list):
        super(CNN, self).__init__()
        act = nn.ReLU()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            act,
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            act,
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            act,
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 2, out_features=64),
            act,
            nn.Linear(in_features=64, out_features=classes_list[0])
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64 * 2, out_features=64),
            act,
            nn.Linear(in_features=64, out_features=classes_list[1])
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=64 * 2, out_features=64),
            act,
            nn.Linear(in_features=64, out_features=classes_list[2])
        )

    def forward(self, x):
        feature = self.cnn(x)
        y1 = self.fc1(feature)
        y2 = self.fc2(feature)
        y3 = self.fc3(feature)
        return y1, y2, y3

    def compute_loss(self, y_true, y_pred, loss):
        total_loss = []
        for i in range(3):
            total_loss.append(loss(y_pred[i], y_true[:, i]))
        return sum(total_loss)


if __name__ == '__main__':
    BATCH_SIZE, EPOCH = 128, 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, x_test, y_train, y_test, y_class, y_class_num = get_data()
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_dataset = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_dataset = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

    model = CNN(y_class_num)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=5e-4)
    lrstep = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=10)
    loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)

    with open('train.log', 'w+') as f:
        f.write('loss,test_loss,acc1,test_acc1,acc2,test_acc2,acc3,test_acc3')

    best_acc = 0
    for epoch in range(EPOCH):
        model.to(DEVICE)
        model.train()
        train_loss, train_cm1, train_cm2, train_cm3 = [], np.zeros(shape=(y_class_num[0], y_class_num[0])), np.zeros(
            shape=(y_class_num[1], y_class_num[1])), np.zeros(shape=(y_class_num[2], y_class_num[2]))
        begin = time.time()
        for x, y in tqdm.tqdm(train_dataset):
            x, y = x.to(DEVICE), y.to(DEVICE).long()

            pred = model(x.float())
            l = model.compute_loss(y, pred, loss)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss.append(float(l.data))
            train_cm1 += cal_cm(y[:, 0], pred[0], y_class_num[0])
            train_cm2 += cal_cm(y[:, 1], pred[1], y_class_num[1])
            train_cm3 += cal_cm(y[:, 2], pred[2], y_class_num[2])
        train_loss = np.mean(train_loss)
        train_acc1 = np.diag(train_cm1).sum() / (train_cm1.sum() + 1e-7)
        train_acc2 = np.diag(train_cm2).sum() / (train_cm2.sum() + 1e-7)
        train_acc3 = np.diag(train_cm3).sum() / (train_cm3.sum() + 1e-7)

        test_loss, test_cm1, test_cm2, test_cm3 = [], np.zeros(shape=(y_class_num[0], y_class_num[0])), np.zeros(
            shape=(y_class_num[1], y_class_num[1])), np.zeros(shape=(y_class_num[2], y_class_num[2]))
        model.eval()
        with torch.no_grad():
            for x, y in tqdm.tqdm(test_dataset):
                x, y = x.to(DEVICE), y.to(DEVICE).long()

                pred = model(x.float())
                l = model.compute_loss(y, pred, loss)
                test_loss.append(float(l.data))
                test_cm1 += cal_cm(y[:, 0], pred[0], y_class_num[0])
                test_cm2 += cal_cm(y[:, 1], pred[1], y_class_num[1])
                test_cm3 += cal_cm(y[:, 2], pred[2], y_class_num[2])
        test_loss = np.nanmean(test_loss)
        test_acc1 = np.diag(test_cm1).sum() / (test_cm1.sum() + 1e-7)
        test_acc2 = np.diag(test_cm2).sum() / (test_cm2.sum() + 1e-7)
        test_acc3 = np.diag(test_cm3).sum() / (test_cm3.sum() + 1e-7)

        if (test_acc1 + test_acc2 + test_acc3) / 3 > best_acc:
            best_acc = (test_acc1 + test_acc2 + test_acc3) / 3
            model.to('cpu')
            torch.save(model, 'model.pt')
        with open('train.log', 'a+') as f:
            f.write('\n{:.8f},{:.8f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(train_loss, test_loss, train_acc1, test_acc1, train_acc2, test_acc2, train_acc3, test_acc3))
        print(
            '{} epoch:{}, time:{:.2f}s, train_loss:{:.8f}, test_loss:{:.8f}, train_acc1:{:.4f}, test_acc1:{:.4f}, train_acc2:{:.4f}, test_acc2:{:.4f}, train_acc3:{:.4f}, test_acc3:{:.4f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch + 1, time.time() - begin, train_loss, test_loss, train_acc1, test_acc1, train_acc2, test_acc2, train_acc3, test_acc3
            ))
        lrstep.step()
