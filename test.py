from CCANet_test.CCANet_test import CCA101

import torch
from data_preprocessing.dataset import MyDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm



batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ResNet18()
# model = ResNet50()
# model = ResNet50_Attention()
# model = ResNet50_Attention_LSTM()
# model = eca_resnet101().to(device)
model = CCA101().to(device)# ResNet50_eca
model_dict = torch.load('model_dict/CCANet_result.pth', map_location=device)
model.load_state_dict(model_dict['ResNet'])

dataset_test = MyDataset('processed/test/', '/processed/label.csv')
# dataset_test = MyDataset('./enhance_processed/CCANet_test/', './enhance_processed/label.csv')
num_test = len(dataset_test)
# print(num_test)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True,
                         drop_last=True)

rmse, mae = 0., 0.
step = 0
paths, labels, predicts = [], [], []
with torch.no_grad():
    loader = tqdm(test_loader)
    for img, label, path in loader:
        paths += list(path)
        labels += torch.flatten(label).tolist()
        img, label = img.to(device), label.to(device).to(torch.float32)
        predict = model(img)
        predicts += torch.flatten(predict).tolist()
        rmse += torch.sqrt(torch.pow(torch.abs(predict - label), 2).mean()).item()
        mae += torch.abs(predict - label).mean().item()
        step += 1
        loader.set_description('step:{} {}/{}'.format(step, step * batch_size, num_test))
    rmse /= step
    mae /= step
print('Test\tMAE:{}\t RMSE:{}'.format(mae, rmse))
# print(len(predicts))
# print(len(paths))
# print(len(labels))
pd.DataFrame({'file': paths, 'label': labels, 'predict': predicts}).to_csv('CCANet_test/testInfo.csv', index=False)
