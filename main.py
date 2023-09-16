from CCANet_train.writer import MyWriter
import os
from data_preprocessing.dataset import MyDataset
from torch.utils.data import DataLoader
from train import train
import torch

batch_size = 64
lr = 0.0001
epochs = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Storage location of trained model parameter files
model_dir = 'model_dict'
# TensorboardLog file storage location
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# Create Tensorboard recorder to save training process losses
writer = MyWriter(log_dir)

# Convert data into processable iterators, image paths, and label paths
dataset_train = MyDataset('processed/train/', 'processed/label.csv')
# dataset_train = MyDataset('enhance_processed/CCANet_train/', 'enhance_processed/label.csv')
# Data loading, shuffle set to True to shuffle data, randomly select, pin_ Memory

train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                        drop_last=True)


dataset_test = MyDataset('processed/validate/', 'processed/label.csv')
# dataset_test = MyDataset('enhance_processed/validate/', 'enhance_processed/label.csv')
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

# started training
train(train_loader, test_loader, writer, epochs, lr, device, model_dir)