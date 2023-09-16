from CCANet_train.CCANet_train import CCA101

import torch
import torch.nn as nn
from CCANet_train.validate import validate
from tqdm import tqdm


def train(train_loader, test_loader, writer, epochs, lr, device, model_dict):
    best_l = 1000
    # model = ResNet18().to(device)
    # model = ResNet34().to(device)# ResNet34
    # model = ResNet50().to(device)# ResNet50
    # model = ResNet50_Attention().to(device)# ResNet50
    # model = ResNet50_Attention_LSTM().to(device)# ResNet50
    # model = ResNet50_eca().to(device)# ResNet50_eca
    # model = eca_resnet101().to(device)# ResNet50_eca
    model = CCA101().to(device)# ResNet50_eca
    # The initialization of the optimizer, model.parameters, is the method used to view network parameters, where lr is the learning rate
    optimizer_e = torch.optim.Adam(model.parameters(), lr=lr)
    # Create a standard for measuring mean square error
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    for epoch in range(epochs):
        # Set the module to training mode
        model.train()
        train_rmes, train_mae, train_loss = 0., 0., 0.
        step = 0
        loader = tqdm(train_loader)
        # print(loader)
        for img, label, _ in loader:  # Image, label, path
            # print(dataset_img.shape)  # torch.Size([128, 3, 128, 128])
            img, label = img.to(device), label.to(device).to(torch.float32)
            # Set the gradient to zero, which means changing the derivative of loss with respect to weight to 0
            optimizer_e.zero_grad()
            # dataset_img = dataset_img.view(-1, 22, 14)
            # dataset_img = dataset_img.reshape((dataset_img.shape[0], 1, dataset_img.shape[1]))
            score = model(img)
            # print(score.shape)  #[128, 1]
            # Calculate the mean square error according to the standard formula
            loss = criterion(score, label)
            # print(loss)
            # Directly obtaining the corresponding Python data type results in training loss
            train_loss += loss.item()
            # print(train_loss)
            rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
            # print(rmse)
            train_rmes += rmse
            mae = torch.abs(score - label).mean().item()
            train_mae += mae
            # The gradient value of each parameter obtained from backpropagation calculation
            loss.backward()
            optimizer_e.step()
            step += 1
            loader.set_description("Epoch:{} Step:{} RMSE:{:.2f} MAE:{:.2f}".format(epoch, step, rmse, mae))
        train_rmes /= step
        train_mae /= step
        train_loss /= step
        # Enable evaluation mode, in which the batchNorm layer, dropout layer,
        # and other network layers added for optimizing training will be turned off,
        # so that there will be no deviation during evaluation
        model.eval()
        val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion)
        writer.log_train(train_rmes, train_mae, train_loss, val_rmes, val_mae, val_loss, epoch)# Write Log
        if val_loss < best_l:
            torch.save({'ResNet': model.state_dict()}, '{}/CCANet.pth'.format(model_dict))
            print('Save model!,Loss Improve:{:.2f}'.format(best_l - val_loss))
            best_l = val_loss
        print('Train RMSE:{:.2f} MAE:{:.2f} \t Val RMSE:{:.2f} MAE:{:.2f}'.format(train_rmes, train_mae, val_rmes, val_mae))
