import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import MyNet_MLP_Improved_GELU_3qubit
from utiles import load_data
import time
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import multiprocessing as mp
from functools import partial

np.random.seed(42)
learningrate = 0.001
batch_size = 128
epochs = 50
num_of_data = 90000  # full:90000

# Define main program parameters
N = 3
nProj_list = np.linspace(start=6 ** N, stop=(6 ** N) / 30, num=10, dtype=int)[:-1]
print("all_start_time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print(nProj_list)
nProj_list = sorted(nProj_list, reverse=True)  # Sort nProj_list in descending order

dictModel = {}
for k in nProj_list:
    dictModel[str(k)] = "./model/3qubit_model/50ModelPauliConcProjR0v7" + str(k)

dictBestModel = {}
for k in nProj_list:
    dictBestModel[str(k)] = "./model/3qubit_model/50bestModelPauliConcProjR0v7" + str(k)

noStates = 90250


# Define function to calculate negative prediction accuracy
def calculate_negative_accuracy(y_test, model_predict):
    # Transfer PyTorch tensors to CPU and convert to NumPy arrays
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()
    if isinstance(model_predict, torch.Tensor):
        model_predict = model_predict.cpu().numpy()

    # Ensure inputs are numpy arrays
    y_test = np.array(y_test)
    model_predict = np.array(model_predict)

    # Find positions of negative values in y_test
    negative_mask = y_test < 0

    # Check if model_predict also predicts negative at these negative positions
    correct_negative_predictions = (model_predict[negative_mask] < 0).sum()

    # Calculate total number of negative regions
    total_negative = negative_mask.sum()

    # Calculate accuracy
    if total_negative == 0:
        return 0.0  # Avoid division by zero

    accuracy = correct_negative_predictions / total_negative
    return accuracy


# Define training and validation function
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, writer, kk):
    best_model = None
    best_mae = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss += criterion(val_outputs, y_val).item()
        avg_val_loss = val_loss / len(val_loader)

        writer.add_scalar(f'Model_{kk}/Train/Loss', avg_train_loss, epoch)
        writer.add_scalar(f'Model_{kk}/Val/Loss', avg_val_loss, epoch)

        if avg_val_loss < best_mae:
            best_mae = avg_val_loss
            best_model = model.state_dict()
    return best_model, best_mae


def train_for_nproj(nProj, device_id, noStates, dictModel, dictBestModel, num_of_data, learningrate, batch_size, epochs,
                    gpu_semaphore):
    gpu_semaphore[device_id].acquire()
    try:
        device = torch.device(f'cuda:{device_id}')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(f'nProj = {nProj} on GPU {device_id}')
        writer = SummaryWriter(log_dir=f'./runs/nProj_{nProj}')

        x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict = load_data(nProj, f'./data/three_qubit_data/nProj{nProj}_train_and_test.npz')
        x_train, x_val, y_train, y_val = x_train[0:100, :], x_val[0:100, :], y_train[0:100, :], y_val[0:100, :]
        x_train, x_val = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], 1, -1), torch.tensor(x_val,
                                                                                                                dtype=torch.float32).view(
            x_val.shape[0], 1, -1)
        y_train, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_model_path = dictModel[str(nProj)] + '.pth'
        best_model = None
        best_mae = float('inf')
        for n in range(5):
            model = MyNet_MLP_Improved_GELU_3qubit(nProj).to(device)
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=learningrate)
            current_model, current_mae = train_model(model, train_loader, val_loader, epochs, criterion, optimizer,
                                                     device, writer, int(nProj / 2))
            if current_mae < best_mae:
                best_mae = current_mae
                best_model = current_model
        torch.save(best_model, best_model_path)
        print(f'Best MAE: {best_mae}')

        x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict = load_data(nProj, noStates)
        x_train, x_val, y_train, y_val = x_train[0:num_of_data, :], x_val[0:num_of_data, :], y_train[0:num_of_data,
                                                                                             :], y_val[0:num_of_data, :]
        x_train, x_val = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], 1, -1), torch.tensor(x_val,
                                                                                                                dtype=torch.float32).view(
            x_val.shape[0], 1, -1)
        y_train, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_model_path_final = dictBestModel[str(nProj)] + '.pth'
        model = MyNet_MLP_Improved_GELU_3qubit(nProj).to(device)
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        # optimizer = optim.Adam(model.parameters(), lr=learningrate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learningrate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        best_val_mae = float('inf')
        criterion = nn.L1Loss()
        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_outputs = model(x_val)
                    val_loss += criterion(val_outputs, y_val).item()
            val_mae = val_loss / len(val_loader)
            scheduler.step(val_mae)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), best_model_path_final)
        print(f'Final Validation MAE: {best_val_mae}')

        x_test = torch.tensor(x_test, dtype=torch.float32).view(x_test.shape[0], 1, -1).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        maxlike_predict = torch.tensor(maxlike_predict, dtype=torch.float32).to(device)
        MLME_predict = torch.tensor(MLME_predict, dtype=torch.float32).to(device)

        model.load_state_dict(torch.load(best_model_path_final, weights_only=True))
        model.eval()
        with torch.no_grad():
            model_predict = model(x_test)
        model_accuracy = calculate_negative_accuracy(y_test, model_predict)
        maxlike_accuracy = calculate_negative_accuracy(y_test, maxlike_predict)
        MLME_accuracy = calculate_negative_accuracy(y_test, MLME_predict)

        mae_criterion = nn.L1Loss(reduction='mean')
        model_mae = mae_criterion(model_predict, y_test).item()
        maxlike_mae = mae_criterion(maxlike_predict, y_test).item()
        MLME_mae = mae_criterion(MLME_predict, y_test).item()

        rmse_criterion = nn.MSELoss(reduction='mean')
        model_rmse = torch.sqrt(rmse_criterion(model_predict, y_test)).item()
        maxlike_rmse = torch.sqrt(rmse_criterion(maxlike_predict, y_test)).item()
        MLME_rmse = torch.sqrt(rmse_criterion(MLME_predict, y_test)).item()

        # Calculate standard error for RMSE (error bar)
        mse_criterion = nn.MSELoss(reduction='none')
        model_mse = mse_criterion(model_predict, y_test).mean(dim=1)
        maxlike_mse = mse_criterion(maxlike_predict, y_test).mean(dim=1)
        MLME_mse = mse_criterion(MLME_predict, y_test).mean(dim=1)
        model_rmse_err_bar = torch.std(torch.sqrt(model_mse)) / np.sqrt(model_mse.shape[0])
        maxlike_rmse_err_bar = torch.std(torch.sqrt(maxlike_mse)) / np.sqrt(maxlike_mse.shape[0])
        MLME_rmse_err_bar = torch.std(torch.sqrt(MLME_mse)) / np.sqrt(MLME_mse.shape[0])

        print(model_rmse, 'model_rmse')
        print(model_rmse_err_bar.item(), 'model_rmse_err_bar')

        result = {
            'nProj': nProj,
            'model_mae': model_mae,
            'maxlike_mae': maxlike_mae,
            'MLME_mae': MLME_mae,
            'model_rmse': model_rmse,
            'maxlike_rmse': maxlike_rmse,
            'MLME_rmse': MLME_rmse,
            'model_rmse_err_bar': model_rmse_err_bar.item(),
            'maxlike_rmse_err_bar': maxlike_rmse_err_bar.item(),
            'MLME_rmse_err_bar': MLME_rmse_err_bar.item(),
            'model_accuracy': model_accuracy,
            'maxlike_accuracy': maxlike_accuracy,
            'MLME_accuracy': MLME_accuracy
        }
        writer.close()
        return result
    finally:
        gpu_semaphore[device_id].release()


# Alternate task assignment for load balancing
gpu_tasks = [[], []]
for i, nProj in enumerate(nProj_list):
    gpu_id = i % 2
    gpu_tasks[gpu_id].append(nProj)

print("GPU 0 tasks:", gpu_tasks[0])
print("GPU 1 tasks:", gpu_tasks[1])

# Create task list, alternating assignment for even initial distribution
tasks = []
for i in range(max(len(gpu_tasks[0]), len(gpu_tasks[1]))):
    if i < len(gpu_tasks[0]):
        tasks.append(
            (gpu_tasks[0][i], 0, noStates, dictModel, dictBestModel, num_of_data, learningrate, batch_size, epochs))
    if i < len(gpu_tasks[1]):
        tasks.append(
            (gpu_tasks[1][i], 1, noStates, dictModel, dictBestModel, num_of_data, learningrate, batch_size, epochs))

print("Initial task allocation:", [(nProj, device_id) for nProj, device_id, _, _, _, _, _, _, _ in tasks[:6]])

# Use Manager to create semaphores, limit concurrent tasks per GPU to 3
manager = mp.Manager()
gpu_semaphore = [manager.Semaphore(3), manager.Semaphore(3)]  # Max 3 tasks per GPU

# Use process pool, limit total parallel processes to 9
results = []
with mp.Pool(processes=9) as pool:
    results = pool.starmap(partial(train_for_nproj, gpu_semaphore=gpu_semaphore), tasks)

results_df = pd.DataFrame(results)
results_df.to_csv('3qubit_model_train_and_test.csv', index=False)
print("all_end_time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))