import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from utiles import *
from model import MyNet_MLP_Improved_GELU_2qubit

# Set random seed
np.random.seed(52)
noState = 5000

# ===============test======================
d_index = 4
columns = ['nProj', 'MLME_mae', 'maxlike_mae', 'dnn_mae', 'MLME_rmse', 'maxlike_rmse', 'dnn_rmse',
           'MLME_rmse_ci_low', 'MLME_rmse_ci_high', 'maxlike_rmse_ci_low', 'maxlike_rmse_ci_high',
           'dnn_rmse_ci_low', 'dnn_rmse_ci_high', 'MLME_accuracy', 'maxlike_accuracy', 'dnn_accuracy']
results = []
# Determine whether to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def x_derefresh(n_x_data, nProj):
    """
    Restore the original complex-valued projection data from the encoded real/imaginary parts and probListSRM.

    Args:
        n_x_data: Encoded input data (shape: [n_samples, nProj * 33]).
        nProj: Number of projections.

    Returns:
        restored_x_data: Restored complex-valued data (shape: [n_samples, total_length]).
    """
    n_samples = n_x_data.shape[0]  # Number of samples
    restored_x_data = []

    for i in range(n_samples):
        sample = []
        for j in range(nProj):
            # Each projection data length is 33 (16 real + 16 imag + 1 probListSRM)
            start_idx = j * 33
            encode_srmPOMs_input = n_x_data[i, start_idx:start_idx + 33]

            # Extract real part, imaginary part, and probListSRM
            real_part = encode_srmPOMs_input[:16]  # First 16 are real parts
            imag_part = encode_srmPOMs_input[16:32]  # Next 16 are imaginary parts
            probListSRM = encode_srmPOMs_input[32]  # Last one is probListSRM

            # Combine real and imaginary parts into complex numbers
            srmPOMs = real_part + 1j * imag_part
            # Concatenate SRM POM and probListSRM
            proj_data = concatenate([srmPOMs, array([probListSRM])], axis=0)
            sample.append(proj_data)

        # Concatenate all projection data
        restored_x_data.append(concatenate(sample, axis=0))

    return array(restored_x_data)


for nProj in range(2, 38, 2):
    # for nProj in [26]:
    print("nProj =", nProj)
    # Load data
    filename = f"./data/out_of_distribution/cover_test_data_{nProj}_5000_unprove.npz"
    data = np.load(filename, allow_pickle=True)

    test_x, test_y, maxlike_predict, MLME_predict = data['x_test'], data['y_test'], data['maxlike_predict'], data[
        'MLME_predict']
    test_x = x_derefresh(test_x, nProj)
    mask_maxlike = [len(item) == 12 for item in maxlike_predict]  # Mask for length 12
    mask_MLME = [len(item) == 12 for item in MLME_predict]  # Mask for length 12
    mask_y = [len(item) == 12 for item in test_y]

    # Masks must have consistent length
    assert len(mask_maxlike) == len(mask_MLME) == len(mask_y), "Masks do not match in length!"

    # Ensure both masks are consistent, i.e., elements have length 12
    final_mask = np.array(mask_maxlike) & np.array(mask_MLME) & np.array(mask_y)
    test_x = test_x[final_mask]
    test_y = test_y[final_mask]
    maxlike_predict = np.array(maxlike_predict[final_mask].tolist(), dtype=np.float64)
    MLME_predict = np.array(MLME_predict[final_mask].tolist(), dtype=np.float64)

    # Filter out outliers using IQR on mean absolute errors per sample
    maxlike_errors = np.mean(np.abs(maxlike_predict - test_y), axis=1)  # Shape: (N,)
    q1_maxlike = np.percentile(maxlike_errors, 25)
    q3_maxlike = np.percentile(maxlike_errors, 75)
    iqr_maxlike = q3_maxlike - q1_maxlike
    lower_maxlike = q1_maxlike - 3 * iqr_maxlike
    upper_maxlike = q3_maxlike + 3 * iqr_maxlike
    good_maxlike = (maxlike_errors >= lower_maxlike) & (maxlike_errors <= upper_maxlike)  # Shape: (N,)

    MLME_errors = np.mean(np.abs(MLME_predict - test_y), axis=1)  # Shape: (N,)
    q1_MLME = np.percentile(MLME_errors, 25)
    q3_MLME = np.percentile(MLME_errors, 75)
    iqr_MLME = q3_MLME - q1_MLME
    lower_MLME = q1_MLME - 3 * iqr_MLME
    upper_MLME = q3_MLME + 3 * iqr_MLME
    good_MLME = (MLME_errors >= lower_MLME) & (MLME_errors <= upper_MLME)  # Shape: (N,)

    # Use intersection to filter common good samples
    good_indices = good_maxlike & good_MLME  # Shape: (N,)

    # Apply filter to all data
    test_x = test_x[good_indices]
    test_y = test_y[good_indices]
    maxlike_predict = maxlike_predict[good_indices]
    MLME_predict = MLME_predict[good_indices]

    print(f'After filtering, remaining samples: {len(test_y)}')

    # Convert to torch tensors and move to GPU (if available)
    test_x = torch.tensor(test_x).view(test_x.shape[0], 1, -1).float().to(device)
    test_y = torch.tensor(test_y).float().to(device)
    maxlike_predict = torch.tensor(maxlike_predict, dtype=torch.float32).to(device)
    MLME_predict = torch.tensor(MLME_predict, dtype=torch.float32).to(device)

    model = MyNet_MLP_Improved_GELU_2qubit(nProj).to(device)  # Move model to GPU (if available)
    best_model_path = "./model/2qubit_model_cover/50bestModelPauliConcProjR0v7" + str(nProj) + ".pth"

    # Load model based on device
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(best_model_path, weights_only=False))  # Load parameters to GPU
    else:
        model.load_state_dict(
            torch.load(best_model_path, weights_only=False, map_location=torch.device('cpu')))  # Load to CPU

    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        outputs = model(test_x)

    # Compute metrics
    mae_criterion = nn.L1Loss(reduction='mean')
    dnn_mae = mae_criterion(outputs, test_y).item()
    maxlike_mae = mae_criterion(maxlike_predict, test_y).item()
    MLME_mae = mae_criterion(MLME_predict, test_y).item()

    rmse_criterion = nn.MSELoss(reduction='mean')
    dnn_rmse = torch.sqrt(rmse_criterion(outputs, test_y)).item()
    maxlike_rmse = torch.sqrt(rmse_criterion(maxlike_predict, test_y)).item()
    MLME_rmse = torch.sqrt(rmse_criterion(MLME_predict, test_y)).item()

    # Use Bootstrap CI
    MLME_rmse_boot, MLME_ci_low, MLME_ci_high = bootstrap_rmse(test_y, MLME_predict)
    maxlike_rmse_boot, maxlike_ci_low, maxlike_ci_high = bootstrap_rmse(test_y, maxlike_predict)
    dnn_rmse_boot, dnn_ci_low, dnn_ci_high = bootstrap_rmse(test_y, outputs)

    # Use bootstrap mean to replace original RMSE (optional; for consistency with CI)
    MLME_rmse = MLME_rmse_boot
    maxlike_rmse = maxlike_rmse_boot
    dnn_rmse = dnn_rmse_boot

    # Original style RMSE and err_bar after IQR filtering
    rmse_criterion = nn.MSELoss(reduction='mean')
    model_rmse_original = torch.sqrt(rmse_criterion(outputs, test_y)).item()
    maxlike_rmse_original = torch.sqrt(rmse_criterion(maxlike_predict, test_y)).item()
    MLME_rmse_original = torch.sqrt(rmse_criterion(MLME_predict, test_y)).item()

    mse_criterion = nn.MSELoss(reduction='none')
    model_mse = mse_criterion(outputs, test_y).mean(dim=1)
    maxlike_mse = mse_criterion(maxlike_predict, test_y).mean(dim=1)
    MLME_mse = mse_criterion(MLME_predict, test_y).mean(dim=1)

    model_rmse_err_bar = torch.std(torch.sqrt(model_mse)) / np.sqrt(model_mse.shape[0])
    maxlike_rmse_err_bar = torch.std(torch.sqrt(maxlike_mse)) / np.sqrt(maxlike_mse.shape[0])
    MLME_rmse_err_bar = torch.std(torch.sqrt(MLME_mse)) / np.sqrt(MLME_mse.shape[0])

    # Accuracy (assuming calculate_negative_accuracy function is defined)
    try:
        dnn_accuracy = calculate_negative_accuracy(test_y, outputs)
        maxlike_accuracy = calculate_negative_accuracy(test_y, maxlike_predict)
        MLME_accuracy = calculate_negative_accuracy(test_y, MLME_predict)
    except NameError:
        # If function not defined, set to None or 0
        dnn_accuracy = maxlike_accuracy = MLME_accuracy = 0.0
        print("Warning: calculate_negative_accuracy not defined, accuracies set to 0.")

    print(f"DNN Test MAE: {dnn_mae}, RMSE: {dnn_rmse} [{dnn_ci_low}, {dnn_ci_high}]")
    print(f"maxlike Test MAE: {maxlike_mae}, RMSE: {maxlike_rmse} [{maxlike_ci_low}, {maxlike_ci_high}]")
    print(f"MLME Test MAE: {MLME_mae}, RMSE: {MLME_rmse} [{MLME_ci_low}, {MLME_ci_high}]")

    results.append({
        'nProj': nProj,
        'MLME_mae': MLME_mae,
        'maxlike_mae': maxlike_mae,
        'dnn_mae': dnn_mae,
        'MLME_rmse': MLME_rmse,
        'maxlike_rmse': maxlike_rmse,
        'dnn_rmse': dnn_rmse,
        'model_rmse_original': model_rmse_original,
        'maxlike_rmse_original': maxlike_rmse_original,
        'MLME_rmse_original': MLME_rmse_original,
        'model_rmse_err_bar': model_rmse_err_bar.item(),
        'maxlike_rmse_err_bar': maxlike_rmse_err_bar.item(),
        'MLME_rmse_err_bar': MLME_rmse_err_bar.item(),
        'MLME_rmse_ci_low': MLME_ci_low,
        'MLME_rmse_ci_high': MLME_ci_high,
        'maxlike_rmse_ci_low': maxlike_ci_low,
        'maxlike_rmse_ci_high': maxlike_ci_high,
        'dnn_rmse_ci_low': dnn_ci_low,
        'dnn_rmse_ci_high': dnn_ci_high,
        'MLME_accuracy': MLME_accuracy,
        'maxlike_accuracy': maxlike_accuracy,
        'dnn_accuracy': dnn_accuracy
    })

# Save losses to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('werner_test_loss.csv', index=False)
print("Results have been saved to 'werner_test_loss.csv'")