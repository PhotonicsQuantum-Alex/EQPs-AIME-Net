import torch
import os
from os.path import join
import numba
from numba import jit, njit, objmode
from numpy import *
from numpy import linalg as la
from scipy import linalg as sa
import smtplib
from sklearn.model_selection import train_test_split
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.utils import resample


def refresh_data(maxlike_predict, MLME_predict, x_test, y_test, expected_length=12):
    """Filter indices in maxlike_predict and MLME_predict where length does not match expected_length,
    and synchronously delete corresponding data in maxlike_predict, MLME_predict, x_test, and y_test.

    Parameters:
    maxlike_predict: sequence containing prediction data (expected each element length is expected_length)
    MLME_predict: sequence containing MLME prediction data (expected each element length is expected_length)
    x_test: test input data, corresponding one-to-one with maxlike_predict
    y_test: test label data, corresponding one-to-one with maxlike_predict
    expected_length: expected length of each element (default 12)

    Returns:
    Cleaned maxlike_predict, MLME_predict, x_test, y_test (all NumPy arrays)
    """
    invalid_indices = []
    # Check lengths in maxlike_predict
    for i, item in enumerate(maxlike_predict):
        if len(item) != expected_length:
            print(f"Inconsistent length in maxlike_predict at index {i}: {len(item)}")
            invalid_indices.append(i)
    # Check lengths in MLME_predict
    for i, item in enumerate(MLME_predict):
        if len(item) != expected_length:
            print(f"Inconsistent length in MLME_predict at index {i}: {len(item)}")
            if i not in invalid_indices:
                invalid_indices.append(i)
    # If there are invalid indices, remove corresponding data
    if invalid_indices:
        print(f"Removing {len(invalid_indices)} invalid indices: {invalid_indices}")
        valid_indices = [i for i in range(len(maxlike_predict)) if i not in invalid_indices]
        # Filter valid data
        maxlike_predict = [maxlike_predict[i] for i in valid_indices]
        MLME_predict = [MLME_predict[i] for i in valid_indices]
        x_test = x_test[valid_indices]
        y_test = y_test[valid_indices]
        # Convert to NumPy arrays and ensure data type is float32
        try:
            maxlike_predict = array(maxlike_predict, dtype=float32)
            MLME_predict = array(MLME_predict, dtype=float32)
        except Exception as e:
            print(f"Error converting to array: {e}")
            raise
    print(f"Shape of maxlike_predict after cleaning: {maxlike_predict.shape}")
    print(f"Shape of MLME_predict after cleaning: {MLME_predict.shape}")
    print(f"Shape of x_test after cleaning: {x_test.shape}")
    print(f"Shape of y_test after cleaning: {y_test.shape}")
    return maxlike_predict, MLME_predict, x_test, y_test


def x_rechange(x_data, nProj):
    n_x_data = []
    for i in range(x_data.shape[0]):
        input = []
        for j in range(nProj):
            srmPOMs = x_data[i, j * 17: j * 17 + 16]  # Get SRM POMs
            probListSRM = real(x_data[i, j * 17 + 16])  # Get real part of probListSRM
            # Concatenate real and imaginary parts together and append probListSRM
            # encode_srmPOMs_input = np.append(np.concatenate([np.real(srmPOMs),np.imag(srmPOMs)]),probListSRM)
            encode_srmPOMs_input = concatenate([real(srmPOMs), imag(srmPOMs), probListSRM.reshape(1)], axis=0)
            input.append(encode_srmPOMs_input)
        n_x_data.append(concatenate(input, axis=0))  # Concatenate all samples
    return array(n_x_data)


def randomHaarState(dim, rank):
    A = random.normal(0, 1, (dim, dim)) + 1j * random.normal(0, 1, (dim, dim))
    q, r = la.qr(A, mode='complete')
    r = divide(diagonal(r), abs(diagonal(r))) * identity(dim)
    rU = q @ r
    B = random.normal(0, 1, (dim, rank)) + 1j * random.normal(0, 1, (dim, rank))
    B = B @ B.T.conj()
    rho = (identity(dim) + rU) @ B @ (identity(dim) + rU.T.conj())
    return rho / trace(rho)


def randompure(dim, n):
    rpure = random.normal(0, 1, [dim, n]) + 1j * random.normal(0, 1, [dim, n])
    rpure = rpure / la.norm(rpure, axis=0)
    rhon = array([dot(rpure[:, [i]], rpure[:, [i]].conjugate().transpose()) for i in range(n)])
    # rhon = reshape(rhon,[n,4])
    return rhon


def mubpom():
    p1 = array([1, 0])
    p2 = array([0, 1])
    mub = zeros([6, 1, 2]) + 1j * zeros([6, 1, 2])
    mub[0] = p1
    mub[1] = p2
    mub[2] = 1 / sqrt(2) * (p1 + p2)
    mub[3] = 1 / sqrt(2) * (p1 - p2)
    mub[4] = 1 / sqrt(2) * (p1 + 1j * p2)
    mub[5] = 1 / sqrt(2) * (p1 - 1j * p2)
    mubp = [transpose(mub[i]) @ conjugate(mub[i]) for i in range(6)]
    return mubp


def mubpom1():
    p1 = array([1, 0])
    p2 = array([0, 1])
    mub = zeros([4, 1, 2]) + 1j * zeros([4, 1, 2])
    mub[0] = p1
    mub[1] = p2
    mub[2] = 1 / sqrt(2) * (p1 + p2)
    mub[3] = 1 / sqrt(2) * (p1 + 1j * p2)
    mubp = [transpose(mub[i]) @ conjugate(mub[i]) for i in range(4)]
    return mubp


def blochFromRho(rho, A):
    l = shape(A)[0]
    return array([real(trace(rho @ A[n])) for n in range(l)])


"""get probabilities from quantum state rho0 and POVM"""


def probdists(rho0, povm):
    l = shape(povm)[0]
    probtrue = array([real(trace(rho0 @ povm[i])) for i in range(l)])
    epsilon = 1e-10
    if sum(probtrue) < epsilon:
        return ones(l) / l
    else:
        probtrue = probtrue / sum(probtrue)
        return probtrue


def herbasis(dim):
    pom1 = zeros([1, dim, dim]) + 1j * zeros([1, dim, dim])
    pom1[0] = identity(dim)
    arrays = [dot(transpose(pom1[0][[i]]), pom1[0][[i]]) for i in range(dim - 1)]
    pom = stack(arrays, axis=0)
    her = concatenate((pom1, pom), axis=0)
    arrays = [dot(transpose(her[0][[i]]), her[0][[j]]) + dot(transpose(her[0][[j]]), her[0][[i]]) for i in range(dim)
              for j in range(i + 1, dim)]
    pom = stack(arrays, axis=0)
    her = concatenate((her, pom), axis=0)
    arrays = [-1j * dot(transpose(her[0][[i]]), her[0][[j]]) + 1j * dot(transpose(her[0][[j]]), her[0][[i]]) for i in
              range(dim) for j in range(i + 1, dim)]
    pom = stack(arrays, axis=0)
    pom = concatenate((her, pom), axis=0)
    return pom


def gellmann(Q, dim):
    q = zeros([dim ** 2, dim, dim]) + 1j * zeros([dim ** 2, dim, dim])
    for i in range(dim ** 2):
        v = Q[i]
        for j in range(0, i):
            v = v - trace(v @ q[j]) * q[j]
        q[i] = v / sqrt(trace(v @ v))
    return q


def pauli():
    s = zeros([3, 2, 2]) + 1j * zeros([3, 2, 2])
    s[0] = array([[1, 0], [0, -1]])
    s[1] = array([[0, 1], [1, 0]])
    s[2] = array([[0, -1j], [1j, 0]])
    return s


# 2-qubit load training data
def localdataload8_cover(nProj, data_dir='./data/two_qubit_data'):
    def x_derefresh(n_x_data, nProj):
        n_samples = n_x_data.shape[0]  # Number of samples
        restored_x_data = []
        for i in range(n_samples):
            sample = []
            for j in range(nProj):
                # Each projection data length is 33 (16 real + 16 imag + 1 probListSRM)
                start_idx = j * 33
                encode_srmPOMs_input = n_x_data[i, start_idx:start_idx + 33]
                # Extract real part, imag part, and probListSRM
                real_part = encode_srmPOMs_input[:16]  # First 16 are real part
                imag_part = encode_srmPOMs_input[16:32]  # Next 16 are imag part
                probListSRM = encode_srmPOMs_input[32]  # Last one is probListSRM
                # Combine real and imag parts into complex
                srmPOMs = real_part + 1j * imag_part
                # Concatenate SRM POM and probListSRM
                proj_data = concatenate([srmPOMs, array([probListSRM])], axis=0)
                sample.append(proj_data)
            # Concatenate all projection data
            restored_x_data.append(concatenate(sample, axis=0))
        return array(restored_x_data)

    """Load data_model_input_based_on_4 data"""
    data_path = join(data_dir, f'cover_test_data_{nProj}_500000.npz')
    data = load(data_path, allow_pickle=True)
    # Extract training, validation, test data and their corresponding labels
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    x_test = data['x_test']
    y_test = data['y_test']
    maxlike_predict = array(data['maxlike_predict'], dtype=float32)
    MLME_predict = array(data['MLME_predict'], dtype=float32)
    x_train = x_derefresh(x_train, nProj)
    x_val = x_derefresh(x_val, nProj)
    x_test = x_derefresh(x_test, nProj)
    # Return all data
    return x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict


# 3-qubit load training data
def load_data(nProj, data_path):
    def x_derefresh(n_x_data, nProj):
        n_samples = n_x_data.shape[0]  # Number of samples
        restored_x_data = []
        for i in range(n_samples):
            sample = []
            for j in range(nProj):
                # Each projection data length is 129 (64 real + 64 imag + 1 probListSRM)
                start_idx = j * (129)
                encode_srmPOMs_input = n_x_data[i, start_idx:start_idx + 129]
                # Extract real part, imag part, and probListSRM
                real_part = encode_srmPOMs_input[:64]  # First 64 are real part
                imag_part = encode_srmPOMs_input[64:128]  # Next 64 are imag part
                probListSRM = encode_srmPOMs_input[128]  # Last one is probListSRM
                # Combine real and imag parts into complex
                srmPOMs = real_part + 1j * imag_part
                # Concatenate SRM POM and probListSRM
                proj_data = concatenate([srmPOMs, array([probListSRM])], axis=0)
                sample.append(proj_data)
            # Concatenate all projection data
            restored_x_data.append(concatenate(sample, axis=0))
        return array(restored_x_data)

    # data_path = f'./N_qubit_data_generatino/3_qubits_data_step_size_0.03/nProj{nProj}_train_and_test.npz'
    data = load(data_path, allow_pickle=True)
    # Extract training, validation, test data and their corresponding labels
    x_train_and_val = data['x_train']
    y_train_and_val = data['quasi_present_train']
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_and_val, y_train_and_val, test_size=0.2,  # Validation set 20%
        random_state=42  # Set random seed for reproducibility
    )
    x_test = data['x_test'][0:4000, :]
    y_test = data['quasi_present_test'][0:4000, :]
    maxlike_predict = data['maxlike_predict'][0:4000, :]
    MLME_predict = data['MLME_predict'][0:4000, :]
    # If it is de, use this
    x_train = x_derefresh(x_train, nProj)
    x_val = x_derefresh(x_val, nProj)
    x_test = x_derefresh(x_test, nProj)
    return x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict


# Exp data load
def localdataload_exp_cover(nProj, noStates):
    """Load .npz file and clean data.

    Parameters:
    nProj: number of projections
    noStates: number of states (default 5000)

    Returns:
    x_test, y_test, maxlike_predict, MLME_predict (cleaned data)
    """
    data_path = f'./data/real_data/exp2x_target_{noStates}_cover_{nProj}.npz'
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    # Load data
    try:
        data = load(data_path, allow_pickle=True)
        print(f"Loaded file: {data_path}")
        print(f"Keys in npz: {data.files}")
    except Exception as e:
        raise Exception(f"Failed to load {data_path}: {e}")

    def x_derefresh(n_x_data, nProj):
        n_samples = n_x_data.shape[0]  # Number of samples
        restored_x_data = []
        for i in range(n_samples):
            sample = []
            for j in range(nProj):
                # Each projection data length is 33 (16 real + 16 imag + 1 probListSRM)
                start_idx = j * 33
                encode_srmPOMs_input = n_x_data[i, start_idx:start_idx + 33]
                # Extract real part, imag part, and probListSRM
                real_part = encode_srmPOMs_input[:16]  # First 16 are real part
                imag_part = encode_srmPOMs_input[16:32]  # Next 16 are imag part
                probListSRM = encode_srmPOMs_input[32]  # Last one is probListSRM
                # Combine real and imag parts into complex
                srmPOMs = real_part + 1j * imag_part
                # Concatenate SRM POM and probListSRM
                proj_data = concatenate([srmPOMs, array([probListSRM])], axis=0)
                sample.append(proj_data)
            # Concatenate all projection data
            restored_x_data.append(concatenate(sample, axis=0))
        return array(restored_x_data)

    # Extract data
    x_test = data['x_test']
    y_test = data['y1']
    maxlike_predict = data['maxlike_predict']
    MLME_predict = data['MLME_predict']
    # Clean data
    maxlike_predict, MLME_predict, x_test, y_test = refresh_data(
        maxlike_predict, MLME_predict, x_test, y_test
    )
    x_test = x_derefresh(x_test, nProj)
    return array(x_test, dtype=float32), array(y_test, dtype=float32), array(maxlike_predict, dtype=float32), array(
        MLME_predict, dtype=float32)


def bootstrap_rmse(y_true, y_pred, n_bootstrap=1000, ci=95):
    """Compute Bootstrap confidence interval for RMSE.

    y_true, y_pred: torch.Tensor, shape (N, dim)

    Returns: mean_rmse, ci_low, ci_high
    """
    rmses = []
    N = y_true.shape[0]
    mse_crit = nn.MSELoss(reduction='none')
    for _ in range(n_bootstrap):
        idx = resample(range(N), replace=True, random_state=random.randint(0, 10000))
        y_boot = y_true[idx]
        pred_boot = y_pred[idx]
        mse_boot = mse_crit(pred_boot, y_boot).mean()
        rmse_boot = torch.sqrt(mse_boot)
        rmses.append(rmse_boot.item())
    rmses = array(rmses)
    alpha = (100 - ci) / 2
    ci_low = percentile(rmses, alpha)
    ci_high = percentile(rmses, 100 - alpha)
    mean_rmse = mean(rmses)
    return mean_rmse, ci_low, ci_high


def calculate_negative_accuracy(y_test, model_predict):
    # Transfer PyTorch tensors to CPU and convert to NumPy arrays
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()
    if isinstance(model_predict, torch.Tensor):
        model_predict = model_predict.cpu().numpy()
    # Ensure inputs are numpy arrays
    y_test = array(y_test)
    model_predict = array(model_predict)
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