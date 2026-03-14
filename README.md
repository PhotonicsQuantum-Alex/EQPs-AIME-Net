# Project Structure

This README provides an overview of the project's file organization, including experiment scripts, data directories, and utility files. The project focuses on quantum machine learning models for 2- and 3-qubit systems, including training, testing, and evaluation on simulated and real experimental data.

The complete code, along with the data and execution files, is available at: https://www.modelscope.cn/datasets/PhotonicsQuantumAlex/EQPs-AIME-Net

## 1. Experiment Scripts

These scripts handle training, testing, and evaluation for various qubit configurations.

1. **2qubit_model_train_and_test.py**  
   Training and testing script for the 2-qubit model.

2. **out_of_distribution_train_test.py**  
   Testing script for Werner states outside the training distribution.

3. **3qubit_model_train_and_test.py**  
   Training and testing script for the 3-qubit model.

4. **experiment_data_test.py**  
   Testing on real experimental data.

5. **3qubit_test_F_and_pure.py**  
   Computation of fidelity (F) and purity for reconstructed EQPs.

## 2. Experiments and Data

### Data Directory (`/data`)
- `/data/2qubit/`: Simulated data for 2-qubit experiments.  
- `/data/3qubit/`: Simulated data for 3-qubit experiments.  
- `/data/real_data/`: Real-world experimental datasets.

### Model Directory (`/model`)
- `/model/2qubit/`: Storage path for trained 2-qubit models.  
- `/model/3qubit/`: Storage path for trained 3-qubit models.

### Training Logs Directory (`/runs`)
- Contains training metrics, logs, and checkpoints generated during model training.

## 3. Utility Scripts

These core scripts provide shared functionality across experiments.

1. **model.py**  
   Defines model architectures for various system sizes (e.g., 2-qubit and 3-qubit).

2. **utils.py**  
   Collection of utility functions for data processing, visualization, and helpers.

## Quick Start

To run experiments:  
1. Install dependencies: `pip install -r requirements.txt`.  
2. Execute a script, e.g., `python 2qubit_model_train_and_test.py`.  

For more details, refer to inline comments in each script. If you encounter issues, check the `/runs` directory for logs.