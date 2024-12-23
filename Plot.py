import numpy as np

import matplotlib.pyplot as plt

def plot_npy_file(file_path):
    data = np.load(file_path)
    
    plt.figure()
    plt.plot(data)
    plt.title('Plot from npy file')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path1 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_losses_run1.npy' 
    file_path2 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/PINN/0.2/None_train_violations_run0.npy'
    file_path4= '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_violations_run0.npy'
    file_path3 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_violations_run2.npy' # Replace with your npy file path
    plot_npy_file(file_path2)