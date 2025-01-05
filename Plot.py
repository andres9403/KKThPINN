import numpy as np
import os
import matplotlib.pyplot as plt

def plot_npy_file(file_paths, save_paths):
    for file in os.listdir(file_paths):
        file_path = os.path.join(file_paths, file)
        save_path = os.path.join(save_paths, file)
        if file_path.endswith('.npy'):
            data = np.load(file_path)
            
            plt.figure()
            plt.plot(data)
            plt.title(file)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.grid(True)
            plt.savefig(save_path.replace('.npy', '.png'))
            plt.close()

if __name__ == "__main__":
    file_path1 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_losses_run1.npy' 
    file_path2 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/NN/0.2/None_train_violations_run0.npy'
    file_path4= '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_violations_run0.npy'
    file_path3 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_violations_run2.npy' # Replace with your npy file path
    file_path5= '/home/hbardool/repos/KKThPINN/data/learning_curves/membrane/KKThPINN/0.2/None_train_violations_run0.npy'
    file_paths = '/home/hbardool/repos/KKThPINN/data/learning_curves/membrane/NN/0.2/'
    result_paths = '/home/hbardool/repos/KKThPINN/results_individual/'
    plot_npy_file(file_paths, result_paths)