import numpy as np
import os
import matplotlib.pyplot as plt

def plot_npy_file(data_paths, results_paths):

    architectures_list = []
    for folder in os.listdir(data_paths):
        architectures_list.append(folder)
    for file in os.listdir(data_paths + architectures_list[0] + '/0.2'):
        if 'val_losses' in file:
            continue
        
        plt.figure()

        for architecture in architectures_list:
            architecture_path = os.path.join(data_paths, architecture)
            curve_npy_path = os.path.join(architecture_path + '/0.2', file) 
            graph = np.load(curve_npy_path)
            curve_npy_path_val = curve_npy_path.replace('train', 'val')
            graph_val = np.load(curve_npy_path_val)

            plt.plot(np.sqrt(graph), label=architecture)
            plt.plot(np.sqrt(graph_val), label=architecture + '_val')
            
        plt.yscale('log')
        plt.legend()
        plt.title(file)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        result_path = os.path.join(results_paths, file)
        out_name = result_path.replace('.npy', '.png')
        out_name = out_name.replace('train_losses', 'losses')
        plt.savefig(out_name.replace('.npy', '.png'))
        plt.close()



            
if __name__ == "__main__":
    data_paths = '/home/hbardool/repos/KKThPINN/data/learning_curves/membrane/' 
    results_paths = '/home/hbardool/repos/KKThPINN/results_overlaid_clean/'
    plot_npy_file(data_paths, results_paths)