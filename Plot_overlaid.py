import numpy as np
import os
import matplotlib.pyplot as plt

def plot_npy_file(data_paths, results_paths):
    ECNN_path = os.path.join(data_paths, 'ECNN', '0.2')
    KKthPINN_path = os.path.join(data_paths, 'KKThPINN', '0.2')
    NN_path = os.path.join(data_paths, 'NN', '0.2')
    PINN_path = os.path.join(data_paths, 'PINN', '0.2')

    for file in os.listdir(ECNN_path):
        if 'val_losses' in file:
            continue
        ECNN_curve_npy_path = os.path.join(ECNN_path, file)
        ECNN_graph = np.load(ECNN_curve_npy_path)
        KKthPINN_curve_npy_path = os.path.join(KKthPINN_path, file)
        KKthPINN_graph = np.load(KKthPINN_curve_npy_path)
        NN_curve_npy_path = os.path.join(NN_path, file)
        NN_graph = np.load(NN_curve_npy_path)
        PINN_curve_npy_path = os.path.join(PINN_path, file)
        PINN_graph = np.load(PINN_curve_npy_path)

        ECNN_curve_npy_path_val = ECNN_curve_npy_path.replace('train', 'val')
        KKthPINN_curve_npy_path_val = KKthPINN_curve_npy_path.replace('train', 'val')
        NN_curve_npy_path_val = NN_curve_npy_path.replace('train', 'val')
        PINN_curve_npy_path_val = PINN_curve_npy_path.replace('train', 'val')
        ECNN_graph_val = np.load(ECNN_curve_npy_path_val)
        KKthPINN_graph_val = np.load(KKthPINN_curve_npy_path_val)
        NN_graph_val = np.load(NN_curve_npy_path_val)
        PINN_graph_val = np.load(PINN_curve_npy_path_val)

        result_path = os.path.join(results_paths, file)

        plt.figure()
        plt.plot(np.sqrt(ECNN_graph), label='ECNN')
        plt.plot(np.sqrt(KKthPINN_graph), label='KKthPINN')
        plt.plot(np.sqrt(NN_graph), label='NN')
        plt.plot(np.sqrt(PINN_graph), label='PINN')
        plt.plot(np.sqrt(ECNN_graph_val), label='ECNN_val')
        plt.plot(np.sqrt(KKthPINN_graph_val), label='KKthPINN_val')
        plt.plot(np.sqrt(NN_graph_val), label='NN_val')
        plt.plot(np.sqrt(PINN_graph_val), label='PINN_val')
        
        plt.yscale('log')


        plt.legend()
        plt.title(file)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        out_name = result_path.replace('.npy', '.png')
        out_name = out_name.replace('train_losses', 'losses')
        plt.savefig(out_name.replace('.npy', '.png'))
        plt.close()



            
if __name__ == "__main__":
    data_paths = '/home/hbardool/repos/KKThPINN/data/learning_curves/membrane/' 
    results_paths = '/home/hbardool/repos/KKThPINN/results_overlaid/'
    plot_npy_file(data_paths, results_paths)