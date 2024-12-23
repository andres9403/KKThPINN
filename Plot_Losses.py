import numpy as np
import matplotlib.pyplot as plt

Method = ['NN', 'PINN', 'KKThPINN']
Losses = ['None_train_losses_', 'None_val_losses_']
Violations = ['None_train_violations_', 'None_val_violations_']

for i in Method:
    for j in Losses:
        file_path = f'/home/andresfel9403/KKThNN/data/learning_curves/membrane/{i}/0.2/{j}run9.npy'
        data = np.load(file_path)
        if j == 'None_train_losses_':
            plt.plot(data, label = f'train {i}')
        else:
            plt.plot(data, label = f'val {i}')

plt.title("Losses")
plt.figure(1)
plt.xlabel("Epoch")
plt.ylabel("RMSE")
#plt.yscale("log") 
plt.xlim(600, 1000)
plt.ylim(0, 0.02)
plt.legend()
plt.show()



# years = 10
# amount = np.empty(years + 1)
# for i in interest_rates:
#     amount[0] = 100
#     for year in range(years):
#         amount[year + 1] = amount[year]*(1 + i)
#     plt.plot(amount, label = f'$\\alpha = {i}$')
# plt.legend()
# plt.show()


# if __name__ == "__main__":
#     file_path1 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_losses_run1.npy' 
#     file_path2 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/PINN/0.2/None_train_violations_run0.npy'
#     file_path4= '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_violations_run0.npy'
#     file_path3 = '/home/andresfel9403/KKThNN/data/learning_curves/membrane/KKThPINN/0.2/None_train_violations_run2.npy' # Replace with your npy file path
#     plot_npy_file(file_path2)