import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 12  # Optional: Adjust tick label size
rcParams['ytick.labelsize'] = 12  # Optional: Adjust tick label size

rcParams['font.size'] = 12

Method = ['NN', 'PINN', 'KKThPINN']
Violations = ['None_train_violations_', 'None_val_violations_']


for i in Method:
    for j in Violations:
        file_path = f'/home/hbardool/repos/KKThPINN/data/learning_curves/PFR/{i}/0.2/{j}run0.npy'
        data = np.load(file_path)
        if j == 'None_train_violations_':
            plt.plot(data, label = f'train {i}')
        else:
            plt.plot(data, label = f'val {i}')

f1 = plt.figure(1)
plt.title("Violations")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.yscale("log") 
plt.xlim(0, 1000)
plt.ylim(0, 20)
plt.legend()


plt.figure(2)
plt.title("Violations")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.yscale("log") 
plt.xlim(600, 1000)
plt.ylim(0, 1)
plt.legend()

plt.show()

