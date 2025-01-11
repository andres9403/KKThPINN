from utils import LoadData, LoadModel
from train import run_training
import torch
import argparse
import time
import copy
import os


def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='NN', help='NN, PINN, KKThPINN')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--input_dim', type=int, default=7, help='3 for cstr, 4 for plant, 5 for distillation, 7 for Membrane')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--z0_dim', type=int, default=8, help='3 for cstr, 5 for plant, 8, for membrane 10 for distillation')

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument("--max_subiter", default=500, type=int)
    parser.add_argument("--eta", default=0.8, type=float)
    parser.add_argument("--sigma", default=2, type=float)
    parser.add_argument("--mu_safe", default=1e+9, type=float)
    parser.add_argument("--dtype", default=32, type=int)

    parser.add_argument('--dataset_type', type=str, default='membrane', help='choose from cstr, plant, distillation')
    parser.add_argument('--dataset_path', default='/home/andresfel9403/KKThNN/KKThPINN/benchmark_membrane.csv', type=str)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--job', type=str, default='prediction', help='choose from train, experiment')
    parser.add_argument('--runs', type=int, default=10)

    args = parser.parse_args()
    return args


def main(args):
    if args.job == 'prediction':
        if args.model == 'NN':
            args.loss_type = 'MSE'
        elif args.model == 'PINN':
            args.loss_type = 'PINN'
        elif args.model == 'KKThPINN':
            args.loss_type = 'MSE'
        elif args.model == 'AugLagNN':
            args.loss_type = 'MSE'
        elif args.model == 'ECNN':
            args.loss_type = 'MSE'
        
        data = {}
        args.run = 0
        data = LoadData(args)
        model = LoadModel(args, data)
        PATH = '/home/andresfel9403/KKThNN/models/membrane/NN/0.2/None_0.2_9.pth'
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # Predict

        for key, value in data.items() :
            print (key, value)

        N = 10    
        dataiter = iter(data['val_loader'])

        image_list = []
        label_list = []
        #assume batch size equal to 1, otherwise divide N by batch size
        for i in range(0, N): 
            image, label = next(dataiter)
            y_pred = model(image)
            print(f'x = {image[0]}')
            print(f'y = {label[0]}')
            print(f'y_pred = {y_pred[0]}')
            image_list.append(image)
            label_list.append(label)



        # x = torch.tensor(data['x_val'], dtype=torch.float32)
        # y = torch.tensor(data['y_val'], dtype=torch.float32)
        # x = x.to(model.device)
        # y = y.to(model.device)

        # with torch.no_grad():
        #     y_pred = model(x)
        #     loss = model.loss(y_pred, y)
        #     print(f'Validation Loss: {loss.item()}')
        #     for i in range(args.input_dim):
        #         print(f'x_{i} = {x[0, i]}')
        #     print(f'y = {y[0]}')
        #     print(f'y_pred = {y_pred[0]}')  









if __name__ == '__main__':
    args = add_arguments()
    print(args)
    main(args)