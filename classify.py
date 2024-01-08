import argparse
import os
import h5py

import numpy

from tqdm import tqdm
import torch


import utility
import process
import models

from dataset.dataset_class import get_dataloader, get_pn_dataloader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':

    # mode argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of batch')
    parser.add_argument('--dim', type=int, default=256, help='ELM weight size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.4, help='Radius of sampling sphere')
    parser.add_argument('--train', type=bool, default=True, help='Conduct training')
    parser.add_argument('--normals', type=bool, default=True, help='Use normals for training')
    parser.add_argument('--sample_size', type=int, default=256,
                        help='Number of sampling points for distance field calculation')
    parser.add_argument('--subset', type=int, default=1024, help='Subset for acceleration and robustness')
    args = parser.parse_args()

    DATA_DIR = './data/modelnet40_ply_hdf5_2048'  # your directory
    SAVE_DIR = './data/weight_data'  # your directory

    num_classes = 40
    train_inst = 9840
    test_inst = 2468

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # check if data preparation is necessary
    data_list_train = utility.makeList(os.path.join(DATA_DIR, 'train_files.txt'))
    data_list_test = utility.makeList(os.path.join(DATA_DIR, 'test_files.txt'))

    if args.train:
        if args.normals:
            # using point and normal
            train_data = get_pn_dataloader(data_list_train, 0.4, 256, 'train', 16, DATA_DIR, SAVE_DIR, args.subset, num_classes)
            test_data = get_pn_dataloader(data_list_test, 0.4, 256, 'test', 16, DATA_DIR, SAVE_DIR, args.subset, num_classes)
            net = models.defineModelPN(args.subset, args.dim, num_classes).to(device)
        else:
            train_data = get_dataloader(data_list_train, 0.4, 256, 'train', 16, DATA_DIR, SAVE_DIR, args.subset,
                                        num_classes)
            test_data = get_dataloader(data_list_test, 0.4, 256, 'test', 16, DATA_DIR, SAVE_DIR, args.subset,
                                       num_classes)
            net = models.defineModel(args.subset, args.dim, num_classes).to(device)
        max_acc = 0.0
        # prepare model

        adam = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.99, 0.999), eps=1e-8, weight_decay=0.000)
        loss_func = torch.nn.CrossEntropyLoss().to(device)


        # training process
        curpred = numpy.zeros((test_inst, num_classes))
        test_labels = numpy.zeros(test_inst)

        for it in range(1, args.epochs+1):
            pbar = tqdm(train_data)
            for b, data in enumerate(pbar):
                t_data = data['data'].to(device)
                print(t_data.shape)
                label = data['label'].to(device)

                output = torch.squeeze(net(t_data), 1)
                pbar.set_description("EPOCH[{}][{}]".format(it, b))

                loss = loss_func(output, label)

                adam.zero_grad()
                loss.backward()
                adam.step()
            

            net.eval()

            curid = 0
            if it % 10==0:
                for t in test_data:
                    x = t['data'].to(device)
                    y = t['label']
                    batchend = y.shape[0]
                    curpred[curid:curid+batchend] = torch.squeeze(net(x),1 ).detach().cpu().numpy()
                    test_labels[curid:curid + batchend] = y
                    curid += batchend

                pred_val = numpy.argmax(curpred, 1)
                correct = numpy.sum(pred_val.flatten() == test_labels.flatten())
                scores = float(correct / float(test_inst))

                print('Test accuracy: ', scores)
                if max_acc < scores:
                    max_acc = scores
                    net_work_dir = "./model_data/" + "dim" + str(args.dim) + "margin" + str(
                        args.margin) + "sample" + str(args.sample_size) + "/"
                    if not os.path.exists(net_work_dir):
                        os.mkdir(net_work_dir)
                    net_work_path = net_work_dir + "best" + str(it) + ".pth"
                    torch.save(net.state_dict(), net_work_path)
                print('Maximum accuracy: %f' % max_acc)

            if it%100 ==0 :
                net_work_dir = "./model_data/" + "dim" + str(args.dim) + "margin" + str(args.margin) + "sample" + str(
                    args.sample_size) + "/"
                if not os.path.exists(net_work_dir):
                    os.mkdir(net_work_dir)
                net_work_path = net_work_dir + str(it) + ".pth"
                torch.save(net.state_dict(), net_work_path)