import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
import cv2
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms.functional as F
import pytorch_ssim

from generator import BRN, print_network

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="BRN")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=12, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")  #1e-3L   5e-4H   5e-5R1400
parser.add_argument("--outf", type=str, default="logs/test", help='path of log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default="datasets/RainTrainH/",help='path to training data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0,1", help='GPU id')
parser.add_argument("--inter_iter", type=int, default=8, help='number of inter_iteration')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():
    if not os.path.isdir(opt.outf):
        os.makedirs(opt.outf)
    # Load dataset
    print('Loading dataset ...\n')
    if (opt.data_path.find('Light') != -1 or opt.data_path.find('Heavy') != -1):
        dataset_train = newDataset(data_path=opt.data_path)
    else:
        dataset_train = Dataset(data_path=opt.data_path)
        
    # dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model

    net = BRN(recurrent_iter=opt.inter_iter, use_GPU=opt.use_GPU)
    net = nn.DataParallel(net)
    #print_network(net)

    #criterion = nn.MSELoss(size_average=False)
    criterion = pytorch_ssim.SSIM()

    # Move to GPU

    model = net.cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.2)  # learning rates
    #scheduler = MultiStepLR(optimizer, milestones=[120, 140], gamma=0.2)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0

    initial_epoch = findLastCheckpoint(save_dir=opt.outf)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.outf, 'net_epoch%d.pth' % initial_epoch)))

    for epoch in range(initial_epoch, opt.epochs):

        scheduler.step(epoch)
        # set learning rate
        for param_group in optimizer.param_groups:
            # param_group["lr"] = current_lr
            print('learning rate %f' % param_group["lr"])
        # train
        for i, (input, target) in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # rain = input - target
            input_train, target_train = Variable(input.cuda()), Variable(target.cuda())
           
            out_train, _, _, _ = model(input_train)

            pixel_loss = criterion(target_train, out_train)
            #mse = criterion(input_train1 - target_train, r)

            loss = (-pixel_loss) #+ mse
            loss.backward()
            
            optimizer.step()
            # results
            model.eval()
            with torch.no_grad():
                out_train, _, _, _ = model(input_train)
                out_train = torch.clamp(out_train, 0., 1.)
                #out_r_train = torch.clamp(out_r_train, 0., 1.)
                psnr_train = batch_PSNR(out_train, target_train, 1.)
            #psnr_train_r = batch_PSNR(out_r_train, rain_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                # writer.add_scalar('loss_r', loss_r.item(), step)
                #writer.add_scalar('PSNR_r on training data', psnr_train_r, step)
            step += 1
        ## the end of each epoch

        model.eval()
        
        with torch.no_grad():
            # log the images
            out_train, _, _, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            #out_r_train = torch.clamp(out_r_train, 0., 1.)
            Img = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
            Imgn = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
            Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        #rainstreak = utils.make_grid(out_r_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        #writer.add_image('estimated rain image', rainstreak, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_latest.pth'))

        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net_epoch%d.pth' % (epoch + 1)))
      

if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            print(opt.data_path.find('RainTrainH'))
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=100)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=100)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        elif (opt.data_path.find('Light') != -1 or opt.data_path.find('Heavy') != -1):
            prepare_data_newrain(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')

    main()
