import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *

from generator import BRN, print_network
import time

parser = argparse.ArgumentParser(description="BRN_Test")
parser.add_argument("--logdir", type=str, default="logs/dataset/BRN", help='path of log files')
parser.add_argument("--data_path", type=str, default="dataset/...", help='path to testing data')
parser.add_argument("--save_path", type=str, default="results/dataset/BRN", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--inter_iter", type=int, default=8, help='number of inter_iteration')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    # Build model
    print('Loading model ...\n')

    model = BRN(opt.inter_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    state_dict = torch.load(os.path.join(opt.logdir, 'net_latest.pth'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    #model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    
    # process data
    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)

            # image
            Img = cv2.imread(img_path)
            h, w, c = Img.shape

            b, g, r = cv2.split(Img)
            Img = cv2.merge([r, g, b])
        
       
            Img = normalize(np.float32(Img))
            Img = np.expand_dims(Img.transpose(2, 0, 1), 0)
       
            ISource = torch.Tensor(Img)
        
            INoisy = ISource 

            if opt.use_GPU:
                ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            else:
                ISource, INoisy = Variable(ISource), Variable(INoisy)

            with torch.no_grad(): # this can save much memory
                torch.cuda.synchronize()
                start_time = time.time()
                out, _, out_r, _ = model(INoisy)
           
                out = torch.clamp(out, 0., 1.)
                out_r = torch.clamp(out_r, 0., 1.)
            
                torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                print(img_name)
                print(dur_time)
                time_test += dur_time
        
            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   
            
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())     
            
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b]) 
       
            save_path = opt.save_path     
        
            cv2.imwrite(os.path.join(save_path, img_name), save_out)
        
            count = count + 1

    print('Avg. time:', time_test/count)

   

if __name__ == "__main__":
    main()

