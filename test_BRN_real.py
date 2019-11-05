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


parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/derain_syn_RRNDSI_loss1_s8", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--data_path", type=str, default="/media/r/dataset/rain/SPA-data/real_test_1000/", help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/r/shangwei/derain/results/syn_RRNDSI_loss1_s8/output", help='path to save results')
parser.add_argument("--save_path_r", type=str, default="/home/r/shangwei/derain/results/syn_RRNDSI_loss1_s8/rainstreak", help='path to save results1')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--inter_iter", type=int, default=8, help='number of inter_iteration')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def normalize(data):
    return data/255.


# TODO: two pixel shuffle functions to process the images
def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf + wc, hf:hf + hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf + wc, hf:hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real

def main():
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.isdir(opt.save_path_r):
        os.makedirs(opt.save_path_r)
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
            Img = pixelshuffle(Img, 2)
       
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
                save_out_r = np.uint8(255 * out_r.data.cpu().numpy().squeeze())
            
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())
                save_out_r = np.uint8(255 * out_r.data.numpy().squeeze())
            
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
        

            save_out_r = save_out_r.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out_r)
            save_out_r = cv2.merge([r, g, b])
       

            save_path = opt.save_path
            save_path_r = opt.save_path_r

            save_out = reverse_pixelshuffle(save_out, 2)
            save_out_r = reverse_pixelshuffle(save_out_r, 2)
        
            cv2.imwrite(os.path.join(save_path, img_name), save_out)
            cv2.imwrite(os.path.join(save_path_r, img_name), save_out_r)
        
            count = count + 1

    print('Avg. time:', time_test/count)

   

if __name__ == "__main__":
    main()
