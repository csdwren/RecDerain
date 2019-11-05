## Single Image Deraining Using Bilateral Recurrent
### Introduction
In this work, we first propose a single recurrent network (SRN) by recursively unfolding a shallow residual network, where a recurrent layer is adopted to propagate deep features across multiple stages.
This simple SRN is effective not only in learning residual mapping for extracting rain streaks, but also in learning direct mapping for predicting clean background image. Furthermore, two SRNs are coupled to simultaneously exploit rain streak layer and clean background image layer. 
Instead of naive combination, we propose bilateral LSTMs, which not only can respectively propagate deep features of rain streak layer and background image layer acorss stages, but also bring the interplay between these two SRNs, finally forming bilateral recurrent network (BRN).
The experimental results demonstrate that our SRN and BRN notably outperform state-of-the-art deep deraining networks on synthetic datasets quantitatively and qualitatively. The proposed methods also perform more favorably in terms of generalization performance on real-world rainy dataset. 


## Prerequisites
- Python 3.6, PyTorch >= 0.4.0
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)
- MATLAB for computing [evaluation metrics](statistic/)


## Datasets

SRN and BRN are evaluated on five datasets*: 
Rain100H [1], Rain100L [1], Rain12 [2], Rain1400 [3] and SPA-data [4].  
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), and download the testing generalization dataset SPA-data [4] from [GoogleDrive](https://drive.google.com/drive/folders/1eSGgE_I4juiTsz0d81l3Kq3v943UUjiG).
And then place the unzipped folders into './datasets/'. Make sure that the path of the extracted file is consistent with '--data_path'. 

*_We note that:_

_(i) The datasets in the website of [1] seem to be modified. 
    But the models and results in recent papers are all based on the previous version, 
    and thus we upload the original training and testing datasets 
    to [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg) 
    and [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g)._ 

_(ii) For RainTrainH, we strictly exclude 546 rainy images that have the same background contents with testing images.
    All our models are trained on remaining 1,254 training samples._
        

## Getting Started

### 1) Testing

We have placed our pre-trained models into `./logs/`. 

Run shell scripts to test the models:
```bash
bash test_Rain12.sh     # test models on Rain12
bash test_Rain100H.sh   # test models on Rain100H
bash test_Rain100L.sh   # test models on Rain100L
bash test_Rain1400.sh   # test models on Rain1400
bash test_real.sh       # test models on SPA-data
```
All the results in the paper are also available at [BaiduYun]().
You can place the downloaded results into `./result/`, and directly compute all the [evaluation metrics](statistic/) in this paper.  

### 2) Evaluation metrics

We also provide the MATLAB scripts to compute the average PSNR and SSIM values reported in the paper.
 

```Matlab
 cd ./statistic
 run statistic_Rain100H.m
 run statistic_Rain100L.m
 run statistic_Rain12.m
 run statistic_Rain1400.m
 run statistic_real.m
```
###
Average PSNR/SSIM values on four datasets:

Dataset    | BRN       |SRN     
-----------|-----------|-----------
Rain100H   |30.47/0.913|29.46/0.899
Rain100L   |38.16/0.982|37.48/0.979
Rain12     |36.74/0.959|36.66/0.961
Rain1400   |32.75/0.948|32.60/0.946
SPA-data   |35.14/0.945|35.08/0.942


### Model Configuration

The following tables provide the configurations of options. 

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0                | GPU id
inter_iter             | 6                | Number of recursive stages
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results


## References
[1] Yang W, Tan RT, Feng J, Liu J, Guo Z, Yan S. Deep joint rain detection and removal from a single image. In IEEE CVPR 2017.

[2] Li Y, Tan RT, Guo X, Lu J, Brown MS. Rain streak removal using layer priors. In IEEE CVPR 2016.

[3] Fu X, Huang J, Zeng D, Huang Y, Ding X, Paisley J. Removing rain from single images via a deep detail network. In IEEE CVPR 2017.

[4] Wang T, Yang X, Xu K, Chen S,Zhang Q, and R. W. Lau, Spatial attentive single-image deraining with a high quality real rain dataset. In IEEE CVPR 2019.
