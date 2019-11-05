
clear all;
close all;

gt_path='/media/r/dataset/rain/SPA-data/real_test_1000/gt/'; %'../datasets/test/Rain100L/';

SRN = '/home/r/shangwei/PReNet-master/results/real/SRN/'; %'../results/Rain100L/SRN/';

BRN = '/home/r/shangwei/PReNet-master/results/real/BRN/'; %'../results/Rain100L/BRN/';

 
struct_model = {
          struct('model_name','SRN','path',SRN),...
          struct('model_name','BRN','path',BRN),...
    };


nimgs=1000;nrain=1;
nmodel = length(struct_model);

psnrs = zeros(nimgs,nmodel);
ssims = psnrs;

for nnn = 1:nmodel
    
    tp=0;ts=0;te=0;
    nstart = 0;
    for iii=nstart+1:nstart+nimgs
        for jjj=1:nrain
            %         fprintf('img=%d,kernel=%d\n',iii,jjj);
            x_true=im2double(imread(fullfile(gt_path,sprintf('%03dgt.png',iii-1))));%x_true
            x_true = rgb2ycbcr(x_true);x_true=x_true(:,:,1);
            

            %%
            x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('%03d.png',iii-1)))));
            x = rgb2ycbcr(x);x = x(:,:,1);
            tp = mean(psnr(x,x_true));
            ts = ssim(x*255,x_true*255);
            
            psnrs(iii-nstart,nnn)=tp;
            ssims(iii-nstart,nnn)=ts;
            
            %
        end
    end
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f\n',struct_model{nnn}.model_name,mean(psnrs(:,nnn)),mean(ssims(:,nnn)));
    
end





