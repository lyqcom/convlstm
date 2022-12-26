from skimage.metrics import structural_similarity
from src.data.movingMNIST import get_dataset
from src.models.convlstm import G_convlstm,G_ConvLSTMCell

from mindspore import nn
import mindspore
from mindspore import context
import numpy as np
import cv2
import os 
import argparse


def batch_mae_frame_float(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    mae = np.sum(np.absolute(x - y), axis=(1, 2), dtype=np.float32)
    return np.mean(mae)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--workroot', type=str, default='/home/ma-user/work/data')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--pretrained_model', type=str, default='/home/ma-user/work/model/ckpt_1/lstm_3-95_172.ckpt')
    
    args = parser.parse_args()
    
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    data_dir = args.workroot #先在训练镜像中定义数据集路径
    data = get_dataset(data_dir,args.batch_size)
    criterion = nn.MSELoss()
    net = G_convlstm(G_ConvLSTMCell,args.batch_size)
    ck = args.pretrained_model #模型路径
    #ck = train_dir +'/0/' + 'lstm-500_78.ckpt'
    mindspore.load_checkpoint(ck,net=net)
    eval_dataset = data.val_dataset
    it = eval_dataset.create_dict_iterator()

    avg_mse = 0
    batch_id = 0
    img_mse, ssim, fmae = [], [], []
    for i in range(10):
        img_mse.append(0)
        ssim.append(0)
        fmae.append(0)
    ssim_Cell = nn.SSIM()


    for d in it:
        batch_id = batch_id + 1
        x = d['input_images']
        y = d['target_images']
        B,S,_,H,W = x.shape
        pre = net(x)#B,S,1,H,W
        pre_ = pre.transpose(0,1,3,4,2)
        y_ = y.transpose(0,1,3,4,2)
        test_ims_ori = pre_.asnumpy()
        img_gen = y_.asnumpy()
        #y = y.reshape(B*S,1,H,W)
        #pre = pre.reshape(B*S,1,H,W)
        for i in range(10):

            x = test_ims_ori[:, i , :, :, 0]
            gx = img_gen[:, i, :, :, 0]
            fmae[i] += batch_mae_frame_float(gx, x)

            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            for b in range(args.batch_size):

                score, _ = structural_similarity(pred_frm[b], real_frm[b], full=True)
                ssim[i] += score


    avg_mse = avg_mse / (batch_id*args.batch_size)
    ssim = np.asarray(ssim, dtype=np.float32)/(args.batch_size*batch_id)
    fmae = np.asarray(fmae, dtype=np.float32)/batch_id


    print('mse per frame: ' + str(avg_mse/10))
    for i in range(10):
        print(img_mse[i] / (batch_id*args.batch_size))


    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(10):
        print(ssim[i])


    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(10):
        print(fmae[i])