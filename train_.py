
        
        
import mindspore
import moxing as mox
import  os
import  sys
import  time
import  glob
import  numpy as np
import  logging
import  argparse

from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import Tensor, Model


from mindspore.common import set_seed
from mindspore import context
from mindspore import nn,DynamicLossScaleManager
from mindspore.parallel._utils import _get_device_num

import mindspore.ops as ops
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
import moxing as mox

from src.data.movingMNIST import get_dataset
from src.models.convlstm import G_convlstm,G_ConvLSTMCell
from src.tools.callback import lr_

set_seed(1996)
environment = 'debug'  
if environment == 'debug':
    workroot = '/home/ma-user/work' #调试任务使用该参数
else:
    workroot = '/home/work/user-job-dir' # 训练任务使用该参数
print('current work mode:' + environment + ', workroot:' + workroot)
parser = argparse.ArgumentParser("convlstm")
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')

parser.add_argument('--data_path',
                    help='path to training/inference dataset folder',
                    default= workroot + '/data/')

parser.add_argument('--train_path',
                    help='model folder to save/load',
                    default= workroot + '/model/')#save_path

parser.add_argument('--num_parallel_workers', type=int, default=1, help='num_parallel_work')
parser.add_argument("--save_every", default=2, type=int, help="Save every ___ epochs(default:2)")

parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: CPU),若要在启智平台上使用NPU，需要在启智平台训练界面上加上运行参数device_target=Ascend')
args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
random_seed = 1996
np.random.seed(random_seed)

 ######################## 将多个数据集从obs拷贝到训练镜像中 （固定写法）########################  
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    return 
 ######################## 将输出的模型拷贝到obs（固定写法）########################  
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return     



def main():
    #rank =get_rank()
    rank =1
    print(os.listdir(workroot))
    #初始化数据和模型存放目录
    data_dir = args.data_path  #先在训练镜像中定义数据集路径
    train_dir = args.train_path #先在训练镜像中定义输出路径
    if not os.path.exists(data_dir):
        raise
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
 ######################## 将数据集从obs拷贝到训练镜像中 （固定写法）########################   
    # 在训练环境中定义data_url和train_url，并把数据从obs拷贝到相应的固定路径，以下写法是将数据拷贝到/home/work/user-job-dir/data/目录下，可修改为其他目录
    #ObsToEnv(args.data_url,data_dir)
    #context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)


    net = G_convlstm(G_ConvLSTMCell,args.batch_size)
    criterion = nn.MSELoss()#
    #net_with_loss = NetWithLoss(net, criterion)
    data = get_dataset(data_dir,args.batch_size)#
    batch_num = data.train_dataset.get_dataset_size()

    optimizer = nn.Adam(
                net.trainable_params(),
                learning_rate = 0.0001) 
    
    eval_network = nn.WithEvalCell(net, criterion)
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2**24)
    model = Model(network=net, loss_scale_manager=loss_scale_manager,loss_fn=criterion, optimizer=optimizer)
    ckpt_save_dir = workroot + '/model/ckpt_' + str(rank)
    loss_cb = LossMonitor(100)

    config_ck = CheckpointConfig(save_checkpoint_steps=500, keep_checkpoint_max=16)
    ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=ckpt_save_dir, config=config_ck)

    lr_cb = lr_(0.5,4)
    print(_get_device_num())
    print("begin train")
    model.train(int(args.epochs), data.train_dataset,
                callbacks=[ckpoint_cb,loss_cb,lr_cb],
                dataset_sink_mode=False)
    print("train success")

    EnvToObs(train_dir, args.train_url)
if __name__ == '__main__':
    main()