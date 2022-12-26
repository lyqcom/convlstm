# 目录
<!-- TOC -->

- [目录](#目录)
    - [Convlstm描述](#Convlstm)
    - [模型架构](#模型架构)
    - [数据集MovingMNIST](#MovingMNIST)
    - [环境要求](#ENV)
    - [快速开始](#Start)
    - [脚本及样例代码](#Code)
    - [脚本参数](#param)
    - [分布式训练](#分布式训练)
    - [评估](#eval)
    - [评估过程](#evalusage)
    - [评估结果](#evalresult)
    - [导出](#export)
    - [导出过程](#exportusage)

    - [推理](#推理)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)

<!-- /TOC -->
# [Convlstm](#Convlstm)

> 
传统的LSTM的关键是细胞状态，表示细胞状态的这条线水平的穿过图的顶部。LSTM的删除或者添加信息到细胞状态的能力是由被称为Gate的结构赋予的。LSTM的第一步是决定要从细胞状态中丢弃什么信息。 该决定由被称为“忘记门”的Sigmoid层实现。它查看ht-1(前一个输出)和xt(当前输入)，并为单元格状态Ct-1(上一个状态)中的每个数字输出0和1之间的数字。1代表完全保留，而0代表彻底删除。
![encoder](img/1.png) 
下一步是决定我们要在细胞状态中存储什么信息。
第一，sigmoid 层称 “输入门层” 决定什么值我们将要更新。然后，一个 tanh 层创建一个新的候选值向量，Ct，会被加入到状态中。下一步，我们会讲这两个信息来产生对状态的更新。
![encoder](img/2.png) 

更新上一个状态值Ct−1了，将其更新为Ct。签名的步骤以及决定了应该做什么，我们只需实际执行即可。我们将上一个状态值乘以ft，以此表达期待忘记的部分。之后我们将得到的值加上 it∗Ct。这个得到的是新的候选值，按照我们决定更新每个状态值的多少来衡量。最后，我们需要决定我们要输出什么。 此输出将基于我们的细胞状态，但将是一个过滤版本
![encoder](img/3.png) 

最终，我们需要确定输出什么值。这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本。首先，我们运行一个 sigmoid 层来确定细胞状态的哪个部分将输出出去。接着，我们把细胞状态通过 tanh 进行处理（得到一个在-1到1之间的值）并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。
![encoder](img/4.png) 

Convlstm模型和传统LSTM的不同：

①ConvLSTM模型中将fully-connect layer改成convolutional layer

②模型的input是3D tensor。

## [模型架构](#目录)
![encoder](img/5.png) 
> 
预测模型包括两个网络，一个编码网络和一个预测网络。，预测网络的初始状态和单元输出从编码网络的最后状态复制。这两个网络都是通过叠加几个ConvLSTM层而形成的。由于预测目标与输入具有相同的维数，将预测网络中的所有状态连接起来，并将它们输入1×1卷积层，生成最终的预测。

## [数据集MovingMNIST](#MovingMNIST)

> 
1、用于生成训练数据的MNIST数据集:train-images-idx3-ubyte.gz (http://yann.lecun.com/exdb/mnist/)
2、测试数据集MovingMNIST：mnist_test_seq.npy (http://www.cs.toronto.edu/~nitish/unsupervised_video/)
>启智平台下 创建单卡训练任务时，请保证data.zip压缩包为以下结构
> data/train-images-idx3-ubyte.gz 
> data/mnist_test_seq.npy     
启智平台下 创建单卡调试任务时，请保证work环境data目录为以下结构
>work/data/train-images-idx3-ubyte.gz               
>work/data//mnist_test_seq.npy 
## [环境要求](#ENV)

> 提供运行该代码前需要的环境配置，包括：
>
> * 第三方库 scikit-image
> * 镜像	tensorflow1.15-mindspore1.5.1-cann5.0.3-euler2.8-aarch64
> * 规格	Ascend: 1*Ascend910|CPU: 24核 96GB

## [快速开始](#Start)
>调试环境下
```shell
cd work/convlstm
python train_.py --batch_size 32 --save_every 5
 
bash scripts/run_single_train.sh [DEVICE_ID] [BACTHSIZE] [EPOCHS_NUMS] [DATAPATH SAVEPATH]
for example: bash scripts/run_single_train.sh 0 32 100 ./data ./model
```

> 训练任务下
创建单卡训练任务
![encoder](img/6.png) 


### [脚本和样例代码](#Code)

> 提供完整的代码目录展示（包含子文件夹的展开），描述每个文件的作用
 ```bash
└─convlstm
    ├──test.py                   # 验证脚本
    ├──export.py					#导出脚本 
    ├──train.py					#启智平台训练脚本 
    ├──train_.py				#启智平台调试脚本 
    ├── README.md               #README  
    └─ src             # 辅助脚本
        └─ data
            ├─movingMNIST.py                 # 数据集
        └─ model
            ├─convlstm.py                 # 模型结构
        └─ tools
            ├─callback.py                 # 回调函数
    └─ scripts
        ├─run_eval.sh
        ├─run_single_train.sh 
```

### [脚本参数](#param)

> batch_size 32
> epochs 100



### [分布式训练](#分布式训练)

> 暂无

## [评估](#eval)

### [评估过程](#evalusage)

```shell
cd work/convlstm
python test.py      
#参数
--workroot '/home/ma-user/work/data' #data path
--pretrained_model 'path/xx.ckpt' 
--batch_szie 32#input batch_szie
#Or bash
bash scripts/run_eval.sh [DEVICE_ID] [DATAPATH] [CKPTPATH] [BACTHSIZE]   
for example: bash cripts/run_eval.sh 0 '/home/ma-user/work/data' 'path/xx.ckpt' 32       
```

### [评估结果](#evalresult)

```log
mse per frame: 34.540098808288576
19.57088385925293
22.804474285888674
26.540182116699217
29.597495532226564
32.90274191894531
36.1155828125
39.4725357421875
42.654525384521484
46.08683151855469
49.655734912109374
ssim per frame: 0.80526114
0.69719815
0.7559249
0.7895438
0.8175223
0.83220935
0.8386961
0.8400583
0.8364529
0.8283874
0.8166177
fmae per frame: 102.57233
68.58177
75.04176
83.23884
90.65259
98.49818
106.07547
114.028946
121.70901
129.81093
138.08585
```

## [导出](#export)

### [导出过程](#exportusage)

```shell
cd work/convlstm
python export.py 
#参数
--pretrained_model 'path/xx.ckpt' #ckpt文件path
--batch_szie 32#input batch_szie
--file_format 'MINDIR'#输出文件格式
```


## [性能](#性能)

### [训练性能](#训练性能)



| Parameters                 | Ascend 910                                                   | 
| -------------------------- | ------------------------------------------------------------ | 
| Model Version              | convlstm                                                     |
| Resource                   | Ascend 910; CPU 2.60GHz, 32cores; Memory 256G; OS Euler2.8  |
| uploaded Date              | 07/29/2022 (month/day/year)                                  |           
| MindSpore Version          | 1.5.1                                                        |
| Dataset                    | MovingMNIST                                                  | 
| Training Parameters        | epoch=100, batch_size=32                                     |
| Optimizer                  | Adam                                                         |
| Loss Function              | MSE                                                          |
| outputs                    | probability                                                  |
| Loss                       | 0.0001319517                                                 |
| Total time                 | 13 hours                                                      |
| Parameters (M)             | 11.2                                                         | 
| Checkpoint for Fine tuning | 103M (.ckpt file)                                             | 
| Scripts                    | [link](https://gitee.com/mindspore/models/tree/master/official/cv/)                     
### [推理性能](#推理性能)





| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | convlstm                    |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 07/29/2022 (month/day/year) |
| MindSpore Version   | 1.5.1                       |
| Dataset             | MovingMNIST                 |
| batch_size          | 32                          |
| outputs             | probability                 |
| MSE            | 34                      |
| Model for inference | 103M (.ckpt file)             |





## 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

### 贡献者

lmh447669785 

* [lmh447669785](https://git.openi.org.cn/lmh447669785/convlstm/)


