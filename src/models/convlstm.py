import mindspore
import mindspore.nn as nn
import mindspore.ops as ops 

from mindspore import dtype as mstype
from mindspore import Parameter
from mindspore import Tensor

from mindspore.common.initializer import One, Normal,Zero

class ConvLSTMCell(nn.Cell):
    def __init__(self,shape,input_dim,hidden_dim,kernel_size):
        super(ConvLSTMCell,self).__init__()
        self.input_dim = input_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2#list?


        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(self.input_dim+self.hidden_dim,4*self.hidden_dim,
            self.kernel_size,stride=1,pad_mode='pad',padding=self.padding)
        self.gro= nn.GroupNorm(4 * self.hidden_dim // 32, 4 * self.hidden_dim)
        self.shape = shape
        self.op = ops.Concat(1)
        self.split = ops.Split(1,4)
        '''        
        self.output_inner = Parameter(Tensor(shape=(10,32,hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
       
        self.hx = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        self.cx = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))

        self.c_next = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        self.h_next = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        '''
        #self.output_inner = Parameter(Tensor(shape=(10,4,hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
       
        #self.hx = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.cx = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())

        #self.c_next = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.h_next = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.input = Tensor(shape=(10,4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.zeros = ops.Zeros()
        self.stack = ops.Stack()
    def construct(self,input_tensors,cur_state,seq_len=10):

        if cur_state is None:
            h_cur , c_cur = self.zeros((32, self.hidden_dim,self.shape[0],self.shape[1]),mstype.float32), self.zeros((32,   self.hidden_dim,self.shape[0],self.shape[1]),mstype.float32)
        else:

            h_cur , c_cur = cur_state
        if input_tensors is None:
            input_tensors = self.zeros((10,32, self.input_dim,self.shape[0],self.shape[1]),mstype.float32)
        #print(input_tensor.shape,h_cur)
        output_inner = []#self.output_inner

        for index in range(seq_len):
            #print(index)
            input_tensor = input_tensors[index,...]
            #print(input_tensor.shape,h_cur.shape)
            combined = self.op((input_tensor,h_cur))
           
            #print(combined.shape,h_cur.shape)
            combined_conv = self.conv(combined)
            combined_conv = self.gro(combined_conv)
            cc_i,cc_f,cc_o,cc_g = self.split(combined_conv)
            i = self.sig(cc_i)
            f = self.sig(cc_f)
            o = self.sig(cc_o)
            g = self.tanh(cc_g)

            c_next = f*c_cur + i*g
            h_next = o*self.tanh(c_next)
            h_cur = h_next
            c_cur = c_next
            output_inner.append(h_next)
            #print(output_inner.shape,h_next.shape,c_next.shape)
        return self.stack(output_inner),(h_next,c_next)
    
class G_ConvLSTMCell(nn.Cell):
    def __init__(self,shape,input_dim,hidden_dim,kernel_size):
        super(G_ConvLSTMCell,self).__init__()
        self.input_dim = input_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2#list?


        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(self.input_dim+self.hidden_dim,4*self.hidden_dim,
            self.kernel_size,stride=1,pad_mode='pad',padding=self.padding)
        self.gro= nn.GroupNorm(4 * self.hidden_dim // 32, 4 * self.hidden_dim)
        self.shape = shape
        self.op = ops.Concat(1)
        self.split = ops.Split(1,4)
        '''        
        self.output_inner = Parameter(Tensor(shape=(10,32,hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
       
        self.hx = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        self.cx = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))

        self.c_next = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        self.h_next = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        '''
        #self.output_inner = Parameter(Tensor(shape=(10,4,hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
       
        #self.hx = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.cx = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())

        #self.c_next = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.h_next = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.input = Tensor(shape=(10,4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.zeros = ops.Zeros()
        self.stack = ops.Stack()
    def construct(self,input_tensors,cur_state,seq_len=10):



        h_cur , c_cur = cur_state

        output_inner = []

        for index in range(seq_len):
           
            input_tensor = input_tensors[index,...]
            #print(input_tensor.shape,h_cur.shape)
            combined = self.op((input_tensor,h_cur))
           
            #print(combined.shape,h_cur.shape)
            combined_conv = self.conv(combined)
            combined_conv = self.gro(combined_conv)
            cc_i,cc_f,cc_o,cc_g = self.split(combined_conv)
            i = self.sig(cc_i)
            f = self.sig(cc_f)
            o = self.sig(cc_o)
            g = self.tanh(cc_g)

            c_next = f*c_cur + i*g
            h_next = o*self.tanh(c_next)
            h_cur = h_next
            c_cur = c_next
            output_inner.append(h_cur)
            #print(output_inner.shape,h_next.shape,c_next.shape)
        return self.stack(output_inner),(h_cur,c_cur)
from mindspore import nn
from collections import OrderedDict


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.Conv2dTranspose(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 pad_mode='pad',
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU()))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(alpha=0.2)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               pad_mode='pad',
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU()))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(alpha=0.2)))
        print(layers)
    return nn.SequentialCell(OrderedDict(layers))    

class G_convlstm(nn.Cell):
    def __init__(self,G_ConvLSTMCell,batch):
        super(G_convlstm,self).__init__()
        self.leak_relu = nn.LeakyReLU(alpha=0.2)
        self.en_conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,pad_mode='pad',padding=1)
        self.en_conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,pad_mode='pad',padding=1)
        self.en_conv3 = nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=2,pad_mode='pad',padding=1)
        

        self.en_rnn1 = G_ConvLSTMCell(shape=(64,64), input_dim=16, kernel_size=(5,5), hidden_dim =64)
        self.en_rnn2 = G_ConvLSTMCell(shape=(32,32), input_dim=64, kernel_size=(5,5), hidden_dim =96)
        self.en_rnn3 = G_ConvLSTMCell(shape=(16,16), input_dim=96, kernel_size=(5,5), hidden_dim =96)
        
        
        self.dn_conv1 = nn.Conv2dTranspose(in_channels=96,out_channels=96,kernel_size=4,stride=2,pad_mode='pad',padding=1)
        self.dn_conv2 = nn.Conv2dTranspose(in_channels=96,out_channels=96,kernel_size=4,stride=2,pad_mode='pad',padding=1)
        self.dn_conv3 = nn.Conv2d(in_channels=64,out_channels=16,kernel_size=3,stride=1,pad_mode='pad',padding=1)
        self.dn_conv4 = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,stride=1,pad_mode='pad',padding=1)        

        self.dn_rnn1 = G_ConvLSTMCell(shape=(16,16), input_dim=96, kernel_size=(5,5), hidden_dim =96)
        self.dn_rnn2 = G_ConvLSTMCell(shape=(32,32), input_dim=96, kernel_size=(5,5), hidden_dim =96)
        self.dn_rnn3 = G_ConvLSTMCell(shape=(64,64), input_dim=96, kernel_size=(5,5), hidden_dim =64)
        
        self.h0 = Tensor(shape=(batch,64,64,64), dtype=mstype.float32, init=Zero())
        self.c0 = Tensor(shape=(batch,64,64,64), dtype=mstype.float32, init=Zero())
        
        self.h1 = Tensor(shape=(batch,96,32,32), dtype=mstype.float32, init=Zero())
        self.c1 = Tensor(shape=(batch,96,32,32), dtype=mstype.float32, init=Zero())
        
        self.h2 = Tensor(shape=(batch,96,16,16), dtype=mstype.float32, init=Zero())
        self.c2 = Tensor(shape=(batch,96,16,16), dtype=mstype.float32, init=Zero())
        
        self.input0 = Tensor(shape=(10,batch,96,16,16), dtype=mstype.float32, init=Zero())
        self.reshape = ops.Reshape()
    def construct(self, inputs):
        inputs = inputs.transpose(1,0,2,3,4)
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.en_conv1(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        h_0 = self.h0
        c_0 = self.c0

        inputs, state_stage1 = self.en_rnn1(inputs,(h_0,c_0),seq_len=10)
        
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.en_conv2(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        h_0 = self.h1
        c_0 = self.c1
        
        inputs, state_stage2 = self.en_rnn2(inputs,(h_0,c_0))
        
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.en_conv3(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        h_0 = self.h2
        c_0 = self.c2
        
        inputs, state_stage3 = self.en_rnn3(inputs,(h_0,c_0))
        
        
        inputs, _ = self.dn_rnn1(self.input0, state_stage3, seq_len=10)
       
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.dn_conv1(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        
        
        inputs, _ = self.dn_rnn2(inputs, state_stage2, seq_len=10)
        
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.dn_conv2(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        
        
        inputs, _ = self.dn_rnn3(inputs, state_stage1, seq_len=10)
        
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.dn_conv3(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.dn_conv4(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        inputs = inputs.transpose(1, 0,2,3,4)
        return inputs
        
class Encoder(nn.Cell):
    def __init__(self, subnets, rnns):
        super(Encoder,self).__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)
        self.reshape = ops.Reshape()
        self.stages = [self.stage1,self.stage2,self.stage3]
        self.rnns = [self.rnn1,self.rnn2,self.rnn3]
    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def construct(self, inputs):

        inputs = inputs.transpose(1,0,2,3,4)  # to S,B,1,64,64
        hidden_states = []
        #logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            #print(i)
            inputs, state_stage = self.forward_by_stage(
                inputs, self.stages[i-1],self.rnns[i-1])
            hidden_states.append(state_stage)
        return hidden_states
class Decoder(nn.Cell):
    def __init__(self, subnets, rnns):
        super(Decoder,self).__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))
            print(self.blocks - index,rnn)
        self.reshape = ops.Reshape()
        self.stages = [self.stage1,self.stage2,self.stage3]
        self.rnns = [self.rnn1,self.rnn2,self.rnn3]
    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        return inputs

        # input: 5D S*B*C*H*W

    def construct(self, hidden_states):
       
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       self.stages[2],
                                        self.rnns[2])
        #print('x',inputs.shape)
        #youcuo list 
        for i in range(0,2):
            inputs = self.forward_by_stage(inputs, hidden_states[1-i],
                                           self.stages[1-i],
                                           self.rnns[1-i])
        #inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        inputs = inputs.transpose(1, 0,2,3,4)  # to B,S,1,64,64
        return inputs


class ED(nn.Cell):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output

encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        ConvLSTMCell(shape=(64,64), input_dim=16, kernel_size=(5,5), hidden_dim =64),
        ConvLSTMCell(shape=(32,32), input_dim=64, kernel_size=(5,5), hidden_dim =96),
        ConvLSTMCell(shape=(16,16), input_dim=96, kernel_size=(5,5), hidden_dim =96)
    ]
]
decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        
        ConvLSTMCell(shape=(16,16), input_dim=96, kernel_size=(5,5), hidden_dim =96),
        ConvLSTMCell(shape=(32,32), input_dim=96, kernel_size=(5,5), hidden_dim =96),
        ConvLSTMCell(shape=(64,64), input_dim=96, kernel_size=(5,5), hidden_dim =64),
    ]
]


