# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import collections.abc
from itertools import repeat
import math
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


class Identity(nn.Cell):
    """Identity"""
    def construct(self, x):
        return x


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1. - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath1D(DropPath):
    def __init__(self, drop_prob):
        super(DropPath1D, self).__init__(drop_prob=drop_prob, ndim=1)

"""Define CSwinTransformer model"""
import mindspore
import numpy as np
import mindspore.common.initializer as weight_init
import mindspore.ops.operations as P
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy
from mindspore import ops

#from src.args import args
#from src.models.swintransformer.misc import _ntuple, Identity, DropPath1D
#from misc import _ntuple, Identity, DropPath1D
to_2tuple = _ntuple(2)

act_layers = {
    "GELU": nn.GELU,
    "gelu": nn.GELU,
}


class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None,
                 act_layer=act_layers["GELU"],
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class LePEAttention(nn.Cell):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None,HW=56):
        super().__init__()
        self.dim = dim
        self.HW = HW
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.matmul = ops.BatchMatMul()
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        stride = 1

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, pad_mode = 'pad',padding=1,group=dim)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.windows2img = WindowReverseConstruct()
        self.softmax = nn.Softmax(axis=-1)
    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = self.HW

        
        x = x.transpose(0,2,1).reshape(B, C, H, W)

        x = window_partition(x, self.H_sp, self.W_sp)

        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = self.HW
        x = x.transpose(0,2,1).reshape(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.reshape(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).transpose(0, 1, 3, 2)

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).transpose(0, 1, 3, 2)
        return x, lepe

    def construct(self, q,k,v):
        """
        x: B L C
        """
        q,k,v = q, k, v

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape

        #assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (self.matmul(q,k.transpose(0,1,3,2)))  # B head N C @ B head C N --> B head N N
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (self.matmul(attn,v)) + lepe
        x = x.transpose(0,2,1,3).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = self.windows2img(x, self.H_sp, self.W_sp, H, W).reshape(B, -1, C)  # B H' W' C

        return x

def window_partition(x,H_sp,W_sp ):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape

    x = x.reshape(B, C,H // H_sp, H_sp, W // W_sp, W_sp)
    #x = np.reshape(x, (B, H // H_sp, H_sp, W // W_sp, W_sp, C))

    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, H_sp* W_sp, C)
   
    return windows

class WindowReverseConstruct(nn.Cell):
    """WindowReverseConstruct Cell"""

    def construct(self,windows, H_sp, W_sp, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = windows.shape[0] // (H * W // H_sp // W_sp)
        x = ops.Reshape()(windows, (B, H // H_sp, W // W_sp, H_sp, W_sp, -1))
        x = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))
        x = ops.Reshape()(x, (B, H, W, -1))
        return x

class Merge_Block(nn.Cell):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm,HW=56):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, pad_mode='pad',padding = 1)
        self.norm = norm_layer([dim_out])
        self.HW = HW
    def construct(self, x):
        B, new_HW, C = x.shape
        
        H = W = self.HW
        
        x = x.transpose(0,2,1).reshape(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.reshape(B, C, -1).transpose(0,2,1)
        x = self.norm(x)
        
        return x
class Identity(nn.Cell):

    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x):
        return x
class CSWinBlock(nn.Cell):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False,HW=56):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.HW = HW
        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)

        self.norm1 = norm_layer([dim])

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1.0-drop)
        
        if last_stage:
            self.attns = nn.CellList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,HW=self.HW)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.CellList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,HW=self.HW)
                for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)
        #############
        #drop_path = drop_path.asnumpy()[0]
        self.drop_path = DropPath1D(drop_path[0]) if drop_path[0] > 0. else Identity()
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer([dim])
        self.cat = ops.Concat(axis=2)
    def construct(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        
        B, L, C = x.shape
        #assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        #qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        B_, N, C = x.shape

        #q = ops.Reshape()(self.q(x), (B_, -1, 3, C))
        
        q = self.q(x)
        
        k = self.k(x)
        
        v = self.v(x)
        


        if self.branch_num == 2:
            x1 = self.attns[0](q[:,:,:C//2],k[:,:,:C//2],v[:,:,:C//2])
            x2 = self.attns[1](q[:,:,C//2:],k[:,:,:C//2],v[:,:,:C//2])
            attened_x = self.cat((x1,x2))
        else:
            attened_x = self.attns[0](q,k,v)
        
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
class Rearrange(nn.Cell):
    def __init__(self):
        super(Rearrange, self).__init__()

    def construct(self, x):
        b,c,h,w = x.shape
        x = x.transpose(0,2,3,1)
        x = x.reshape(b,h*w,c)
        return x
class CSWinTransformer(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=64, depth=[2,4,32,2], split_size = [1,2,7,7],
                 num_heads= [ 2, 4, 8, 16 ], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.4, hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
  
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads
        self.mean = ops.ReduceMean(keep_dims=False)
        self.stage1_conv_embed = nn.SequentialCell(
            nn.Conv2d(in_chans, embed_dim, 7, 4, pad_mode='pad',padding= 2),
            Rearrange(),
            nn.LayerNorm([embed_dim])
        )

        curr_dim = embed_dim
        #linspace = ops.print
        #start = Tensor(0, mindspore.float32)
        #stop = Tensor(0.4, mindspore.float32)
        #dpr = [x.item() for x in linspace(start, stop, 12)]  # stochastic depth decay rule

        dpr = dpr = [[0.0],
            [0.0108],
            [0.0216],
            [0.0324],
            [0.0432],
            [0.0540],
            [0.0648],
            [0.0756],
            [0.0864],
            [0.0972],
            [0.1081],
            [0.1189],
            [0.1297],
            [0.1405],
            [0.1513],
            [0.1621],
            [0.1729],
            [0.1837],
            [0.1945],
            [0.2054],
            [0.2162],
            [0.2270],
            [0.2378],
            [0.2486],
            [0.2594],
            [0.2702],
            [0.2810],
            [0.2918],
            [0.3027],
            [0.3135],
            [0.3243],
            [0.3351],
            [0.3459],
            [0.3567],
            [0.3675],
            [0.3783],
            [0.3891],
            [0.3922],
            [0.3954],
            [0.4]]

        self.stage1 = nn.CellList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size//4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,HW = 56)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2,HW=56)
        curr_dim = curr_dim*2
        self.stage2 = nn.CellList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size//8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer,HW=28)
            for i in range(depth[1])])
        #np.sum(depth[:1])+i
        self.merge2 = Merge_Block(curr_dim, curr_dim*2,HW=28)
        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size//16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer,HW=14)
            for i in range(depth[2])])
        #np.sum(depth[:2])+i
        self.stage3 = nn.CellList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*2,HW=14)
        curr_dim = curr_dim*2
        self.stage4 = nn.CellList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size//32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=True,HW=7)
            for i in range(depth[-1])])
        #np.sum(depth[:-1])+i
        self.norm = norm_layer([curr_dim])
        # Classifier head
        self.head = nn.Dense(curr_dim, num_classes) if num_classes > 0 else Identity()
                        
        self.init_weights()

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))


    def forward_features(self, x):
        B = x.shape[0]
        x = self.stage1_conv_embed(x)
        
        for blk in self.stage1:
            x = blk(x)

            '''        
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3], 
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)

            
            for blk in blocks:
                if self.use_chk:
                    #x = checkpoint.checkpoint(blk, x)
                    pass
                else:
                    x = blk(x)'''


        x = self.merge1(x)
        for n in self.stage2:
            x = n(x)
        x = self.merge2(x)
        for n in self.stage3:
            x = n(x)
        x = self.merge3(x)
        for l in self.stage4:
            x = l(x)
        x = self.norm(x)
        #x = x.transpose(1,0,2)
        x = self.mean(x,1)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x