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
"""
##############export checkpoint file into air, mindir models#################
python export.py
"""
import argparse
from mindspore.context import ParallelMode
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.models.convlstm import G_convlstm,G_ConvLSTMCell





def run_export():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR',
                        help='file format')
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args_opt.device_id)


    shape = (args_opt.batch_size,10,1,64,64)
    network = G_convlstm(G_ConvLSTMCell,args_opt.batch_size)

    param_dict = load_checkpoint(args_opt.pretrained_model)
    load_param_into_net(network, param_dict)
    network.set_train(False)

    input_arr = (Tensor(np.ones(shape), dtype=ms.float32))

    export(network, input_arr, file_name='convlstm', file_format=args_opt.file_format)


if __name__ == '__main__':
    run_export()