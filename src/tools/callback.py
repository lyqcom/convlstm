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
"""callback function"""

from mindspore.train.callback import Callback

from mindspore.ops import functional as F



class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, src_url, train_url, save_freq=50):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.src_url = src_url
        self.train_url = train_url
        self.save_freq = save_freq
        self.best_acc = 0.

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        result = self.model.eval(self.eval_dataset)
        if result['MSE'] < self.best_acc:
            self.best_acc = result["MSE"]
        print("epoch: %s MSE: %s, best MSE is %s , MAE is %s" %
              (cb_params.cur_epoch_num, result["MSE"], self.best_acc,result["MAE"]), flush=True)
        '''   
        if True:
            import moxing as mox
            if cur_epoch_num % self.save_freq == 0:
                mox.file.copy_parallel(src_url=self.src_url, dst_url=self.train_url)
        '''   

class lr_(Callback):
    """StopAtTime"""
    def __init__(self, fc = 0.5,times = 4):
        """init"""
        super(lr_, self).__init__()
        self.fc =  fc
        self.times = times
        self.num = 0
        self.best = 1
    def begin(self, run_context):
        """begin"""


    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        
        optimizer = cb_params.optimizer

        arr_lr = cb_params.optimizer.learning_rate.asnumpy()
        if loss[0].asnumpy() >= self.best:
            self.num = self.num + 1
            if self.num >= self.times :
                new_lr = arr_lr*self.fc
                F.assign(cb_params.optimizer.learning_rate, Tensor(new_lr, mstype.float32))
                print('change lr is:',new_lr)
                self.num = 0
                
        else:
            self.best = loss[0].asnumpy()
            
            self.num = 0   
        print('loss is :',loss[0].asnumpy(),'lr is :',arr_lr) 