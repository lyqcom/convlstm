#!/bin/bash
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "Usage: bash scripts/run_eval.sh [DEVICE_ID] [DATAPATH] [CKPTPATH] [BACTHSIZE]"
echo "for example: bash cripts/run_eval.sh 0 '/home/ma-user/work/data' 'path/xx.ckpt' 32"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 4 ]
then
    echo "Usage: bash scripts/run_eval.sh [DEVICE_ID] [DATAPATH] [CKPTPATH] [BACTHSIZE]"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export DEVICE_ID=$1
DATAPATH=$2
CKPTPATH=$3
BACTHSIZE=$4



rm -rf LOG$1
mkdir ./LOG$1
cp ./*.py ./LOG$1
cp -r ./src ./LOG$1
cd ./LOG$1 || exit

echo "start eval for device $1"

env > env.log

python test.py      
--workroot=$DATAPATH \
--pretrained_model=$CKPTPATH \
--batch_szie=$BACTHSIZE > log.txt 2>&1 &

cd ../