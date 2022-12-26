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
echo "bash scripts/run_single_train.sh DEVICE_ID BACTHSIZE EPOCHS_NUMS DATAPATH SAVEPATH"
echo "for example: bash scripts/run_single_train.sh 0 32 100 ./data ./model"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 5 ]
then
    echo "Usage: bash scripts/run_single_train.sh [DEVICE_ID] [BACTHSIZE] \
[EPOCHS_NUMS] [DATAPATH] [SAVEPATH]"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export DEVICE_ID=$1
BACTHSIZE=$2
EPOCHS_NUMS=$3
DATAPATH=$4
SAVEPATH=$5

rm -rf LOG$1
mkdir ./LOG$1
cp ./*.py ./LOG$1
cp -r ./src ./LOG$1
cd ./LOG$1 || exit

echo "start training for device $1"

env > env.log


python train_.py  \
--batch_size=$BACTHSIZE  \
--epochs=$EPOCHS_NUMS \
--data_path=$DATAPATH \
--train_path=$SAVEPATH > log.txt 2>&1 &


cd ../