# 通用时间戳方案尝试

项目地址：[【WIP】通用时间戳功能](https://km.sankuai.com/collabpage/1593781949)
方案汇总：[通用时间戳方案汇总](https://km.sankuai.com/collabpage/1696478197)

本仓库下包含：均匀分布、使用vad的均匀分布、使用vad结合字符长度的均匀分布、GradSeg无监督方案、Meta的MMS align方案的代码及其评估脚本。

所有示例执行脚本的执行目录均在当前目录下。
## 均匀分布方案
代码地址：`split_evenly/split_evenly.py`，均匀分布、使用vad的均匀分布、使用vad结合字符长度的均匀分布三种方式都写在这个文件中了，通过args进行控制。

### 均匀分布
执行脚本示例：

```sh
export PYTHONPATH=./:$PYTHONPATH

# alignment tolerance, default is 10, means 100ms
tolerance=10 
# the filepath for the test folder. (wavs)
val_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/data/asr/raw/buckeye/test
# the filepath for the vad file path. 
vad_file=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/users/zhanglei253/alignment_about/alignment_dev/grad_seg/data/test_vad.txt

python split_evenly/split_evenly.py --tolerance $tolerance \
    --val_path $val_path \
    --vad_file $vad_file
```

### 使用vad的均匀分布

执行脚本示例：

```sh
export PYTHONPATH=./:$PYTHONPATH

# alignment tolerance, default is 10, means 100ms
tolerance=10 
# the filepath for the test folder. (wavs)
val_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/data/asr/raw/buckeye/test
# the filepath for the vad file path. 
vad_file=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/users/zhanglei253/alignment_about/alignment_dev/grad_seg/data/test_vad.txt

python split_evenly/split_evenly.py --tolerance $tolerance \
    --val_path $val_path \
    --vad_file $vad_file \
    --vad
```

### 使用vad结合字符长度的均匀分布
执行脚本示例：
```sh
export PYTHONPATH=./:$PYTHONPATH

# alignment tolerance, default is 10, means 100ms
tolerance=10 
# the filepath for the test folder. (wavs)
val_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/data/asr/raw/buckeye/test
# the filepath for the vad file path. 
vad_file=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/users/zhanglei253/alignment_about/alignment_dev/grad_seg/data/test_vad.txt

python split_evenly/split_evenly.py --tolerance $tolerance \
    --val_path $val_path \
    --vad_file $vad_file \
    --vad \
    --char
```

## GradSeg方案

这里可以在train集训练，然后分别使用dev和test集评估等。由于是对齐任务，这里示例脚本是直接使用测试集进行和评估的，如果需要分别训练测试，在数据加载部分同步修改一下就好。

### 使用torchaudio上的wav2vec进行特征抽取
执行脚本示例：
```sh
export PYTHONPATH=./:$PYTHONPATH

# the filepath for the test folder. (wavs)
val_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/data/asr/raw/buckeye/test
python grad_seg/grad_seg_wav2vec.py --min_separation 3 \
    --gpu -1 \
    --reg 1e7  \
    --target_perc 20 \
    --frames_per_word 15 \
    --val_path $val_path  \
    --tolerance 10 
```

### 使用HuBERT进行特征抽取

这里使用的HuBERT模型为：[facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)

执行脚本示例：
```sh
export PYTHONPATH=./:$PYTHONPATH

# alignment tolerance, default is 10, means 100ms
tolerance=10 
# the filepath for the test folder. (wavs)
val_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/data/asr/raw/buckeye/test
hubert_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/tools/aligner/mms-aligner/data/hubert-large-ls960-ft
python grad_seg/grad_seg_hubert.py --min_separation 3 \
    --gpu -1 \
    --reg 1e7  \
    --target_perc 20 \
    --frames_per_word 15 \
    --val_path $val_path  \
    --tolerance $tolerance \
    --hubert_path $hubert_path
```

## MMS align方案

uroman地址：[uroman](https://github.com/isi-nlp/uroman)
预训练对齐模型地址：[ctc_alignment_mling_uroman/model.pt](https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt)
字典地址：[ctc_alignment_mling_uroman/dictionary.txt](https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt)
执行脚本示例：
```sh
export CUDA_VISIBLE_DEVICES=3

val_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/data/asr/raw/buckeye/test
uroman_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/tools/aligner/mms-aligner/uroman/bin
model_path_name=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/tools/aligner/mms-aligner/data/ctc_alignment_mling_uroman_model.pt
dict_path_name=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/tools/aligner/mms-aligner/data/ctc_alignment_mling_uroman_model.dict
python mms_aligner/eval_aligner.py -l cnm \
    -v $val_path \
    -u  $uroman_path \
    --model_path_name $model_path_name \
    --dict_path_name $dict_path_name
```