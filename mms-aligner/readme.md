MMS-aligner通用时间戳
项目文件地址：/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/tools/aligner/mms-aligner 

修改`run.sh`中的`data_path`,`text_path`,`output_path`和`sample_rate`即可进行对齐
`data_path`为wav文件所在的文件夹地址
`text_path`为转录文本的地址，token之间用空格隔开，第一个为key，例如:`s07_7383 when i was a kid my father got this tape recorder`
`output_path`为保存对齐结果的文件地址，输出为json格式。
`sample_rate`为输入文件的采样率，默认16K

```sh
data_path=test_data/wavs
text_path=test_data/text
output_path=~/Downloads/buckeye_split/output.json
sample_rate=16000

uroman_path=uroman/bin/
model_path_name=data/ctc_alignment_mling_uroman_model.pt
dict_path_name=data/ctc_alignment_mling_uroman_model.dict

python local/pytorch/aligner.py -d $data_path \
    -t $text_path \
    -u  $uroman_path \
    -o data/output.json \
    --model_path_name $model_path_name \
    --dict_path_name $dict_path_name \
    --sample_rate $sample_rate

```