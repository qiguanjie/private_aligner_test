scp_path=test_data/wav.scp
text_path=test_data/text
output_path=test_data/align_result
sample_rate=16000

mkdir -p $output_path

uroman_path=uroman/bin/
model_path_name=data/ctc_alignment_mling_uroman_model.pt
dict_path_name=data/ctc_alignment_mling_uroman_model.dict

python local/pytorch/aligner.py --scp_path $scp_path \
    -t $text_path \
    -u  $uroman_path \
    -o $output_path \
    --model_path_name $model_path_name \
    --dict_path_name $dict_path_name \
    --sample_rate $sample_rate