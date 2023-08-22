import torch
import torchaudio
import argparse
import time
import json
import glob
import os
from tqdm import tqdm

from text_normalization import text_normalize
from align_utils import (
    get_uroman_tokens,
    time_to_frame,
    load_model_dict,
    merge_repeats,
    get_spans,
    force_align,
    merge_words,
)

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_emissions(model, waveform):
    waveform = waveform.to(DEVICE)
    total_duration = len(waveform[0]) / SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
            ]

            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)
    return emissions, stride


def get_alignments(
    waveform,
    tokens,
    token_indices,
    model,
    dictionary,
    norm_transcripts
):
    # Generate emissions
    emissions, stride = generate_emissions(model, waveform)
    blank = dictionary["<blank>"]
    path = force_align(emissions, torch.Tensor(token_indices).to(DEVICE).long(), blank_id=blank)
    segments = merge_repeats(path,{v: k for k, v in dictionary.items()})
    tokens_segments = get_spans(tokens,segments)
    word_segments = merge_words(tokens_segments,tokens)
    ret_word_align = {i:{'token':norm_transcripts[i],'start':word_segments[i].start * stride / 1000,'end':word_segments[i].end * stride / 1000,'score':word_segments[i].score} for i in range(len(word_segments))}
    return ret_word_align

def load_text(text_filepath):
    text_dict = {}
    with open(text_filepath,'r') as f:
        for line in f:
            key,text = line.strip().split(' ',maxsplit=1)
            text_dict[key] = {}
            text_dict[key]['norm_transcript'] = text_normalize(text.strip()).split(' ')
    return text_dict

def process_text(text_dict,dictionary,uroman_path):
    """
    process text: normalization and uroman
    """
    all_norm_transcripts = []
    for key,text_item in text_dict.items():
        all_norm_transcripts = all_norm_transcripts + text_item['norm_transcript']
    all_uroman_tokens = get_uroman_tokens(all_norm_transcripts, uroman_path)
    start = 0
    for key,text_item in text_dict.items():
        text_item['uroman_tokens'] = all_uroman_tokens[start:start+len(text_item['norm_transcript'])]
        text_item['token_indices'] = [dictionary[c] for c in " ".join(text_item['uroman_tokens']).split(" ") if c in dictionary]
        start += len(text_item['norm_transcript'])
    assert start == len(all_uroman_tokens)
    return text_dict

    

def main(args):
    model, dictionary = load_model_dict(args.model_path_name,args.dict_path_name)
    model = model.to(DEVICE)
    output_folder = args.output_path
    text_dict = load_text(args.text_path)
    text_dict = process_text(text_dict,dictionary,uroman_path=args.uroman_path)

    scp_list = open(args.scp_path,'r').readlines()
    key_wav_list = [scp.strip().split(' ') for scp in scp_list]
    align_result = {}

    resampler = None
    if args.sample_rate != SAMPLING_FREQ:
        resampler = torchaudio.transforms.Resample(args.sample_rate, 16000)

    for key,wav in tqdm(key_wav_list,total=len(key_wav_list),desc="loading and align data"):
        key = wav.split('/')[-1].split('.')[0]
        waveform, _ = torchaudio.load(wav)
        assert _ == args.sample_rate
        if resampler:
            waveform = resampler(waveform)
        
        transcripts_item = text_dict.get(key,None)
        if transcripts_item is None:
            print(f"transcripts empty!!! key: {key}, drop out it.")
            continue
        
        norm_transcript = transcripts_item['norm_transcript']
        uroman_tokens = transcripts_item['uroman_tokens']
        token_indices = transcripts_item['token_indices']
        if token_indices == []:
            print(f"token_indices empty!!! key: {key}, drop out it.")
            continue

        word_aligns = get_alignments(
            waveform,
            uroman_tokens,
            token_indices,
            model,
            dictionary,
            norm_transcript
        )
        align_result[key] = word_aligns
        with open(os.path.join(output_folder,key+".json"),'w+') as output:
            print(json.dumps(word_aligns,ensure_ascii=False),file=output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and segment long audio files")
    parser.add_argument('--scp_path', type=str, default='', help='wav.scp file path for the align wavs')
    parser.add_argument('-t','--text_path', type=str, default='', help='text file path for the align wavs')
    parser.add_argument('-o','--output_path', type=str, default='', help='output file path for the align wavs')
    parser.add_argument("--sample_rate", type=int, default=16000, help="sample rate")
    parser.add_argument('--model_path_name',type=str,
                        default="/tmp/ctc_alignment_mling_uroman_model.pt",
                        help="the model path or model name")
    parser.add_argument('--dict_path_name',type=str,
                        default="/tmp/ctc_alignment_mling_uroman_model.dict",
                        help="the dict path or dict name")
    parser.add_argument("-u", "--uroman_path", type=str, default="eng", help="Location to uroman/bin")
    print("Using torch version:", torch.__version__)
    print("Using torchaudio version:", torchaudio.__version__)
    print("Using device: ", DEVICE)
    args = parser.parse_args()
    main(args)