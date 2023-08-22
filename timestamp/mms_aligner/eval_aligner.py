import torch
import torchaudio
import argparse
import time
from boltons import fileutils
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
    model,
    dictionary,
    norm_transcripts
):
    # Generate emissions
    emissions, stride = generate_emissions(model, waveform)
    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file ")
        token_indices = []
    if token_indices == []:
        print(f"Empty transcript, due to all chars being digital (uroman) !!!!! for audio file ,tokens: {tokens}\tnorm_transcripts:{norm_transcripts}")
        return [], stride
    blank = dictionary["<blank>"]
    path = force_align(emissions, torch.Tensor(token_indices).to(DEVICE).long(), blank_id=blank)
    segments = merge_repeats(path,{v: k for k, v in dictionary.items()})
    tokens_segments = get_spans(tokens,segments)
    word_segments = merge_words(tokens_segments,tokens)
    ret_word_align = [[norm_transcripts[i],word_segments[i].start,word_segments[i].end,word_segments[i].score] for i in range(len(word_segments))]
    return ret_word_align,stride


def main(args):
    model, dictionary = load_model_dict(args.model_path_name,args.dict_path_name)
    model = model.to(DEVICE)

    wavs = list(fileutils.iter_find_files(args.val_path, "*.wav"))
    all_cnt = 0
    tp_cnt = 0
    words_list = []
    transcripts_list = []
    norm_transcripts_list = []
    tokens_list = []
    wavform_list = []
    all_norm_transcripts = []
    for wav in tqdm(wavs,total=len(wavs),desc="loading data"):
        waveform, _ = torchaudio.load(wav)
        assert _ == SAMPLING_FREQ
        word_fn = wav.replace("wav", "word")
        words = [w.strip().split() for w in open(word_fn, 'r').readlines()]
        transcripts = [word[-1] for word in words]
        norm_transcripts = [text_normalize(line.strip(), args.lang) for line in transcripts]

        wavform_list.append(waveform)
        words_list.append(words)
        transcripts_list.append(transcripts)
        norm_transcripts_list.append(norm_transcripts)
        for char in norm_transcripts:
            all_norm_transcripts.append(char)
    
    all_norm_lens =[0] + [len(_) for _ in norm_transcripts_list]
    for i in range(1,len(all_norm_lens)):
        all_norm_lens[i] += all_norm_lens[i-1]
    start_time = time.time()
    all_tokens = get_uroman_tokens(all_norm_transcripts, args.uroman_path, args.lang)
    end_time = time.time()
    print(f"get_uroman_tokens time: {end_time-start_time}")
    tokens_list = [all_tokens[all_norm_lens[i]:all_norm_lens[i+1]] for i in range(len(all_norm_lens)-1)]
    word_aligns_list = []
    for words,transcripts,norm_transcripts,tokens,waveform in tqdm(zip(words_list,transcripts_list,norm_transcripts_list,tokens_list,wavform_list),total=len(wavs),desc="predicting"):
        word_aligns, stride = get_alignments(
            waveform,
            tokens,
            model,
            dictionary,
            norm_transcripts
        )
        word_aligns_list.append(word_aligns)

    empty_tokens_cnt = 0
    for words,word_aligns in tqdm(zip(words_list,word_aligns_list),total=len(word_aligns_list),desc="evalution"):
        if word_aligns == []:
            empty_tokens_cnt += 1
            continue
        for word,word_align in zip(words,word_aligns):
            all_cnt += 2
            if int(word[0]) // 320  - 5 <= word_align[1] <= int(word[0]) // 320 + 5:
                tp_cnt += 1
            if int(word[1]) // 320 - 5 <= word_align[2] <= int(word[1]) // 320+ 5:
                tp_cnt += 1
        all_cnt += 4
        tp_cnt += 2
        if int(words[0][0]) // 320 - 5 <= word_aligns[0][1] <= int(words[0][0]) // 320 + 5:
            tp_cnt += 1
        if int(words[-1][0]) // 320 - 5 <= word_aligns[-1][2] <= int(words[-1][0]) // 320 + 5:
            tp_cnt += 1
    print(f"{tp_cnt=}\t{all_cnt=}")
    print(f"{tp_cnt/all_cnt:6.2%}")
    print(f"{empty_tokens_cnt=}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and segment long audio files")
    parser.add_argument('-v','--val_path', type=str, default='',
                        help='dir of .wav, .wrd files for val data')
    parser.add_argument("-l", "--lang", type=str, default="eng", help="ISO code of the language")
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
