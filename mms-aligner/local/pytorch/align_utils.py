import re 
import os
import torch
import tempfile
import math
import torchaudio
from dataclasses import dataclass
from torchaudio.models import wav2vec2_model

# iso codes with specialized rules in uroman
special_isos_uroman = "ara, bel, bul, deu, ell, eng, fas, grc, ell, eng, heb, kaz, kir, lav, lit, mkd, mkd2, oss, pnt, pus, rus, srp, srp2, tur, uig, ukr, yid".split(",")
special_isos_uroman = [i.strip() for i in special_isos_uroman]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def get_uroman_tokens(norm_transcripts, uroman_root_dir, iso = None):
    tf = tempfile.NamedTemporaryFile()  
    tf2 = tempfile.NamedTemporaryFile()  
    with open(tf.name, "w") as f:
        for t in norm_transcripts:
            f.write(t + "\n")

    assert os.path.exists(f"{uroman_root_dir}/uroman.pl"), "uroman not found"
    cmd = f"perl {uroman_root_dir}/uroman.pl"
    if iso in special_isos_uroman:
        cmd += f" -l {iso} "
    cmd +=  f" < {tf.name} > {tf2.name}" 
    
    os.system(cmd)
    outtexts = []
    with open(tf2.name) as f:
        for line in f:
            line = " ".join(line.strip())
            line =  re.sub(r"\s+", " ", line).strip()
            outtexts.append(line)
    assert len(outtexts) == len(norm_transcripts)
    uromans = []
    for ot in outtexts:
        uromans.append(normalize_uroman(ot))
    return uromans


@dataclass
class Point:
    token_indice: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float


    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_indice == path[i2].token_indice:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(Segment(idx_to_token_map[path[i1].token_indice], path[i1].time_index,path[i2 - 1].time_index + 1,score))
        i1 = i2
    return segments

def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)

def load_model_dict(model_path_name,dict_path_name):
    # model_path_name = "/tmp/ctc_alignment_mling_uroman_model.pt"

    print("Downloading model and dictionary...")
    if os.path.exists(model_path_name):
        print("Model path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt",
            model_path_name,
        )
        assert os.path.exists(model_path_name)
    state_dict = torch.load(model_path_name, map_location="cpu")

    model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        aux_num_out=31,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # dict_path_name = "/tmp/ctc_alignment_mling_uroman_model.dict"
    if os.path.exists(dict_path_name):
        print("Dictionary path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt",
            dict_path_name,
        )
        assert os.path.exists(dict_path_name)
    dictionary = {}
    with open(dict_path_name) as f:
        dictionary = {l.strip(): i for i, l in enumerate(f.readlines())}

    return model, dictionary

def get_spans(tokens, segments):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    sil = "<blank>"
    for (seg_idx, seg) in enumerate(segments):
        if(tokens_idx == len(tokens)):
           assert(seg_idx == len(segments) - 1)
           assert(seg.label == '<blank>')
           continue
        cur_token = tokens[tokens_idx].split(' ')
        ltr = cur_token[ltr_idx]
        if seg.label == "<blank>": continue
        # assert(seg.label == ltr) # TODO 连续相同的seg
        if(ltr_idx) == 0: start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                    intervals.append((seg_idx, seg_idx))
                    tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for (idx, (start, end)) in enumerate(intervals):
        span = segments[start:end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == sil:
                pad_start = prev_seg.start if (idx == 0) else int((prev_seg.start + prev_seg.end)/2)
                span = [Segment(sil, pad_start, span[0].start)] + span
        if end+1 < len(segments):
            next_seg = segments[end+1]
            if next_seg.label == sil:
                pad_end = next_seg.end if (idx == len(intervals) - 1) else math.floor((next_seg.start + next_seg.end) / 2)
                span = span + [Segment(sil, span[-1].end, pad_end)]
        spans.append(span)
    return spans

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(DEVICE)
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

def backtrack(trellis, emission, token_indices, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, token_indices[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, token_indices[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(int(token_indices[j - 1]), t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        breakpoint()
        # raise ValueError("Failed to align")
        
    return path[::-1]

def force_align(emissions, token_indices, blank_id=0):
    trellis = get_trellis(emissions, token_indices, blank_id=blank_id)
    path = backtrack(trellis, emissions, token_indices, blank_id=blank_id)
    return path

def merge_words(tokens_segments,tokens):
    words = []
    for (token_seg,token) in zip(tokens_segments,tokens):
        score = sum([char.score for char in token_seg]) / len(token_seg)
        words.append(Segment(token,token_seg[0].start,token_seg[-1].end,score))
    return words

def convert_16k(resampler,input_wav,input_sample):
    """
    Convert input wav to 16k sample rate
    """
    resampler = torchaudio.transforms.Resample(input_sample, 16000)
    waveform_16k = resampler(input_wav)
    return waveform_16k