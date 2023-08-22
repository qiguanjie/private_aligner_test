"""
Alignment information is obtained by evenly splitting the input characters.
Assume that all audio clips have silence at the beginning and the end.
"""

import argparse
import numpy as np
import tqdm
from utils.metric import boundary_metric
from utils.data_loader import (load_data,get_bounds)


def get_args():
    parser = argparse.ArgumentParser(description='evenly split word segmentation')
    parser.add_argument('--val_path', type=str, help='dir of .wav, .wrd files for val data')
    parser.add_argument('--vad', action='store_true', help='whether to use vad')
    parser.add_argument('--char', action='store_true', help='whether to use char length to split')
    parser.add_argument('--vad_file', type=str, help='vad file path')
    parser.add_argument('--tolerance', type=int, default=2, help='tolerance for the eval')
    
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    assert (not args.vad) or args.vad_file

    vad_dict = {}
    # load the vad file, if use the vad
    if args.vad:
        vad_file = open(args.vad_file)
        for line in vad_file:
            line = line.strip()
            key,vad_str = line.split("  ")
            vad_list = eval(vad_str.replace("[ ","[").replace(" ",","))
            vad_dict[key] = vad_list

    # the frame counter for the per embedding
    frames_per_embedding = 160

    val_wavs, val_bounds, val_wavs_keys,val_wavs_words = load_data(args.val_path, 
                                                                   return_keys=True,
                                                                   return_words=True)
    ref_bounds = []
    seg_bounds = []
    for idx in tqdm.tqdm(range(len(val_wavs_keys)),desc="predicting"):
            words = val_wavs_words[idx]
            evenly_split_boundaries = []
            wav_sum_len = len(val_wavs[idx][0]) // frames_per_embedding
            start = 0
            end = wav_sum_len-1

            # Load the ground truth boundaries, e.g. [0, 0, 1, 0, ...], where '1' represents a boundary and each element corresponds to 10ms.
            boundaries = np.array(get_bounds(val_bounds[idx])) // frames_per_embedding
            boundaries = boundaries.tolist()

            # Skip if there is no silence segment at the beginning or end, and the end boundary is out of the audio range
            if boundaries[0] == 0 or boundaries[-1] == end or boundaries[-1] > end:
                continue
            
            if args.vad:
                # with vad
                now_vad = vad_dict[val_wavs_keys[idx]]
                end = min(len(now_vad)-1,end)
                # update the start and end position using vad
                while start < len(now_vad) and now_vad[start] == 0:
                    start += 1
                while end >= 0 and now_vad[end] == 0:
                    end -= 1

                if args.char:
                    # use char length to split
                    
                    words_lens = [len(_) for _ in words]
                    splits_cnt = sum(words_lens)
                    # If there is no silence at the end, increment splits_cnt by average word length to unify the processing
                    if end == wav_sum_len-1:
                        aver_word_len = sum(words_lens) / sum(words_lens)
                        splits_cnt += aver_word_len
                        words_lens.append(aver_word_len)
                    each_duration = (end-start) / splits_cnt
                    _i = start
                    for j in range(len(words_lens)):
                        evenly_split_boundaries.append(int(_i))
                        _i += each_duration * words_lens[j]
                    evenly_split_boundaries.append(end)
                else:
                    splits_cnt = len(words)
                    # If there is no silence at the end, increment splits_cnt by 1 to unify the processing
                    if end == wav_sum_len-1:
                        splits_cnt += 1
                    each_duration = (end-start) / splits_cnt
                    _i = start
                    for j in range(splits_cnt):
                        evenly_split_boundaries.append(int(_i))
                        _i += each_duration
                    evenly_split_boundaries.append(end)
            else:
                # without vad
                splits_cnt = len(words) + 2
                each_duration = wav_sum_len / splits_cnt
                _i = each_duration
                for j in range(splits_cnt-1):
                    evenly_split_boundaries.append(int(_i))
                    _i += each_duration
                # evenly_split_boundaries.append(end)
            now_len = (len(val_wavs[idx][0]) // frames_per_embedding)
            ref_bound = np.zeros(now_len)
            seg_bound = np.zeros(now_len)
            
            ref_bound[boundaries] = 1
            seg_bound[evenly_split_boundaries] = 1
            ref_bound[-1] = 1
            seg_bound[-1] = 1
            assert ref_bound.tolist().count(1) == seg_bound.tolist().count(1)
            ref_bounds.append(ref_bound)
            seg_bounds.append(seg_bound)
            
    precision = boundary_metric(ref_bounds, seg_bounds, args.tolerance)
    print(f"Final result: {precision=:6.2%}")

if __name__ == '__main__':
    main()