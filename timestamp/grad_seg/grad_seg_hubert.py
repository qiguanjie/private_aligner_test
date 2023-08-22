import json
import torch
import argparse
import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from transformers import AutoProcessor, HubertModel,HubertForCTC,Wav2Vec2Processor
from utils.metric import boundary_metric
from utils.data_loader import (load_data,get_bounds,get_embed_hubert)
import os
import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='GradSeg word segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_path', type=str, default='',
                        help='dir of .wav, .wrd files for training data')
    parser.add_argument('--val_path', type=str, default='',
                        help='dir of .wav, .wrd files for val data')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    
    parser.add_argument('--tolerance', type=int, default=2,
                        help='tolerance for the eval')

    parser.add_argument('--min_separation', type=int, default=4,
                        help='min separation between words')
    parser.add_argument('--target_perc', type=int, default=40,
                        help='target quantization percentile')

    parser.add_argument('--frames_per_word', type=int,
                        default=10, help='5 words in a second')
    parser.add_argument('--loss', type=str, default='ridge',
                        help='ridge || logres')
    parser.add_argument('--C', type=float, default=1.0,
                        help='logistic regression parameter')
    parser.add_argument('--reg', type=float, default=1e4,
                        help='ridge regularization')

    parser.add_argument('--result_file', type=str,
                        help='the file path for exporting the result')

    parser.add_argument('--hubert_path', type=str, default='facebook/hubert-large-ls960-ft',help='hubert model path')
    args = parser.parse_args()
    return args

def get_grad_mag(e):
    e = np.pad(e, 1, mode='reflect')
    e = e[2:] - e[:-2]
    mag = e ** 2
    return mag.mean(1)


def get_seg(d, num_words, min_separation):
    idx = np.argsort(d)
    selected = []
    for i in idx[::-1]:
        if len(selected) >= num_words:
            break
        if len(selected) == 0 or (np.abs(np.array(selected) - i)).min() > min_separation:
            selected.append(i)
    return np.sort(np.array(selected))
    

def main():
    args = get_args()
    print(args)
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    # init seeds
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model init
    model = HubertModel.from_pretrained(args.hubert_path)
    model = model.to(DEVICE)

    # Here, we can train on the training set and evaluate on the development or test set separately. 
    # However, due to the specific nature of the forced alignment task, we can directly train and test on the test set.

    # train_wavs, train_bounds = load_data(args.train_path, args.train_n)
    val_wavs, val_bounds, wav_keys, wav_words = load_data(args.val_path,return_keys=True, return_words=True)
    train_wavs, train_bounds = val_wavs, val_bounds

    train_e = get_embed_hubert(train_wavs, model,DEVICE)
    val_e = train_e

    frames_per_embedding = 160  # (not 320), instead of multiplying by 2 later
    print("frame duration (s): %f" % (frames_per_embedding/16000))

    ds = []
    for idx in range(len(train_e)):
        d = get_grad_mag(train_e[idx])
        ds.append(d)
        
    ds = np.concatenate(ds)
    th = np.percentile(ds, args.target_perc)
    targets = ds > th
    if args.loss == 'ridge':
        clf = Ridge(alpha=args.reg)
    else:
        clf = LogisticRegression(C=args.C, max_iter=1000)
    train_e_np = np.concatenate(train_e)
    mu = train_e_np.mean(0)[None, :]
    std = train_e_np.std(0)[None, :]
    clf.fit((train_e_np - mu)/std, targets)

    ref_bounds = []
    seg_bounds = []
    result_file = None
    if args.result_file is not None:
        result_file = open(args.result_file, 'w+')
    for idx in tqdm.tqdm(range(len(val_e))):
        if args.loss == 'logres':
            d = clf.predict_proba((val_e[idx] - mu)/std)[:, 1]
        else:
            d = clf.predict((val_e[idx] - mu)/std)
        num_words = len(wav_words[idx]) -1
        p = get_seg(d, num_words, args.min_separation)

        p = p * 2
        p = np.minimum(p, 2*(len(d)-1))
        p = p.astype('int')

        boundaries = np.array(get_bounds(val_bounds[idx])) // frames_per_embedding
        boundaries = np.minimum(boundaries[1:-1], (len(d)-1)*2)

        ref_bound = np.zeros(len(d)*2)
        seg_bound = np.zeros(len(d)*2)
        ref_bound[boundaries] = 1
        seg_bound[p] = 1

        ref_bound[-1] = 1
        seg_bound[-1] = 1
        ref_bounds.append(ref_bound)
        seg_bounds.append(seg_bound)
        if result_file is not None:
            now_result = {}
            now_result['key'] = wav_keys[idx]
            now_result['word_list'] = wav_words[idx]
            now_result['boundaries'] = p.tolist()
            print(json.dumps(now_result, ensure_ascii=False), file=result_file)

    result_file.close()
    precision = boundary_metric(ref_bounds, seg_bounds, args.tolerance)
    print(f"Final result: {precision=:6.2%}")

if __name__ == '__main__':
    main()