from boltons import fileutils
import numpy as np
import torch
import torchaudio
import tqdm


def load_data(path,return_keys=False,return_words=False):
	wavs = list(fileutils.iter_find_files(path, "*.wav"))
	all_wavs = []
	all_bounds = []
	all_wavs_keys = []
	all_wavs_words = []
	rp = np.random.permutation(len(wavs))
	wavs = [wavs[i] for i in rp]
	print("start load data")
	for wav in tqdm.tqdm(wavs,total=len(wavs)):
		word_fn = wav.replace("wav", "word")
		words = open(word_fn, 'r').readlines()
		words = [w.strip().split() for w in words]
		bounds = [(int(w[0]), int(w[1])) for w in words]
		waveform, sample_rate = torchaudio.load(wav)
		if len(bounds) > 0:
			all_wavs.append(waveform)
			all_bounds.append(bounds)
			all_wavs_keys.append(wav.split('/')[-1].split('.')[0])
			all_wavs_words.append([_[2] for _ in words])
	return_list = [all_wavs, all_bounds]
	if return_keys:
		return_list.append(all_wavs_keys)
	if return_words:
		return_list.append(all_wavs_words)
	return return_list


def get_embed_hubert(wavs, model,DEVICE):
	es = []
	for waveform in tqdm.tqdm(wavs,total=len(wavs),desc="loading embed"):
		waveform = waveform.to(DEVICE)
		es.append(model(waveform).last_hidden_state.cpu().detach().numpy()[0])
	return es

def get_embed_wav2vec(wavs, model,DEVICE,extract_layer=-1):
	es = []
	with torch.no_grad():
		model.eval()
		for waveform in tqdm.tqdm(wavs,total=len(wavs),desc="loading embed"):
			waveform = waveform.to(DEVICE)
			x, _ = model.extract_features(waveform)
			x = x[extract_layer]
			es.append(x.data.cpu().detach().numpy()[0])
	return es

def get_bounds(boundaries):
	l = []
	l.append(boundaries[0][0])
	for i in range(len(boundaries)-1):
		l.append((boundaries[i][1] + boundaries[i+1][0])//2)
	l.append(boundaries[-1][1])
	return l
