from utils import *
a = '''
CUDA_VISIBLE_DEVICES=1 python3 main.py --cross_utterance 1 --modality text,audio --tensors_path unique --labels_path test_data/labels2.pk --transcripts_path test_data/transcripts.pk --audio_path test_data/mfb.pk --wav_dir test_data/wavs --mode inference --evaluate_inference 0 --print_transcripts 1
'''
b = a.split('main.py')[1].strip().split(' ')
print('"args": [' + ' '.join(lmap(lambda elt: '"' + '='.join(elt)+'",', lzip(b[::2], b[1::2])))[:-1] + ']')