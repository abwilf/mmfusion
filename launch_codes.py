from utils import *
a = '''
            CUDA_VISIBLE_DEVICES=1 python3 main.py --cross_utterance 1 --modality text,audio --tensors_path unique --labels_path test_data/val_labels.pk --transcripts_path test_data/transcripts.pk --audio_path test_data/mfb.pk --wav_dir test_data/wavs --overwrite_mfbs 1 --mode inference --evaluate_inference 1 --print_transcripts 1 --seq_len 150 --hffn_path val_model

'''
b = a.split('.py')[1].strip().split(' ')
print('"args": [' + ' '.join(lmap(lambda elt: '"' + '='.join(elt)+'",', lzip(b[::2], b[1::2])))[:-1] + ']')