from utils import *
from consts import *

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

sys.path.append(MMSDK_PATH)
from mmsdk import mmdatasdk
sys.path.append(STANDARD_GRID_PATH)
import standard_grid

import copy
import hashlib
import soundfile as sf
import librosa
import multiprocessing.dummy as mp_thread
import multiprocessing as mp_proc
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import requests, atexit
import random
from notebook_util import setup_no_gpu, setup_one_gpu, setup_gpu
from utils import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Progbar
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback, TensorBoard
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import l1_l2, l2
import tensorflow_hub as hub
import tensorflow_text as text

import librosa
import soundfile as sf
import scipy.io.wavfile as wav
import subprocess

sys.path.append(DEEPSPEECH_PATH)
import convert

from transcribe import *
from speaker_verification import *
from models import *


metadata_template = { "root name": '', "computational sequence description": '', "computational sequence version": '', "alignment compatible": '', "dataset name": '', "dataset version": '', "creator": '', "contact": '', "featureset bib citation": '', "dataset bib citation": ''}

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 
map_name_to_handle = { 'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', 'bert_en_cased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3', 'bert_multi_cased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3', 'small_bert/bert_en_uncased_L-2_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1', 'small_bert/bert_en_uncased_L-2_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1', 'small_bert/bert_en_uncased_L-2_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1', 'small_bert/bert_en_uncased_L-2_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1', 'small_bert/bert_en_uncased_L-4_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1', 'small_bert/bert_en_uncased_L-4_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1', 'small_bert/bert_en_uncased_L-4_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1', 'small_bert/bert_en_uncased_L-4_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1', 'small_bert/bert_en_uncased_L-6_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1', 'small_bert/bert_en_uncased_L-6_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1', 'small_bert/bert_en_uncased_L-6_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1', 'small_bert/bert_en_uncased_L-6_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1', 'small_bert/bert_en_uncased_L-8_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1', 'small_bert/bert_en_uncased_L-8_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1', 'small_bert/bert_en_uncased_L-8_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1', 'small_bert/bert_en_uncased_L-8_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1', 'small_bert/bert_en_uncased_L-10_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1', 'small_bert/bert_en_uncased_L-10_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1', 'small_bert/bert_en_uncased_L-10_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1', 'small_bert/bert_en_uncased_L-10_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1', 'small_bert/bert_en_uncased_L-12_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1', 'small_bert/bert_en_uncased_L-12_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1', 'small_bert/bert_en_uncased_L-12_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1', 'small_bert/bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1', 'albert_en_base': 'https://tfhub.dev/tensorflow/albert_en_base/2', 'electra_small': 'https://tfhub.dev/google/electra_small/2', 'electra_base': 'https://tfhub.dev/google/electra_base/2', 'experts_pubmed': 'https://tfhub.dev/google/experts/bert/pubmed/2', 'experts_wiki_books': 'https://tfhub.dev/google/experts/bert/wiki_books/2', 'talking-heads_base': 'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1', }
map_model_to_preprocess = { 'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'bert_en_cased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2', 'small_bert/bert_en_uncased_L-2_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-2_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-2_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-2_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-4_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-4_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-4_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-4_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-6_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-6_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-6_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-6_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-8_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-8_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-8_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-8_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-10_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-10_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-10_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-10_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-12_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-12_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-12_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'bert_multi_cased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2', 'albert_en_base': 'https://tfhub.dev/tensorflow/albert_en_preprocess/2', 'electra_small': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'electra_base': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'experts_pubmed': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'experts_wiki_books': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'talking-heads_base': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', }
tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]


######### MFB CREATION ######### 
# Parameters and code to create mfbs from wav files is below
# get_mfb gets MFBs of shape (timesteps, 40) from a single wav
# get_mfbs does this with a clamp value and z normalization using statistics from all mfbs being considered
# get_mfb_intervals gets the timestamps associated with each 40 dimensional vector; timestamps are used in alignment with the lexical modality using CMU-Multimodal-SDK
# deploy_unaligned_mfb_csd creates all mfbs from a wav directory in args['wav_dir'], parallelized across threads, saves pickle file (can load with load_pk) in args['audio_path']

n_mels = 40
n_fft = 2048
hop_length = 160 # mfbs are extracted in intervals of .1 second
fmin = 0
fmax = None
SR = 16000
n_iter = 32
MFB_WIN_STEP = .01
EPS = 1e-6
clampVal = 3.0

def get_mfb(wav_file):
    y, sr = librosa.load(wav_file, sr=SR)
    y = librosa.effects.preemphasis(y, coef=0.97)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax, htk=False)
    return mel_spec.T

def get_mfbs(wav_file):
    mfb = get_mfb(wav_file)
    mfb = (mfb - mfb_stats['mean']) / (mfb_stats['std'] + EPS)
    
    mfb[mfb>clampVal] = clampVal
    mfb[mfb<-clampVal] = -clampVal
    return mfb

def get_mfb_intervals(end, step):
    end = trunc(end*100, decs=2)
    step = trunc(step*100, decs=2)

    a = np.arange(0, end, step)

    a = trunc(a / 100, decs=2)
    end = trunc(end/100, decs=2)
    step = trunc(step/100, decs=2)

    b = np.concatenate([a[1:], [a[-1] + step]], axis=0)
    return np.vstack([a,b]).T
 
data = {}
def add_unaligned_mfb(video_key):
    wav_path = join(args['wav_dir'], f'{video_key}.wav')
    mfbs = get_mfbs(wav_path)
    intervals = get_mfb_intervals(MFB_WIN_STEP*mfbs.shape[0], MFB_WIN_STEP)
    if not mfbs.shape[0] == intervals.shape[0]:
        save_pk('temp.pk', {'mfbs': mfbs, 'intervals': intervals})
        assert False, 'See temp.pk for failure details'

    data[video_key] = {
        'features': mfbs,
        'intervals': intervals
    }

def get_mfb_stats(video_key):
    wav_path = join(args['wav_dir'], f'{video_key}.wav')
    mfb = get_mfb(wav_path)
    mean, std = np.mean(mfb), np.std(mfb)
    mfb_stats['mean'].append(mean)
    mfb_stats['std'].append(std)
    mfb_stats['length'].append(mfb.shape[0])

def deploy_unaligned_mfb_csd(check_labels=True):
    csd_name = f'mfb_temp'

    wav_keys = lmap(lambda elt: elt.split('/')[-1].split('.wav')[0], glob(join(args['wav_dir'], '*.wav')))
    if check_labels:
        labels = get_compseq(args['labels_path'], 'labels')
        label_keys = lkeys(labels)
        assert subset(label_keys, wav_keys), f'The keys of the videos in the wav directory {args["wav_dir"]} must be a subset of the keys in the labels file {args["labels_path"]}. e.g.: if there exists a label with the key "abc", there should be a corresponding "abc.wav" in the wav directory'

    if exists(args['audio_path']) and not args['overwrite_mfbs']:
        print(f'MFBs exist in {args["audio_path"]}.  Moving on...')
        return args["audio_path"]

    print(f'Getting global statistics over mfbs...\n')
    num_workers = 5
    pool = mp_thread.Pool(num_workers)

    global mfb_stats
    mfb_stats = {
        'mean': [],
        'std': [],
        'length': [],
    }

    # non parallelized
    # for wav_key in tqdm(wav_keys[:10]):
    #     get_mfb_stats(wav_key)
    # exit()

    for _ in tqdm(pool.imap_unordered(get_mfb_stats, wav_keys), total=len(wav_keys)):
        pass
    pool.close() 
    pool.join()

    mfb_stats['mean'] = np.mean(mfb_stats['mean'])
    mfb_stats['std'] = np.mean(mfb_stats['std'])

    print(f'Mapping {args["wav_dir"]} to unaligned mfbs in {args["audio_path"]}...\n')
    num_workers = 5
    pool = mp_thread.Pool(num_workers)

    # # non parallelized
    # for wav_key in tqdm(wav_keys[:10]):
    #     add_unaligned_mfb(wav_key)
    # exit()

    for _ in tqdm(pool.imap_unordered(add_unaligned_mfb, wav_keys), total=len(wav_keys)):
        pass
    pool.close() 
    pool.join()

    save_pk(args['audio_path'].replace('.csd', '.pk'), data)
    return args["audio_path"]
######### 


######### BERT EMBEDDINGS #########
# 
# text_tensor is a tensor of shape (num_utterances, seq_len), where all unused places contain '0.0' (a quirk of CMU-Multimodal-SDK alignment with strings)
# e.g.: if you had a dataset with two utterances and a sequence length of ten, this function expects an array of shape (2,10), looking perhaps like this:
# 
# array([['how', 'is', 'your', 'day', '0.0', '0.0', '0.0', '0.0', '0.0',
#         '0.0'],
#     ['i', "don't", 'know', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
#         '0.0']], dtype='<U32')
#     
# you will receive an array of shape (num_utterances,128,512).  128 is the sequence length bert returns, and 512 is the dimensionality of each 
# word vector, zero-filling the elements that do not exist (e.g. text[0,3] would be a normal 512 vector, but text[0,4] would be all zeros because the utterance ends after "day"
# 

def get_bert_embeddings(text_tensor):
    # convert between datatypes
    text = np.apply_along_axis(lambda row: b' '.join(row) if 'S' in str(row.dtype) else b' '.join(lmap(lambda elt: elt.encode('ascii'), row)), -1, np.squeeze(text_tensor))
    v = np.vectorize(lambda elt: elt[:(elt.find(b'0.0')-1)])
    text = v(text)

    print('Loading bert model and converting words to embeddings...')
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
    bert_model = hub.KerasLayer(tfhub_handle_encoder)
    
    encodings = []
    for sentence in tqdm(text):
        preprocessed = bert_preprocess_model([sentence])
        num_words = np.squeeze(np.sum(preprocessed['input_mask'], axis=-1))

        encoded = np.squeeze(bert_model(preprocessed)['sequence_output'].numpy())
        encoded[num_words:] = 0

        encodings.append(encoded)
    text = np.squeeze(arlist(encodings))
    return text

######### End BERT

######### Data Preprocessing and associated helper functions  #########
# *seq() are helper functions for dealing with computational sequence objects (required for CMU-Multimodal-SDK interface)
# label_map_fn is where I map labels from their original form to their eventual form.  For IEMOCAP, labels start out on a 0-5 scale.  <3 = negative (0), ==3 = neutral (1), >3 = positive (2)
# load_data is where a lot of the data heavy lifting happens.  
#   In training, the dataset is aligned and preprocessed into its correct form (depending on cross-utterance vs within-utterance).  BERT embeddings are extracted, the dataset is reshaped, and passed to the model stage
#   In inference, the same process happens, but first wavs are split by utterance boundaries using the VAD from deepspeech, transcribed using MS azure (utterances with no recognized speech are removed and the others renamed to preserve correct numbering), 
#     and passed to MS Azure's speaker verification system with the profile_id of the participant to see how likely it was that our participant spoke this utterance (as opposed to someone else)

def get_compseq(path, key_name):
    if 'pk' in path:
        a = load_pk(path)
        compseq = mmdatasdk.computational_sequence(key_name)
        compseq.setData(a, key_name)
        metadata_template['root name'] = key_name
        compseq.setMetadata(metadata_template, key_name)
    else:
        assert 'csd' in path
        a = mmdatasdk.mmdataset({key_name: path})
        compseq = a[key_name]
    return compseq

def get_compseq_obj(obj, key_name):
    if type(obj) is dict:
        compseq = mmdatasdk.computational_sequence(key_name)
        compseq.setData(obj, key_name)
        metadata_template['root name'] = key_name
        compseq.setMetadata(metadata_template, key_name)
    else:
        compseq = obj[key_name]
    return compseq

def add_seq(dataset, obj, key_name, obj_type='path'):
    if obj_type == 'path':
        compseq = get_compseq(obj, key_name)
    else:
        compseq = get_compseq_obj(obj, key_name)
    dataset.computational_sequences[key_name] = compseq

def load_data():
    if exists(args['tensors_path']) and not args['overwrite_tensors'] and not args['mode']=='inference':
        print('Loading data...')
        train, val, test = load_pk(args['tensors_path'])
        return train, val, test

    dataset = mmdatasdk.mmdataset(recipe={'dummy': args['dummy_path']})
    del dataset.computational_sequences['dummy']
    
    labels = load_pk(args['labels_path']) if 'pk' in args['labels_path'] else load_json(args['labels_path'])
    fake_labels = ( (args['mode'] == 'inference') and (not args['evaluate_inference']) ) # used in inference when evaluation labels are not provided (actual prediction)
    transcripts = None
    if args['mode'] == 'inference':
        # speaker verification
        args['speaker_ver'] = {} # vid: [ [{profile_id: 'profile id', score: 'score' for all profile_ids in rank order] for _ in num_utts]
        
        timing_path = '/'.join(args['transcripts_path'].split('/')[:-1])+'/timing.pk'

        if args['wav_dir'][-1]=='/':
            args['wav_dir'] = args['wav_dir'][:-1]
        temp_wav_dir = args['wav_dir']+'_segments'
        recom_dir = args['wav_dir']+'_recombined'
        
        rmtree(temp_wav_dir)
        convert.split_wavs(args['wav_dir'], temp_wav_dir_in=temp_wav_dir, agg_in=args['VAD_agg'])
        assert args['mode'] == 'inference', 'Transcribing on preformatted datasets is not supported yet'

        wav_paths = glob(join(temp_wav_dir, '*'))
        print(f'Transcribing {temp_wav_dir}')
        if 'text' in args['modality']:
        if False:
            transcripts = { wav_path.split('/')[-1].replace('.wav', ''): dict(lzip(['features', 'intervals', 'confidence'], get_transcript(wav_path))) for wav_path in tqdm(wav_paths) }
            save_pk(args['transcripts_path'], transcripts)
        else:
            transcripts = load_pk(args['transcripts_path']) # for activation


        # remove wav_segments that have no recognized speech
        no_recognized = [k for k,v in transcripts.items() if len(v['features'])==0]
        for elt in no_recognized:
            rmfile(join(temp_wav_dir, f'{elt}.wav'))

        # rename rest
        unique_videos = set(lmap(lambda elt: elt.split('/')[-1].replace('.wav', '').split('[')[0], glob(join(temp_wav_dir, '*.wav'))))
        for vid in unique_videos:
            recognized = [k for k,v in transcripts.items() if len(v['features'])>0 and k.split('[')[0] == vid]
            recognized = sorted(recognized, key=lambda elt: int(elt.replace(']','').split('[')[1]))

            unrecognized = [k for k,v in transcripts.items() if len(v['features'])==0 and k.split('[')[0] == vid]
            
            idxs = np.arange(len(recognized))
            
            orig_new = [(k, f'{k.split("[")[0]}[{idx}]') for k,idx in zip(recognized, idxs)]
            unrecognized = [elt for elt in unrecognized if elt not in lzip(*orig_new)[1]] # filter only down to overflow - intermediate stuff will be overwritten

            # move wavs, transcripts keys
            for orig, new in orig_new:
                shutil.move(join(temp_wav_dir, f'{orig}.wav'), join(temp_wav_dir, f'{new}.wav'))
                transcripts[new] = transcripts[orig]
                if new != orig:
                    del transcripts[orig]
            for unr in unrecognized:
                del transcripts[unr]

        # grab labels to realign dummy intervals
        if fake_labels:
            labels = {}

        # recombine transcripts to be a single "video" key for cross utterance
        b = transcripts
        unique_vid_keys = np.unique(lmap(lambda elt: elt.split('[')[0], lkeys(b)))
        combined = {}

        for vid_key in unique_vid_keys:
            sorted_rel_keys = sorted(lfilter(lambda elt: vid_key in elt.split('[')[0], lkeys(b)), key=lambda elt: int(elt.replace(']','').split('[')[1]))

            new_intervals = []
            maxes = [] # for recreating intervals
            for i in range(len(sorted_rel_keys)):
                new_max = np.max(np.concatenate(new_intervals)) if i > 0 else 0
                maxes.append(new_max)

                intervals_to_add = b[sorted_rel_keys[i]]['intervals']

                if fake_labels:
                    if vid_key not in labels:
                        labels[vid_key] = {
                            'features': ar([ [1] for _ in range(len(sorted_rel_keys))]),
                            'intervals': []
                        }

                # get the length of the file so the added intervals at the end to not lose time
                x,sr = librosa.load(join(temp_wav_dir, f'{vid_key}[{i}].wav'))
                intervals_to_add[-1,-1] = x.shape[0]/sr
                intervals_to_add = intervals_to_add + new_max
                new_intervals.append(intervals_to_add)

            new_intervals = np.concatenate(new_intervals)

            assert np.all(np.argsort(new_intervals[:,1]) == np.arange(new_intervals.shape[0])) and np.all(np.argsort(new_intervals[:,0]) == np.arange(new_intervals.shape[0])), f'New intervals must be sorted properly after recombining: {new_intervals}'
            combined[vid_key] = {
                'features': np.concatenate([ b[rel_key]['features'] for rel_key in sorted_rel_keys ]),
                'intervals': new_intervals,
            }

            maxes += [new_intervals.max()]
            labels[vid_key]['intervals'] = ar(lzip(maxes[:-1], maxes[1:]))
            labels[vid_key]['features'] = ar(labels[vid_key]['features']).reshape((-1,1))

        transcripts = combined

        # recombine wavs
        rmtree(recom_dir)
        mkdirp(recom_dir)
        for vid_key in unique_vid_keys:
            wav_paths = sorted(glob(join(temp_wav_dir,f'{vid_key}[*')), key=lambda elt: int( elt.split('/')[-1].replace(']','').replace('.wav','').split('[')[1] ) )
            sr = librosa.load(wav_paths[0])[1]
            x = np.concatenate([librosa.load(elt)[0] for elt in wav_paths])
            sf.write(join(recom_dir,f'{vid_key}.wav'),x,sr)
        
        api_key = load_json(join(BASE_PATH, 'azure_secrets.json'))['speaker_verification_key']
        assert type(args['speaker_profile']) == str
        speaker_profiles = [args['speaker_profile']]

        if 'text' in args['modality']:
            for vid_key in unique_vid_keys:
                wav_paths = sorted(glob(join(temp_wav_dir,f'{vid_key}[*')), key=lambda elt: int( elt.split('/')[-1].replace(']','').replace('.wav','').split('[')[1] ) )
                args['speaker_ver'][vid_key] = []
                for wav_path in wav_paths:
                    try:
                        results = identify_user(api_key, wav_path, speaker_profiles)['profilesRanking']
                        args['speaker_ver'][vid_key].append(results[0]['score'])
                    except:
                        args['speaker_ver'][vid_key].append(-1)

        save_pk(args['labels_path'], labels)
        save_pk(timing_path, labels)

        if 'audio' in args['modality']:
            args['wav_dir'] = recom_dir
    
    if 'text' in args['modality']:
        if transcripts is None:
            transcripts = load_pk(args['transcripts_path'])
        add_seq(dataset, transcripts, 'text', obj_type='obj')
    
    if 'audio' in args['modality']:
        deploy_unaligned_mfb_csd(check_labels=(args['mode']!='inference'))
        add_seq(dataset, args['audio_path'], 'audio')

    if 'audio' in args['modality'] and 'text' in args['modality']:
        dataset.align('text', collapse_functions=[avg])
        dataset.impute('text')
            
    if args['mode'] == 'inference':
        timing = load_pk(timing_path)
        for k in labels.keys():
            labels[k]['intervals']=timing[k]['intervals']

    add_seq(dataset, labels, 'labels', obj_type='obj')

    dataset.align('labels')
    dataset.hard_unify()

    data = {}
    for key in np.sort(arlist(dataset['labels'].keys())):
        data[key] = {
            'features': np.array([[key]]),
            'intervals': ar(dataset['labels'][key]['intervals'])
        }

    compseq = mmdatasdk.computational_sequence('ids')
    compseq.setData(data, 'ids')
    metadata_template['root name'] = 'ids'
    compseq.setMetadata(metadata_template, 'ids')
    dataset.computational_sequences['ids'] = compseq

    tensors = dataset.get_tensors(
        seq_len=args['seq_len'],
        non_sequences=['labels', 'ids'],
        direction=True,
        folds=None
    )[0]

    labels = (tensors['labels'] if not fake_labels else np.ones(tensors[args['modality'].split(',')[0]].shape[0:1])).reshape(-1)
    ids = np.squeeze(tensors['ids'])
    text = tensors['text']

    if args['mode'] == 'inference':
        args['test_keys'] = np.squeeze(tensors['ids'])
        args['train_keys'] = ar([])

    if args['train_keys'] is None:
        args['train_keys'], args['test_keys'] = train_test_split(np.squeeze(tensors['ids']), test_size=.2, random_state=11)
        
    train_idxs = np.where(arlmap(lambda elt: elt in args['train_keys'], np.squeeze(tensors['ids'])))[0]
    if args['mode'] == 'inference':
        train_idxs, val_idxs = ar([]).astype('int32'), ar([]).astype('int32')
    else:
        train_idxs, val_idxs = train_test_split(train_idxs, test_size=.2, random_state=11)

    test_idxs = np.where(arlmap(lambda elt: elt in args['test_keys'], np.squeeze(tensors['ids'])))[0]
    assert len(train_idxs) + len(val_idxs) + len(test_idxs) == len(tensors['ids']), 'If this assertion fails, it means not all utterance keys were accounted for in the keys provided'

    if 'text' in args['modality']:
        text = get_bert_embeddings(text)

    if 'audio' in args['modality']:
        audio = tensors['audio']

    if 'text' in args['modality'] and 'audio' not in args['modality']:
        train = text[train_idxs], labels[train_idxs], ids[train_idxs]
        val = text[val_idxs], labels[val_idxs], ids[val_idxs]
        test = text[test_idxs], labels[test_idxs], ids[test_idxs]

    elif 'audio' in args['modality'] and 'text' not in args['modality']:
        train = audio[train_idxs], labels[train_idxs], ids[train_idxs]
        val = audio[val_idxs], labels[val_idxs], ids[val_idxs]
        test = audio[test_idxs], labels[test_idxs], ids[test_idxs]
    
    else: # both
        train = text[train_idxs], audio[train_idxs], labels[train_idxs], ids[train_idxs]
        val = text[val_idxs], audio[val_idxs], labels[val_idxs], ids[val_idxs]
        test = text[test_idxs], audio[test_idxs], labels[test_idxs], ids[test_idxs]
    
    if args['mode'] != 'inference':
        args['train_sample_weight'] = get_sample_weight(labels[train_idxs])
        args['val_sample_weight'] = get_sample_weight(labels[val_idxs])
        args['test_sample_weight'] = get_sample_weight(labels[test_idxs])

    if args['mode'] != 'inference':
        print(f'Saving tensors to {args["tensors_path"]}.pk')
        save_pk(args['tensors_path'], (train, val, test))
    else:
        save_pk('./.temp_tensors.pk', tensors)
    return train, val, test
######### End data processing


def main(args_in):
    global args
    args = args_in

    assert args['cross_utterance']==0, 'Cross utterance training / inference is no longer supported.  Please see cross_utterance_appendix.py if you\'d like to implement this.'

    train, val, test = load_data()    
    if 'text' in args['modality'] and 'audio' in args['modality']:
        return train_within_multi(train,val,test)
    elif 'text' in args['modality']:
        return train_within_uni_text(train,val,test)
    elif 'audio' in args['modality']:
        return train_within_uni_audio(train,val,test)

def main_inference(args_in):
    global args
    args = args_in

    assert args['cross_utterance']==0, 'Cross utterance training / inference is no longer supported.  Please see cross_utterance_appendix.py if you\'d like to implement this.'

    _,_, test = load_data()

    print('Predicting...')
    model = tf.keras.models.load_model(args['model_path'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args['text_lr']),
        loss='sparse_categorical_crossentropy',
        weighted_metrics=['sparse_categorical_accuracy'],
    )

    # max_utts = model._build_input_shape[1] #56 for iemocap
    if len(args['modality'].split(','))>1: # multimodal
        test_text, test_audio, test_labels, ids = test
        if args['evaluate_inference']:
            model.evaluate({'text': test_text, 'audio': test_audio}, test_labels, batch_size=args['bs'])
        preds = model.predict({'text': test_text, 'audio': test_audio}, batch_size=args['bs'])

    else:
        test_data, test_labels, ids = test
        if args['evaluate_inference']:
            model.evaluate(test_data, test_labels, batch_size=args['bs'])
        preds = model.predict(test_data)

    if args['print_transcripts']:
        a = {k: ' '.join(v['features'].reshape(-1)) for k,v in load_pk(args['transcripts_path']).items()}
        reverse_label_map = [0,1,2]
            
        d = {}
        corrects = test_labels==np.argmax(preds, axis=-1)
        for id,corr in zip(ids,corrects):
            k = id.split('[')[0]
            if k not in d:
                d[k] = [corr]
            else:
                d[k].append(corr)

        for k,v in d.items():
            d[k] = np.sum(v)/len(v)

        df = pd.DataFrame({'id': ids, 'pred': np.argmax(preds, axis=-1), 'label': test_labels, 'correct': (np.argmax(preds, axis=-1)==test_labels).astype('int32')})
        print(df)
        save_pk('preds/df.pk', df)

    print('Your output will be in output/inference.pk')
    full_res = {
        'data': test_data if len(args['modality'].split(','))==1 else (test_audio, test_text),
        'predictions': preds,
        'ids': ids,
        'speaker_ver': args['speaker_ver']
    }
    return full_res


tensors_base_path = join(BASE_PATH, 'tensors')
data_path = join(BASE_PATH, 'data')
ie_path = join(data_path, 'iemocap')
models_path = join(BASE_PATH, 'models')
uni_path = join(data_path, 'uni.pk')

params = [
    # core paths
    ('--transcripts_path',str,join(ie_path, 'IEMOCAP_TimestampedWords.pk'), 'Path to transcripts.'),
    ('--audio_path',str,join(ie_path, 'mfb.pk'), 'Where mfbs will be (or are already) saved.'),
    ('--labels_path',str, join(ie_path, 'IEMOCAP_EmotionLabels.pk'), 'Path to labels.'),
    ('--dummy_path',str, join(ie_path, 'IEMOCAP_EmotionLabels.pk'), 'Used to initialize dataset.  Can be any valid labels file.  You probably will not change this.'),
    ('--tensors_path',str, join(tensors_base_path, 'tensors.pk'), 'Where tensors will be stored for this dataset.  If "unique", will store a hash in tensors folder.'),
    ('--wav_dir',str, join(ie_path, 'wavs'), 'Path to wav dir.'),
    ('--model_path',str,join(models_path, 'model'), 'Where model will be saved.'),
    ('--hffn_path',str,join(models_path, 'hffn'), 'Where uni text, uni audio, and hffn models will be saved.'),
    ('--uni_path',str,uni_path, 'Where unimodal activations are stored before feeding into hffn (in the case of cross utterance multimodal)'),

    # core options
    ('--mode',str,'train', 'train or inference'),
    ('--speaker_profile',str,'', 'profile id of speaker this recording comes from (used for speaker verification in inference mode)'),
    ('--cross_utterance', int, 0, 'If 0, build a within-utterance model.  If 1, build a cross utterance model.'),
    ('--modality', str,'text', 'modalities separated by ,.  options are ["text,audio", "audio,text", "audio", "text"]'),
    ('--evaluate_inference', int, 0, 'If mode==inference and this is 1, evaluate using labels passed in instead of returning prediction.  This can be a good sanity check on the training process if you have labels, and want to see how well your saved model is inferring on some dataset.'),
    ('--overwrite_tensors', int, 0, 'Do you want to overwrite tensors in tensors_path or not?  By default, we will use tensors we find in tensors_path instead of regenerating.'),
    ('--overwrite_mfbs', int, 0, 'Do you want to overwrite the mfbs in audio_path by recreating the mfbs from wav_dir?'),
    ('--keys_path',str, '', '(Optional) path to json file with keys "train_keys" and "test_keys" which each contain nonoverlapping video (if cross-utterance) / utterance (if within) key lists'),
    ('--print_transcripts',int, 0, 'Print transcripts and labels during inference. NOTE: change keys if not IEMOCAP.'),
    ('--VAD_agg',int, 0, 'Aggressiveness of VAD during inference'),
    ('--num_labels',int, 3, 'Dimensionality of labels'),

    # hyperparameters
    ('--epochs', int, 500, ''),
    ('--trials', int, 1, 'Number of trials you want to run.  All results will be saved and appended, and the model from the last trial will be saved. This is useful if there is variability in how well your model performs, so you can build performance statistics across trials.'),
    ('--seq_len', int, 50, 'The max sequence length.  If involving text, this will mean max number of words per utterance.  If involving only audio, this will be the max number of mfbs.'),
    ('--bs', int, 10, 'Batch size'),

    ('--lstm_units_text', int, 32, 'Number of units used in text lstm'),
    ('--drop_text', float, .2, 'Dropout rate for text'),
    ('--drop_text_lstm', float, .3, 'Dropout rate for text lstm'),
    ('--text_lr', float, 1e-3, 'Learning rate for text'),
    ('--filters_text', int,50, 'Number of filters used for cross utterance text conv layers'),

    ('--drop_audio', float, .2, 'Dropout rate for audio'),
    ('--drop_audio_lstm', float, .3, 'Dropout rate for audio lstm'),
    ('--audio_lr', float, 1e-3, 'Audio learning rate'),
    ('--filters_audio', int, 50, 'Number of filters for audio conv layers'),
    
    ('--drop_within_multi', float, .2, 'Dropout rate for within multimodal'),
    ('--multi_lr', float, 1e-3, 'Learning rate for within multimodal')
]

if __name__ == '__main__':
    parser = standard_grid.ArgParser()

    mkdirp(tensors_base_path)
    mkdirp(ie_path)
    mkdirp(models_path)

    for param in params:
        parser.register_parameter(*param)

    args = vars(parser.compile_argparse())

    builtins.args = args # make args a cross-module global variable.  Not particularly good practice, but cuts down development time dramatically when adding hyperparams for a search

    assert args['mode'] in ['train', 'inference']
    keys = load_json(args['keys_path'])
    if keys is not None:
        args['train_keys'], args['test_keys'] = LD(keys)[['train_keys', 'test_keys']]
        assert np.all(arlmap(lambda elt: elt not in args['train_keys'], args['test_keys'])), 'There cannot be any overlapping elements between train and test keys'
        assert np.all(arlmap(lambda elt: elt not in args['test_keys'], args['train_keys'])), 'There cannot be any overlapping elements between train and test keys'
    else:
        args['train_keys'] = None

    assert 'text' in args['modality'] or 'audio' in args['modality'], f'modality flag must contain text, audio, or both, but is instead: {args["modality"]}'

    if args['tensors_path'] == 'unique':
        hash_object = hashlib.sha512(str(int(np.random.random()*1000)).encode("utf-8"))
        args['tensors_path'] = join(tensors_base_path, str(hash_object.hexdigest())[:10])

    out_dir = 'output/'
    mkdirp(out_dir)

    if args['mode'] == 'train':
        full_res = {} 
        for trial in range(args['trials']):
            res = main(args)
            for k,v in res.items():
                if k not in full_res:
                    full_res[k] = [v]
                else:
                    full_res[k].append(v)
            
        full_res = {k:ar(v) for k,v in full_res.items()}
        save_json(join(out_dir, 'results.txt'), full_res)

    else:
        full_res = main_inference(args)
        save_pk(join(out_dir, 'inference.pk'), full_res)
