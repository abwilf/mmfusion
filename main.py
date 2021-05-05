from utils import *
from consts import *
sys.path.append(MMSDK_PATH)
from mmsdk import mmdatasdk
sys.path.append(STANDARD_GRID_PATH)
import standard_grid

import copy
import hashlib
import soundfile as sf
import librosa

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import multiprocessing.dummy as mp_thread
import multiprocessing as mp_proc
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils import class_weight
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
# deploy_unaligned_mfb_csd creates all mfbs from a wav directory in args['wav_dir'], parallelized across threads

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


######### Data Preprocessing and associated helper functions for IEMOCAP  #########
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

### TODO: MODIFY ##
num_labels = 3 # iemocap
args = {}
args['num_labels'] = num_labels
def label_map_fn(labels):
    '''
    input: a one-dimensional array of labels (e.g., shape (10,...))
    output: a one-dimensional array of labels as integers

    e.g.:
        iemocap: input = ['ang', 'ang', 'neu', 'hap','sad'], output = [0,0,1,2,3]
    '''
    if args['mode'] =='inference' and args['evaluate_inference']:
        return labels

    # # iemocap
    # label_map = {'ang': 0, 'hap': 1, 'exc': 1, 'neu': 2, 'sad': 3}
    # if len(labels.shape) > 1:
    #     labels = np.squeeze(labels, axis=1)
    # return arlmap(lambda elt: label_map[elt.decode('utf8') if type(elt) in [bytes, np.bytes_] else elt], labels).astype('int32')

    ## iemocap val
    labels[labels<3]=0
    labels[labels==3]=1
    labels[labels>3]=2
    return labels.astype('int32').reshape((-1))

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
        
        timing_path = '/'.join(args['labels_path'].split('/')[:-1])+'/timing.pk'

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
        # if False:
            transcripts = { wav_path.split('/')[-1].replace('.wav', ''): dict(lzip(['features', 'intervals', 'confidence'], get_transcript(wav_path))) for wav_path in tqdm(wav_paths) }
        else:
            transcripts = load_pk(args['transcripts_path']) # for activation

        # save_pk(args['transcripts_path'], transcripts)

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

    if args['cross_utterance']:
        print('Reshaping data as cross utterance in the shape (num_vids, max_utts, 128, 512)...')
        audio, text, labels, utt_masks = [], [], [], []

        vid_keys = lvmap(lambda elt: elt.split('[')[0], tensors['ids'].reshape(-1))
        max_utts = np.unique(vid_keys, return_counts=True)[1].max()
        unique_vid_keys = pd.Series(vid_keys).unique()

        if args['mode'] == 'inference':
            args['test_keys'] = np.squeeze(unique_vid_keys)
            args['train_keys'] = ar([])

        elif args['train_keys'] is None:
            args['train_keys'], args['test_keys'] = train_test_split(unique_vid_keys, test_size=.2, random_state=11)

        train_idxs = np.where(arlmap(lambda elt: elt in args['train_keys'], unique_vid_keys))[0]

        if args['mode'] == 'inference':
            train_idxs, val_idxs = ar([]).astype('int32'), ar([]).astype('int32')
        else:
            train_idxs, val_idxs = train_test_split(train_idxs, test_size=.2, random_state=11)

        test_idxs = np.where(arlmap(lambda elt: elt in args['test_keys'], unique_vid_keys))[0]
        assert len(train_idxs) + len(val_idxs) + len(test_idxs) == len(unique_vid_keys), 'If this assertion fails, it means not all video keys were accounted for in the keys provided'

        for vid_key in unique_vid_keys:
            vid_idxs = np.where(vid_keys==vid_key)[0]
            num_utts = len(vid_idxs)
            
            if 'text' in args['modality']:
                relevant_text = np.squeeze(tensors['text'][vid_idxs], axis=-1)
                utt_padded_text = np.pad(relevant_text, ((0,max_utts-num_utts), (0,0)), 'constant')
                text.append(utt_padded_text)
            
            if 'audio' in args['modality']:
                relevant_audio = tensors['audio'][vid_idxs]
                utt_padded_audio = np.pad(relevant_audio, ((0,max_utts-num_utts), (0,0), (0,0)), 'constant')
                audio.append(utt_padded_audio)

            if not fake_labels:
                relevant_labels = np.squeeze(tensors['labels'][vid_idxs]).reshape((num_utts,-1))
                relevant_labels = label_map_fn(relevant_labels).reshape(-1)
                utt_padded_labels = np.pad(relevant_labels, ((0,max_utts-num_utts)), 'constant')
                labels.append(utt_padded_labels)
            
            utt_mask = np.ones(max_utts)
            utt_mask[num_utts:] = 0
            utt_masks.append(utt_mask)
        
        # get text in sentence form: (num_vids, max_utts)
        text = np.apply_along_axis(lambda row: b' '.join(row) if 'S' in str(row.dtype) else b' '.join(lmap(lambda elt: elt.encode('ascii'), row)), -1, text)
        v = np.vectorize(lambda elt: elt[:(elt.find(b'0.0')-1)])
        text = v(text)
        
        del dataset

        if fake_labels:
            modality = text if text.shape != () else audio
            labels = np.ones(ar(modality).shape[0:2])

        if 'text' in args['modality']:
            print('Loading bert model and converting words to embeddings...')
            bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
            bert_model = hub.KerasLayer(tfhub_handle_encoder)

            new_data = []
            for utt_mask, utts in tqdm(lzip(utt_masks, list(text))):
                num_utts = int(np.sum(utt_mask))
                utts = utts[:num_utts]
                text_preprocessed = bert_preprocess_model(utts)
                encoded = bert_model(text_preprocessed)['sequence_output'].numpy()

                num_words = np.sum(text_preprocessed['input_mask'], axis=-1)
                for i,num_word in enumerate(num_words):
                    encoded[i, num_word:, :] = 0

                encoded = np.pad(encoded, ((0, max_utts-num_utts), (0,0), (0,0)), 'constant')
                new_data.append(encoded)
            text = ar(new_data)

        labels = ar(labels).astype('int32')
        utt_masks = ar(utt_masks)

        if args['mode'] == 'inference':
            train_utt_masks, train_labels = ar([]), ar([])
            val_utt_masks, val_labels = ar([]), ar([])
        else:
            train_labels = labels[train_idxs]
            train_utt_masks = utt_masks[train_idxs]
            train_class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels).astype('int32'), y=np.concatenate([arr[:amt] for arr,amt in zip(train_labels, np.sum(train_utt_masks, axis=-1).astype('int32'))]))
            train_class_sample_weights = lvmap(lambda elt: train_class_weights[elt], train_labels)
            train_utt_masks = train_class_sample_weights * train_utt_masks

            val_labels = labels[val_idxs]
            val_utt_masks = utt_masks[val_idxs]
            val_class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(val_labels).astype('int32'), y=np.concatenate([arr[:amt] for arr,amt in zip(val_labels, np.sum(val_utt_masks, axis=-1).astype('int32'))]))
            val_class_sample_weights = lvmap(lambda elt: val_class_weights[elt], val_labels)
            val_utt_masks = val_class_sample_weights * val_utt_masks

        test_labels = labels[test_idxs]
        test_utt_masks = utt_masks[test_idxs]
        if args['mode'] != 'inference':
            test_class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(test_labels).astype('int32'), y=np.concatenate([arr[:amt] for arr,amt in zip(test_labels, np.sum(test_utt_masks, axis=-1).astype('int32'))]))
            test_class_sample_weights = lvmap(lambda elt: test_class_weights[elt], test_labels)
            test_utt_masks = test_class_sample_weights * test_utt_masks
        
        if 'audio' in args['modality'] and 'text' in args['modality']:
            train = ar(text)[train_idxs], ar(audio)[train_idxs], train_labels, train_utt_masks, unique_vid_keys[train_idxs]
            val = ar(text)[val_idxs], ar(audio)[val_idxs], val_labels, val_utt_masks, unique_vid_keys[val_idxs]
            test = ar(text)[test_idxs], ar(audio)[test_idxs], test_labels, test_utt_masks, unique_vid_keys[test_idxs]

        elif 'audio' in args['modality']:
            train = ar(audio)[train_idxs], train_labels, train_utt_masks, unique_vid_keys[train_idxs]
            val = ar(audio)[val_idxs], val_labels, val_utt_masks, unique_vid_keys[val_idxs]
            test = ar(audio)[test_idxs], test_labels, test_utt_masks, unique_vid_keys[test_idxs]

        elif 'text' in args['modality']:
            train = ar(text)[train_idxs], train_labels, train_utt_masks, unique_vid_keys[train_idxs]
            val = ar(text)[val_idxs], val_labels, val_utt_masks, unique_vid_keys[val_idxs]
            test = ar(text)[test_idxs], test_labels, test_utt_masks, unique_vid_keys[test_idxs]

    else: # within utterance
        labels = label_map_fn(np.squeeze(tensors['labels'])) if not fake_labels else np.ones(tensors[args['modality'].split(',')[0]].shape[0:1])
        ids = np.squeeze(tensors['ids'])

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
            text = np.apply_along_axis(lambda row: b' '.join(row) if 'S' in str(row.dtype) else b' '.join(lmap(lambda elt: elt.encode('ascii'), row)), -1, np.squeeze(tensors['text']))
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

                # encoded = bert_model(preprocessed)['pooled_output'].numpy()
                encodings.append(encoded)
            text = np.squeeze(arlist(encodings))

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

######### Models #########
# We built support for different kinds of models - ones that considered within utterance or cross utterance context unimodally (audio xor text) and multimodally (audio + text)

def train_cross_multi(train, val, test):
    train_text, train_audio, train_labels, train_utt_masks, train_ids = train
    val_text, val_audio, val_labels, val_utt_masks, val_ids = val
    test_text, test_audio, test_labels, test_utt_masks, test_ids = test

    train_cross_uni_audio(
        train=(train_audio, train_labels, train_utt_masks, train_ids),
        val=(val_audio, val_labels, val_utt_masks, val_ids),
        test=(test_audio, test_labels, test_utt_masks, train_ids)
    )
    train_cross_uni_text(
        train=(train_text, train_labels, train_utt_masks, train_ids), 
        val=(val_text, val_labels, val_utt_masks, val_ids), 
        test=(test_text, test_labels, test_utt_masks, train_ids)
    )

    import hffn
    u = load_pk(args['uni_path'])
    return hffn.multimodal(u, args)


def train_within_multi(train, val, test):
    train_text, train_audio, train_labels, train_ids = train
    val_text, val_audio, val_labels, val_ids = val
    test_text, test_audio, test_labels, test_ids = test

    dropout=args['drop_within_multi']
    TD = TimeDistributed

    text_input = Input(shape=train_text.shape[1:], name='text')
    text_mask = Masking(mask_value =0)(text_input)
    text_lstm = Bidirectional(LSTM(32, activation='tanh', return_sequences=False, dropout=0.3))(text_mask)
    text_drop = Dropout(dropout)(text_lstm)
    text_inter = Dense(100, activation='tanh')(text_drop)
    text_drop2 = Dropout(dropout)(text_inter)

    audio_input = Input(shape=train_audio.shape[1:], name='audio')
    audio_conv = Conv1D(filters=50, kernel_size=3, padding='same', data_format='channels_last', dtype='float32')(audio_input)
    audio_drop = Dropout(dropout)(audio_conv)
    audio_conv2 = Conv1D(filters=50, kernel_size=4, padding='same', data_format='channels_last', dtype='float32')(audio_drop)
    audio_drop2 = Dropout(dropout)(audio_conv2)
    audio_mp = MaxPool1D(pool_size=4, data_format='channels_last')(audio_drop2)
    audio_conv3 = Conv1D(filters=50, kernel_size=2, padding='same', data_format='channels_last', dtype='float32')(audio_mp)
    audio_drop3 = Dropout(dropout)(audio_conv3)
    audio_gmp = GlobalMaxPooling1D()(audio_drop3)

    concat = tf.keras.layers.concatenate([audio_gmp, text_drop2], axis=-1)

    dense1 = Dense(100, activation='relu')(concat)
    drop2 = Dropout(dropout)(dense1)
    clf = Dense(num_labels, activation='softmax')(drop2)

    model = Model({'text': text_input, 'audio': audio_input}, clf)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=args['multi_lr']),
        loss='sparse_categorical_crossentropy',
        weighted_metrics=['sparse_categorical_accuracy'],
    )
    model.summary()
    train_history = model.fit(
        x={'text': train_text, 'audio': train_audio},
        y=train_labels,
        batch_size=10,
        sample_weight=args['train_sample_weight'],
        epochs=500,
        validation_data=({'text': val_text, 'audio': val_audio}, val_labels, args['val_sample_weight']),
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)],
        verbose=1,
    ).history
    eval_history = model.evaluate(
        x={'text': test_text, 'audio': test_audio},
        y=test_labels,
        sample_weight=args['test_sample_weight'],
        batch_size=10,
    )
    mkdirp(args['model_path'])
    print('Saving model...')
    model.save(args['model_path'], include_optimizer=False)

    res = {
        'train_losses': ar(train_history['loss']),
        'train_accs': ar(train_history['sparse_categorical_accuracy']),
        'val_losses': ar(train_history['val_loss']),
        'val_accs': ar(train_history['val_sparse_categorical_accuracy']),
        'test_acc': eval_history[1],
        'test_loss': eval_history[0],
    }
    return res


def train_cross_uni_audio(train, val, test):
    train_data, train_labels, train_utt_masks, train_ids = train
    val_data, val_labels, val_utt_masks, val_ids = val
    test_data, test_labels, test_utt_masks, test_ids = test

    input = Input(shape=(train_data.shape[1],train_data.shape[2],train_data.shape[3]))
    conv = TimeDistributed(Conv1D(filters=args['filters_audio'], dilation_rate=1, kernel_size=16, padding='same', data_format='channels_last', dtype='float32'))(input)
    drop = TimeDistributed(Dropout(args['drop_audio']))(conv)
    conv2 = TimeDistributed(Conv1D(filters=args['filters_audio'], dilation_rate=2, kernel_size=16, padding='same', data_format='channels_last', dtype='float32'))(drop)
    drop2 = TimeDistributed(Dropout(args['drop_audio']))(conv2)
    mp = TimeDistributed(MaxPool1D(pool_size=4, data_format='channels_last'))(drop2)
    conv3 = TimeDistributed(Conv1D(filters=args['filters_audio'], dilation_rate=2, kernel_size=8, padding='same', data_format='channels_last', dtype='float32'))(mp)
    drop3 = TimeDistributed(Dropout(args['drop_audio']))(conv3)
    gmp = TimeDistributed(GlobalMaxPooling1D())(drop3)

    gru = Bidirectional(GRU(32, activation='tanh', return_sequences=True, dropout=args['drop_audio_lstm']))(gmp)
    drop4 = TimeDistributed(Dropout(args['drop_audio']))(gru)
    dense1 = TimeDistributed(Dense(100, activation='relu'))(drop4)
    drop5 = TimeDistributed(Dropout(args['drop_audio']))(dense1)
    dense2 = TimeDistributed(Dense(num_labels, activation='softmax'))(drop5)

    model = Model(input, dense2)
    aux = Model(input, dense1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=args['audio_lr']),
        loss='sparse_categorical_crossentropy',
        weighted_metrics=['sparse_categorical_accuracy'],
        sample_weight_mode='temporal',
    )
    model.summary()
    train_history = model.fit(
        x=train_data,
        y=train_labels,
        sample_weight=train_utt_masks,
        batch_size=args['bs'],
        epochs=args['epochs'],
        validation_data=(val_data, val_labels, val_utt_masks),
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)],
    ).history
    eval_history = model.evaluate(
        x=test_data,
        y=test_labels,
        sample_weight=test_utt_masks,
        batch_size=args['bs'],
    )
    print('Saving model...')
    if len(args['modality'].split(','))>1: # multimodal
        hffn_save_path = join(args['hffn_path'], 'uni_audio')
        mkdirp(hffn_save_path)
        aux.save(hffn_save_path, include_optimizer=False)

    else:
        mkdirp(args['model_path'])
        model.save(args['model_path'], include_optimizer=False)
        
    uni = load_pk(args['uni_path'])
    uni = {} if uni is None else uni
    uni['audio_train'] = aux.predict(x=train_data, batch_size=10)
    uni['audio_train_mask'] = train_utt_masks
    uni['audio_train_label'] = train_labels

    uni['audio_val'] = aux.predict(x=val_data, batch_size=10)
    uni['audio_val_mask'] = val_utt_masks
    uni['audio_val_label'] = val_labels

    uni['audio_test'] = aux.predict(x=test_data, batch_size=10)
    uni['audio_test_mask'] = test_utt_masks
    uni['audio_test_label'] = test_labels
    save_pk(args['uni_path'], uni)

    res = {
        'train_losses': ar(train_history['loss']),
        'train_accs': ar(train_history['sparse_categorical_accuracy']),
        'val_losses': ar(train_history['val_loss']),
        'val_accs': ar(train_history['val_sparse_categorical_accuracy']),
        'test_acc': eval_history[1],
        'test_loss': eval_history[0],
    }
    return res

def train_within_uni_audio(train, val, test):
    train_data, train_labels, train_ids = train
    val_data, val_labels, val_ids = val
    test_data, test_labels, test_ids = test

    input = Input(shape=(train_data.shape[1],train_data.shape[2]))
    conv = Conv1D(filters=args['filters_audio'], dilation_rate=1, kernel_size=3, padding='same', data_format='channels_last', dtype='float32')(input)
    drop = Dropout(args['drop_audio'])(conv)
    conv2 = Conv1D(filters=args['filters_audio'], dilation_rate=2, kernel_size=4, padding='same', data_format='channels_last', dtype='float32')(drop)
    drop2 = Dropout(args['drop_audio'])(conv2)
    mp = MaxPool1D(pool_size=4, data_format='channels_last')(drop2)
    conv3 = Conv1D(filters=args['filters_audio'], dilation_rate=2, kernel_size=2, padding='same', data_format='channels_last', dtype='float32')(mp)
    drop3 = Dropout(args['drop_audio'])(conv3)
    gmp = GlobalMaxPooling1D()(drop3)
    drop4 = Dropout(args['drop_audio'])(gmp)
    dense1 = Dense(100, activation='relu')(drop4)
    drop5 = Dropout(args['drop_audio'])(dense1)
    dense2 = Dense(num_labels, activation='softmax')(drop5)

    model = Model(input, dense2)
    aux = Model(input, dense1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=args['audio_lr']),
        loss='sparse_categorical_crossentropy',
        weighted_metrics=['sparse_categorical_accuracy'],
    )
    model.summary()
    train_history = model.fit(
        x=train_data,
        y=train_labels,
        batch_size=args['bs'],
        sample_weight=args['train_sample_weight'],
        epochs=args['epochs'],
        validation_data=(val_data, val_labels),
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)],
    ).history
    eval_history = model.evaluate(
        x=test_data,
        y=test_labels,
        batch_size=args['bs'],
        sample_weight=args['test_sample_weight'],
    )
    mkdirp(args['model_path'])
    print('Saving model...')
    model.save(args['model_path'], include_optimizer=False)

    res = {
        'train_losses': ar(train_history['loss']),
        'train_accs': ar(train_history['sparse_categorical_accuracy']),
        'val_losses': ar(train_history['val_loss']),
        'val_accs': ar(train_history['val_sparse_categorical_accuracy']),
        'test_acc': eval_history[1],
        'test_loss': eval_history[0],
    }
    return res


def train_cross_uni_text(train, val, test):
    train_data, train_labels, train_utt_masks, train_ids = train
    val_data, val_labels, val_utt_masks, val_ids = val
    test_data, test_labels, test_utt_masks, test_ids = test

    def res_block(x, filters):
        x_skip = x

        x = TimeDistributed(Conv1D(filters=filters, kernel_size=4, dilation_rate=1, padding='same', data_format='channels_last', dtype='float32'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation(tf.keras.activations.relu))(x)

        x = TimeDistributed(Conv1D(filters=filters, kernel_size=8, dilation_rate=2, padding='same', data_format='channels_last', dtype='float32'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation(tf.keras.activations.relu))(x)

        x = TimeDistributed(Conv1D(filters=filters, kernel_size=8, dilation_rate=2, padding='same', data_format='channels_last', dtype='float32'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        
        x = Add()([x, x_skip])

        x = TimeDistributed(Activation(tf.keras.activations.relu))(x)
        return x

    res = { 'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs': [], 'test_loss': [], 'test_accs': [] }

    input = Input(shape=(train_data.shape[1],train_data.shape[2],train_data.shape[3]))
    conv = TimeDistributed(Conv1D(filters=args['filters_text'], kernel_size=4, dilation_rate=1, padding='same', data_format='channels_last', dtype='float32'))(input)
    drop = TimeDistributed(Dropout(args['drop_text']))(conv)
    mp = TimeDistributed(MaxPool1D(pool_size=4, data_format='channels_last'))(drop)

    res = res_block(mp, filters=args['filters_text'])
    gmp = TimeDistributed(GlobalMaxPooling1D())(res)

    lstm = Bidirectional(LSTM(args['lstm_units_text'], activation='tanh', return_sequences=True, dropout=args['drop_text_lstm']))(gmp)
    drop = Dropout(args['drop_text'])(lstm)
    inter = TimeDistributed(Dense(100, activation='tanh'))(drop)
    drop2 = Dropout(args['drop_text'])(inter)
    clf = TimeDistributed(Dense(num_labels, activation='softmax'))(drop2)

    model = Model(input, clf)
    aux = Model(input, inter)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=args['text_lr']),
        loss='sparse_categorical_crossentropy',
        weighted_metrics=['sparse_categorical_accuracy'],
        sample_weight_mode='temporal',
    )
    model.summary()
    train_history = model.fit(
        x=train_data,
        y=train_labels,
        sample_weight=train_utt_masks,
        batch_size=args['bs'],
        epochs=args['epochs'],
        validation_data=(val_data, val_labels, val_utt_masks),
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)],
    ).history
    eval_history = model.evaluate(
        x=test_data,
        y=test_labels,
        sample_weight=test_utt_masks,
        batch_size=args['bs'],
    )
    print('Saving model...')
    if len(args['modality'].split(','))>1: # multimodal
        hffn_save_path = join(args['hffn_path'], 'uni_text')
        mkdirp(hffn_save_path)
        aux.save(hffn_save_path, include_optimizer=False)

    else:
        mkdirp(args['model_path'])
        model.save(args['model_path'], include_optimizer=False)

    uni = load_pk(args['uni_path'])
    uni = {} if uni is None else uni
    uni['text_train'] = aux.predict(x=train_data, batch_size=10)
    uni['text_train_mask'] = train_utt_masks
    uni['text_train_label'] = train_labels

    uni['text_val'] = aux.predict(x=val_data, batch_size=10)
    uni['text_val_mask'] = val_utt_masks
    uni['text_val_label'] = val_labels

    uni['text_test'] = aux.predict(x=test_data, batch_size=10)
    uni['text_test_mask'] = test_utt_masks
    uni['text_test_label'] = test_labels
    save_pk(args['uni_path'], uni)

    res = {
        'train_losses': ar(train_history['loss']),
        'train_accs': ar(train_history['sparse_categorical_accuracy']),
        'val_losses': ar(train_history['val_loss']),
        'val_accs': ar(train_history['val_sparse_categorical_accuracy']),
        'test_acc': eval_history[1],
        'test_loss': eval_history[0],
    }
    return res


def get_sample_weight(labels,class_weights=None):
    labels = labels.astype('int32')
    if class_weights is None:
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels).astype('int32'), y=labels)
    sample_weight = lvmap(lambda elt: class_weights[elt], labels)
    return sample_weight

def train_within_uni_text(train, val, test):
    train_data, train_labels, train_ids = train
    val_data, val_labels, val_ids = val
    test_data, test_labels, test_ids = test

    res = { 'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs': [], 'test_loss': [], 'test_accs': [] }

    input = Input(shape=(train_data.shape[1], train_data.shape[2]))
    lstm = Bidirectional(LSTM(args['lstm_units_text'], activation='tanh', return_sequences=False, dropout=args['drop_text_lstm']))(input)
    drop = Dropout(args['drop_text'])(lstm)
    inter = Dense(100, activation='tanh')(drop)
    drop2 = Dropout(args['drop_text'])(inter)
    clf = Dense(num_labels, activation='softmax')(drop2)

    model = Model(input, clf)
    aux = Model(input, inter)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=args['text_lr']),
        loss='sparse_categorical_crossentropy',
        weighted_metrics=['sparse_categorical_accuracy'],
    )
    model.summary()
    train_history = model.fit(
        x=train_data,
        y=train_labels,
        sample_weight=args['train_sample_weight'],
        batch_size=args['bs'],
        epochs=args['epochs'],
        validation_data=(val_data, val_labels, args['val_sample_weight']),
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)],
    ).history

    eval_history = model.evaluate(
        x=test_data,
        y=test_labels,
        sample_weight=args['test_sample_weight'],
        batch_size=args['bs'],
    )
    mkdirp(args['model_path'])
    print('Saving model...')
    model.save(args['model_path'], include_optimizer=False)
    res = {
        'train_losses': ar(train_history['loss']),
        'train_accs': ar(train_history['sparse_categorical_accuracy']),
        'val_losses': ar(train_history['val_loss']),
        'val_accs': ar(train_history['val_sparse_categorical_accuracy']),
        'test_acc': eval_history[1],
        'test_loss': eval_history[0],
    }
    return res

######### End models #########

def main(args_in):
    global args
    args = args_in

    train, val, test = load_data()
    if args['cross_utterance']:
        if 'text' in args['modality'] and 'audio' in args['modality']:
            return train_cross_multi(train,val,test)

        elif 'text' in args['modality']:
            return train_cross_uni_text(train,val,test)

        elif 'audio' in args['modality']:
            return train_cross_uni_audio(train,val,test)
    
    else: # within
        if 'text' in args['modality'] and 'audio' in args['modality']:
            return train_within_multi(train,val,test)
        elif 'text' in args['modality']:
            return train_within_uni_text(train,val,test)
        elif 'audio' in args['modality']:
            return train_within_uni_audio(train,val,test)

def main_inference(args_in):
    global args
    args = args_in

    _,_, test = load_data()

    print('Predicting...')
    if not args['cross_utterance']: # within
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

    else:
        if len(args['modality'].split(','))>1: # multimodal
            test_text, test_audio, test_labels, test_utt_masks, ids = test

            print('Loading models for HFFN inference...')
            uni_text_model = tf.keras.models.load_model(join(args['hffn_path'], 'uni_text'))
            uni_audio_model = tf.keras.models.load_model(join(args['hffn_path'], 'uni_audio'))
            
            uni = {
                'text': uni_text_model.predict(x=test_text, batch_size=10),
                'audio': uni_audio_model.predict(x=test_audio, batch_size=10),
                'mask': test_utt_masks,
                'label': test_labels
            }

            import hffn
            preds = hffn.inference(uni, args)
            
        else:
            test_data, test_labels, test_utt_masks, ids = test
            model = tf.keras.models.load_model(args['model_path'])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args['text_lr']),
                loss='sparse_categorical_crossentropy',
                weighted_metrics=['sparse_categorical_accuracy'],
                sample_weight_mode='temporal',
            )
            max_utts = model._build_input_shape[1] #56 for iemocap
            
            if args['evaluate_inference']:
                model.evaluate(test_data, test_labels, batch_size=args['bs'], sample_weight=test_utt_masks)
            preds = model.predict(test_data)
    
    if args['print_transcripts']:
        a = {k: ' '.join(v['features'].reshape(-1)) for k,v in load_pk(args['transcripts_path']).items()}
        reverse_label_map = [0,1,2]
        if args['cross_utterance']:
            utt_masks = np.sum(test_utt_masks, axis=-1).astype('int32')
            vid_ids = np.concatenate([ [vid_id]*num_utts for vid_id,num_utts in zip(ids, utt_masks)])
            vid_ids

            utt_ids = np.concatenate([ np.arange(num_utts) for num_utts in utt_masks ])
            utt_ids

            y_pred = np.argmax(preds, axis=-1)
            y_preds = np.concatenate([ y_pred[i,:num_utts] for i,num_utts in zip(np.arange(utt_masks.shape[0]), utt_masks)])

            y_true = np.concatenate([ test_labels[i,:num_utts] for i,num_utts in zip(np.arange(utt_masks.shape[0]), utt_masks)])
            y_true.shape

            df = pd.DataFrame({'vid_ids': vid_ids, 'utt_ids': utt_ids, 'pred': y_preds, 'label': y_true, 'correct': y_preds==y_true})
            tensors = load_pk('./.temp_tensors.pk')
            df['text'] = df.apply(lambda elt: ' '.join(tensors['text'][np.where(tensors['ids'].reshape(-1)==f"{elt['vid_ids']}[{elt['utt_ids']}]")[0][0]].reshape(-1)).replace(' 0.0', ''), axis=1)
            df = df[['vid_ids', 'utt_ids', 'text', 'correct', 'pred', 'label']]
            
        else:
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
            save_pk('df.pk', df)

    print('Your output will be in output/inference.pk')
    full_res = {
        'data': test_data if len(args['modality'].split(','))==1 else (test_audio, test_text),
        'utt_masks': None if not args['cross_utterance'] else test_utt_masks,
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
    args['num_labels'] = num_labels

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
    # init_except_hook()
    # init_exit_hook()

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
