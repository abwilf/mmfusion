from utils import *
from consts import *
sys.path.append(MMSDK_PATH)
from mmsdk import mmdatasdk
sys.path.append(STANDARD_GRID_PATH)
import standard_grid

import copy
import hashlib

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from mfb_util import *
import multiprocessing.dummy as mp_thread
import multiprocessing as mp_proc
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils import class_weight
import requests, atexit
import random
from notebook_util import setup_no_gpu, setup_one_gpu, setup_gpu
# setup_one_gpu()
# setup_no_gpu()

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
# from official.nlp import optimization  # to create AdamW optmizer - only needed if fine tuning

sys.path.append(DEEPSPEECH_PATH)
import convert

from transcribe import *

metadata_template = { "root name": '', "computational sequence description": '', "computational sequence version": '', "alignment compatible": '', "dataset name": '', "dataset version": '', "creator": '', "contact": '', "featureset bib citation": '', "dataset bib citation": ''}

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 
map_name_to_handle = { 'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', 'bert_en_cased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3', 'bert_multi_cased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3', 'small_bert/bert_en_uncased_L-2_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1', 'small_bert/bert_en_uncased_L-2_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1', 'small_bert/bert_en_uncased_L-2_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1', 'small_bert/bert_en_uncased_L-2_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1', 'small_bert/bert_en_uncased_L-4_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1', 'small_bert/bert_en_uncased_L-4_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1', 'small_bert/bert_en_uncased_L-4_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1', 'small_bert/bert_en_uncased_L-4_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1', 'small_bert/bert_en_uncased_L-6_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1', 'small_bert/bert_en_uncased_L-6_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1', 'small_bert/bert_en_uncased_L-6_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1', 'small_bert/bert_en_uncased_L-6_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1', 'small_bert/bert_en_uncased_L-8_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1', 'small_bert/bert_en_uncased_L-8_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1', 'small_bert/bert_en_uncased_L-8_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1', 'small_bert/bert_en_uncased_L-8_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1', 'small_bert/bert_en_uncased_L-10_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1', 'small_bert/bert_en_uncased_L-10_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1', 'small_bert/bert_en_uncased_L-10_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1', 'small_bert/bert_en_uncased_L-10_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1', 'small_bert/bert_en_uncased_L-12_H-128_A-2': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1', 'small_bert/bert_en_uncased_L-12_H-256_A-4': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1', 'small_bert/bert_en_uncased_L-12_H-512_A-8': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1', 'small_bert/bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1', 'albert_en_base': 'https://tfhub.dev/tensorflow/albert_en_base/2', 'electra_small': 'https://tfhub.dev/google/electra_small/2', 'electra_base': 'https://tfhub.dev/google/electra_base/2', 'experts_pubmed': 'https://tfhub.dev/google/experts/bert/pubmed/2', 'experts_wiki_books': 'https://tfhub.dev/google/experts/bert/wiki_books/2', 'talking-heads_base': 'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1', }
map_model_to_preprocess = { 'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'bert_en_cased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2', 'small_bert/bert_en_uncased_L-2_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-2_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-2_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-2_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-4_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-4_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-4_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-4_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-6_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-6_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-6_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-6_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-8_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-8_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-8_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-8_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-10_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-10_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-10_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-10_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-12_H-128_A-2': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-12_H-256_A-4': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-12_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'small_bert/bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'bert_multi_cased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2', 'albert_en_base': 'https://tfhub.dev/tensorflow/albert_en_preprocess/2', 'electra_small': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'electra_base': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'experts_pubmed': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'experts_wiki_books': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', 'talking-heads_base': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2', }
tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features
        
data = {}
def add_unaligned_mfb(video_key):
    wav_path = join(args['wav_dir'], f'{video_key}.wav')
    mfbs, _, _ = new_get_mfbs(wav_path)
    intervals = get_mfb_intervals(MFB_WIN_STEP*mfbs.shape[0], MFB_WIN_STEP)
    if not mfbs.shape[0] == intervals.shape[0]:
        save_pk('temp.pk', {'mfbs': mfbs, 'intervals': intervals})
        assert False, 'See temp.pk for failure details'

    data[video_key] = {
        'features': mfbs,
        'intervals': intervals
    }

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

    # compseq = mmdatasdk.computational_sequence(csd_name)
    # compseq.setData(data, csd_name)
    # metadata_template['root name'] = csd_name
    # compseq.setMetadata(metadata_template, csd_name)
    # compseq.deploy(args["audio_path"].replace('.pk', '.csd'))
    save_pk(args['audio_path'].replace('.csd', '.pk'), data)
    return args["audio_path"]

def csd_to_pk(ds, key, path=None):
    new_text = {}
    for k in ds[key].keys():
        new_text[k] = {
            'features': ar(ds[key][k]['features']),
            'intervals': ar(ds[key][k]['intervals']),
        }
    if path is not None:
        save_pk(path, new_text)
    return new_text

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
# num_labels = 3 # mosei
num_labels = 4 # iemocap
def label_map_fn(labels):
    '''
    input: a one-dimensional array of labels (e.g., shape (10,...))
    output: a one-dimensional array of labels as integers

    e.g.:
        iemocap: input = ['ang', 'ang', 'neu', 'hap','sad'], output = [0,0,1,2,3]
        mosei: input of shape (num_labels,1,7) with values for each of [sentiment,happy,sad,anger,surprise,disgust,fear] . output = [0,1,2] for neg, pos, neu sentiment
    '''
    # # iemocap
    label_map = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    if len(labels.shape) > 1:
        labels = np.squeeze(labels, axis=1)
    return arlmap(lambda elt: label_map[elt.decode('utf8') if type(elt) in [bytes, np.bytes_] else elt], labels).astype('int32')

    # # mosei
    # labels = labels[:,0]
    # labels[labels>0]=1
    # labels[labels==0]=2
    # labels[labels<0]=0
    # return labels.astype('int32')
###

def load_data():
    if exists(args['tensors_path']) and not args['overwrite_tensors'] and not args['mode']=='inference':
        print('Loading data...')
        train, test = load_pk(args['tensors_path'])
        return train, test

    dataset = mmdatasdk.mmdataset(recipe={'dummy': args['dummy_path']})
    del dataset.computational_sequences['dummy']

    if args['mode'] == 'inference':
        if args['wav_dir'][-1]=='/':
            args['wav_dir'] = args['wav_dir'][:-1]
        temp_wav_dir = args['wav_dir']+'_segments'
        
        if ( not exists(args['transcripts_path']) and 'text' in args['modality'] ) or ('audio' in args['modality']):
            convert.split_wavs(args['wav_dir'], temp_wav_dir_in=temp_wav_dir, agg_in=2)

        if not exists(args['transcripts_path']) and 'text' in args['modality']:
            assert args['mode'] == 'inference', 'Transcribing on preformatted datasets is not supported yet'

            wav_paths = glob(join(temp_wav_dir, '*'))
            print(f'Transcribing {args["wav_dir"]} into {temp_wav_dir}')
            transcripts = { wav_path.split('/')[-1].replace('.wav', ''): dict(lzip(['features', 'intervals', 'confidence'], get_transcript(wav_path))) for wav_path in tqdm(wav_paths) }
            save_pk(args['transcripts_path'], transcripts)
        
        if 'audio' in args['modality']:
            args['wav_dir'] = temp_wav_dir
    
    if 'text' in args['modality']:
        add_seq(dataset, args['transcripts_path'], 'text')
    
    if 'audio' in args['modality']:
        deploy_unaligned_mfb_csd(check_labels=(args['mode']!='inference'))
        add_seq(dataset, args['audio_path'], 'audio')

    sub_ds = copy.deepcopy(dataset['audio'] if 'audio' in args['modality'] else dataset['text']) # audio as default b/c alignment will be braodest

    if 'audio' in args['modality'] and 'text' in args['modality']:
        dataset.align('text', collapse_functions=[avg])
        dataset.impute('text')
    
    fake_labels = ( (args['mode'] == 'inference') and (not args['evaluate_inference']) )
    
    if not fake_labels:
        add_seq(dataset, args['labels_path'], 'labels')
    else:
        labels = {k: {'features': ar([[1]]), 'intervals': ar([ [sub_ds[k]['intervals'].min(), sub_ds[k]['intervals'].max()] ]) } for k in sub_ds.keys()}
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
        print('Reshaping data as cross utterance...')
        ## reshape dataset into (num_vids, max_utts, 128, 512)
        b = [(elt.split('[')[0], int(elt.split('[')[1].replace(']', '')), idx) for idx,elt in enumerate(np.squeeze(tensors['ids']))]
        b.sort(key=lambda elt: (elt[0], elt[1]))
        vid_keys,utt_idxs,full_idxs = lzip(*b)
        max_utts = max(utt_idxs)+1

        audio, text, labels, utt_masks = [], [], [], []
        vid_keys = ar(vid_keys)
        unique_vid_keys = np.unique(vid_keys)

        if args['train_keys'] is None:
            args['train_keys'], args['test_keys'] = train_test_split(unique_vid_keys, test_size=.2, random_state=11)

        if args['mode'] == 'inference':
            args['test_keys'] = np.squeeze(unique_vid_keys)
            args['train_keys'] = ar([])

        train_idxs = np.where(arlmap(lambda elt: elt in args['train_keys'], unique_vid_keys))[0]
        test_idxs = np.where(arlmap(lambda elt: elt in args['test_keys'], unique_vid_keys))[0]
        assert len(train_idxs) + len(test_idxs) == len(unique_vid_keys), 'If this assertion fails, it means not all video keys were accounted for in the keys provided'

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
                relevant_labels = label_map_fn(relevant_labels)
                utt_padded_labels = np.pad(relevant_labels, ((0,max_utts-num_utts)), 'constant')
                labels.append(utt_padded_labels)
            
            utt_mask = np.ones(max_utts)
            utt_mask[num_utts:] = 0
            utt_masks.append(utt_mask)
        
        # get text in sentence form: (num_vids, max_utts)
        text = np.apply_along_axis(lambda row: b' '.join(row), -1, text)
        v = np.vectorize(lambda elt: elt[:(elt.find(b'0.0')-1)])
        text = v(text)
        
        del tensors # GPU memory
        del dataset

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
                encoded = np.pad(encoded, ((0, max_utts-num_utts), (0,0), (0,0)), 'constant')
                new_data.append(encoded)
            text = ar(new_data)

        if 'audio' in args['modality'] and 'text' in args['modality']:
            train = ar(audio)[train_idxs], ar(text)[train_idxs], ar(labels)[train_idxs], ar(utt_masks)[train_idxs]
            test = ar(audio)[test_idxs], ar(text)[test_idxs], ar(labels)[test_idxs], ar(utt_masks)[test_idxs]

        elif 'audio' in args['modality']:
            train = ar(audio)[train_idxs], ar(labels)[train_idxs], ar(utt_masks)[train_idxs]
            test = ar(audio)[test_idxs], ar(labels)[test_idxs], ar(utt_masks)[test_idxs]

        elif 'text' in args['modality']:
            train = ar(text)[train_idxs], ar(labels)[train_idxs], ar(utt_masks)[train_idxs]
            test = ar(text)[test_idxs], ar(labels)[test_idxs], ar(utt_masks)[test_idxs]

    else: # within utterance
        labels = label_map_fn(np.squeeze(tensors['labels'])) if not fake_labels else np.ones(tensors[args['modality'].split(',')[0]].shape[0:1])
        ids = np.squeeze(tensors['ids'])

        if args['train_keys'] is None:
            args['train_keys'], args['test_keys'] = train_test_split(np.squeeze(tensors['ids']), test_size=.2, random_state=11)
        
        if args['mode'] == 'inference':
            args['test_keys'] = np.squeeze(tensors['ids'])
            args['train_keys'] = ar([])

        train_idxs = np.where(arlmap(lambda elt: elt in args['train_keys'], np.squeeze(tensors['ids'])))[0]
        test_idxs = np.where(arlmap(lambda elt: elt in args['test_keys'], np.squeeze(tensors['ids'])))[0]
        assert len(train_idxs) + len(test_idxs) == len(tensors['ids']), 'If this assertion fails, it means not all utterance keys were accounted for in the keys provided'
    
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
            test = text[test_idxs], labels[test_idxs], ids[test_idxs]

        elif 'audio' in args['modality'] and 'text' not in args['modality']:
            train = audio[train_idxs], labels[train_idxs], ids[train_idxs]
            test = audio[test_idxs], labels[test_idxs], ids[test_idxs]
        
        else: # both
            train = text[train_idxs], audio[train_idxs], labels[train_idxs], ids[train_idxs]
            test = text[test_idxs], audio[test_idxs], labels[test_idxs], ids[test_idxs]

    if args['mode'] != 'inference':
        print(f'Saving tensors to {args["tensors_path"]}')
        save_pk(args['tensors_path'], (train, test))
    return train, test
    
def train_cross_multi(train, test):
    train_audio, train_text, train_labels, train_utt_masks = train
    test_audio, test_text, test_labels, test_utt_masks = test

    train_cross_uni_audio(
        train=(train_audio, train_labels, train_utt_masks), 
        test=(test_audio, test_labels, test_utt_masks)
    )
    train_cross_uni_text(
        train=(train_text, train_labels, train_utt_masks), 
        test=(test_text, test_labels, test_utt_masks)
    )

    import hffn
    u = load_pk(args['uni_path'])
    return hffn.multimodal(u, args)


def train_within_multi(train,test):
    train_audio, train_text, train_labels, train_ids = train
    test_audio, test_text, test_labels, test_ids = test

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
        epochs=500,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=20)],
        verbose=1,
    ).history
    eval_history = model.evaluate(
        x={'text': test_text, 'audio': test_audio},
        y=test_labels,
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


def train_cross_uni_audio(train,test):
    train_data, train_labels, train_utt_masks = train
    test_data, test_labels, test_utt_masks = test

    input = Input(shape=(train_data.shape[1],train_data.shape[2],train_data.shape[3]))
    conv = TimeDistributed(Conv1D(filters=args['filters_audio'], dilation_rate=1, kernel_size=3, padding='same', data_format='channels_last', dtype='float32'))(input)
    drop = TimeDistributed(Dropout(args['drop_audio']))(conv)
    conv2 = TimeDistributed(Conv1D(filters=args['filters_audio'], dilation_rate=2, kernel_size=4, padding='same', data_format='channels_last', dtype='float32'))(drop)
    drop2 = TimeDistributed(Dropout(args['drop_audio']))(conv2)
    mp = TimeDistributed(MaxPool1D(pool_size=4, data_format='channels_last'))(drop2)
    conv3 = TimeDistributed(Conv1D(filters=args['filters_audio'], dilation_rate=2, kernel_size=2, padding='same', data_format='channels_last', dtype='float32'))(mp)
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
        validation_split=0.15,
        callbacks=[EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=30)],
    ).history
    eval_history = model.evaluate(
        x=test_data,
        y=test_labels,
        sample_weight=test_utt_masks,
        batch_size=args['bs'],
    )
    mkdirp(args['model_path'])
    print('Saving model...')
    model.save(args['model_path'], include_optimizer=False)

    uni = load_pk(args['uni_path'])
    uni = {} if uni is None else uni
    uni['audio_train'] = aux.predict(x=train_data, batch_size=10)
    uni['audio_train_mask'] = train_utt_masks
    uni['audio_train_label'] = train_labels
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

def train_within_uni_audio(train,test):
    train_data, train_labels, train_ids = train
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
        epochs=args['epochs'],
        validation_split=0.15,
        callbacks=[EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=30)],
    ).history
    eval_history = model.evaluate(
        x=test_data,
        y=test_labels,
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


def train_cross_uni_text(train,test):
    train_data, train_labels, train_utt_masks = train
    test_data, test_labels, test_utt_masks = test

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
        validation_split=0.15,
        callbacks=[EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=30)],
    ).history
    eval_history = model.evaluate(
        x=test_data,
        y=test_labels,
        sample_weight=test_utt_masks,
        batch_size=args['bs'],
    )
    mkdirp(args['model_path'])
    print('Saving model...')
    model.save(args['model_path'], include_optimizer=False)

    uni = load_pk(args['uni_path'])
    uni = {} if uni is None else uni
    uni['text_train'] = aux.predict(x=train_data, batch_size=10)
    uni['text_train_mask'] = train_utt_masks
    uni['text_train_label'] = train_labels
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

def train_within_uni_text(train, test):
    train_data, train_labels, train_ids = train
    test_data, test_labels, test_ids = test

    def res_block(x, filters):
        x_skip = x

        x = Conv1D(filters=filters, kernel_size=4, dilation_rate=1, padding='same', data_format='channels_last', dtype='float32')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.relu)(x)

        x = Conv1D(filters=filters, kernel_size=8, dilation_rate=2, padding='same', data_format='channels_last', dtype='float32')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.relu)(x)

        x = Conv1D(filters=filters, kernel_size=8, dilation_rate=2, padding='same', data_format='channels_last', dtype='float32')(x)
        x = BatchNormalization()(x)
        
        x = Add()([x, x_skip])

        x = Activation(tf.keras.activations.relu)(x)
        return x

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
        batch_size=args['bs'],
        epochs=args['epochs'],
        validation_split=0.15,
        callbacks=[EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=10)],
    ).history
    eval_history = model.evaluate(
        x=test_data,
        y=test_labels,
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

def main(args_in):
    global args
    args = args_in

    train, test = load_data()
    if args['cross_utterance']:
        if 'text' in args['modality'] and 'audio' in args['modality']:
            return train_cross_multi(train, test)

        elif 'text' in args['modality']:
            return train_cross_uni_text(train,test)

        elif 'audio' in args['modality']:
            return train_cross_uni_audio(train,test)
    
    else: # within
        if 'text' in args['modality'] and 'audio' in args['modality']:
            return train_within_multi(train,test)
        elif 'text' in args['modality']:
            return train_within_uni_text(train,test)
        elif 'audio' in args['modality']:
            return train_within_uni_audio(train,test)

def main_inference(args_in):
    global args
    args = args_in

    _, test = load_data() # only consider test data with dummy labels

    print('Loading classifier...')
    model = tf.keras.models.load_model(args['model_path'])
    print('Predicting...')
    if not args['cross_utterance']: # within
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args['text_lr']),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['sparse_categorical_accuracy'],
        )

        # max_utts = model._build_input_shape[1] #56 for iemocap
        if len(args['modality'].split(','))>1: # multimodal
            test_audio, test_text, test_labels, ids = test
            if args['evaluate_inference']:
                preds = model.evaluate({'text': test_text, 'audio': test_audio}, test_labels, batch_size=args['bs'])
            else:
                preds = model.predict({'text': test_text, 'audio': test_audio}, batch_size=args['bs'])

        else:
            test_data, test_labels, ids = test
            if args['evaluate_inference']:
                preds = model.evaluate(test_data, test_labels, batch_size=args['bs'])
            else:
                preds = model.predict(test_data)

    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args['text_lr']),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['sparse_categorical_accuracy'],
            sample_weight_mode='temporal',
        )
        max_utts = model._build_input_shape[1] #56 for iemocap
        if len(args['modality'].split(','))>1: # multimodal
            test_audio, test_text, test_labels, test_utt_masks = test
            if args['evaluate_inference']:
                preds = model.evaluate({'text': test_text, 'audio': test_audio}, test_labels, batch_size=args['bs'], sample_weight=test_utt_masks)
            else:
                preds = model.predict({'text': test_text, 'audio': test_audio}, batch_size=args['bs'])

        else:
            test_data, test_labels, test_utt_masks = test
            if args['evaluate_inference']:
                preds = model.evaluate(test_data, test_labels, batch_size=args['bs'], sample_weight=test_utt_masks)
            else:
                preds = model.predict(test_data)
    
    a = {k: ' '.join(np.squeeze(v['features'])) for k,v in load_pk(args['transcripts_path']).items()}
    label_map = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    reverse_label_map = ['ang', 'hap', 'neu', 'sad']
    
    for i,id in enumerate(ids):
        id = id.split('[')[0]
        print('Transcript:', a[id], '; ', end='')
        label = np.argmax(preds, axis=-1)[i]
        print('Label:', reverse_label_map[label])

    full_res = {
        'data': test_data if len(args['modality'].split(','))==1 else (test_audio, test_text),
        'utt_masks': None if not args['cross_utterance'] else test_utt_masks,
        'predictions': preds,
        'ids': ids,
    }
    return full_res

if __name__ == '__main__':
    parser = standard_grid.ArgParser()

    tensors_base_path = join(BASE_PATH, 'tensors')
    data_path = join(BASE_PATH, 'data')
    ie_path = join(data_path, 'iemocap')
    models_path = join(BASE_PATH, 'models')
    uni_path = join(data_path, 'uni.pk')
    mkdirp(tensors_base_path)
    mkdirp(ie_path)
    mkdirp(models_path)

    params = [
        # core paths
        ('--transcripts_path',str,join(ie_path, 'IEMOCAP_TimestampedWords.pk'), 'Path to transcripts.'),
        ('--audio_path',str,join(ie_path, 'mfb.pk'), 'Where mfbs will be (or are already) saved.'),
        ('--labels_path',str, join(ie_path, 'IEMOCAP_EmotionLabels.pk'), 'Path to labels.'),
        ('--dummy_path',str, join(ie_path, 'IEMOCAP_EmotionLabels.pk'), 'Used to initialize dataset.  Can be any valid labels file.  You probably will not change this.'),
        ('--tensors_path',str, join(tensors_base_path, 'tensors.pk'), 'Where tensors will be stored for this dataset.  If "unique", will store a hash in tensors folder.'),
        ('--wav_dir',str, join(ie_path, 'wavs'), 'Path to wav dir.'),
        ('--model_path',str,join(models_path, 'model'), 'Where model will be saved.'),
        ('--uni_path',str,uni_path, 'Where unimodal activations are stored before feeding into hffn (in the case of cross utterance multimodal)'),

        # core options
        ('--mode',str,'train', 'train or inference'),
        ('--cross_utterance', int, 0, 'If 0, build a within-utterance model.  If 1, build a cross utterance model.'),
        ('--modality', str,'text', 'modalities separated by ,.  options are ["text,audio", "audio,text", "audio", "text"]'),
        ('--evaluate_inference', int, 0, 'If mode==inference and this is 1, evaluate using labels passed in instead of returning prediction.  This can be a good sanity check on the training process if you have labels, and want to see how well your saved model is inferring on some dataset.'),
        ('--overwrite_tensors', int, 0, 'Do you want to overwrite tensors in tensors_path or not?  By default, we will use tensors we find in tensors_path instead of regenerating.'),
        ('--overwrite_mfbs', int, 0, 'Do you want to overwrite the mfbs in audio_path by recreating the mfbs from wav_dir?'),
        ('--keys_path',str, join(ie_path, 'keys.json'), '(Optional) path to json file with keys "train_keys" and "test_keys" which each contain nonoverlapping video (if cross-utterance) / utterance (if within) key lists'),

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
    for param in params:
        parser.register_parameter(*param)

    args = vars(parser.compile_argparse())

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
        # full_res['train_keys'] = args['train_keys']
        # full_res['test_keys'] = args['test_keys']
        save_json(join(out_dir, 'results.txt'), full_res)

    else:
        full_res = main_inference(args)
        save_pk(join(out_dir, 'inference.pk'), full_res)
