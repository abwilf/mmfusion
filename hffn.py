# HFFN module called by main
from utils import *
from consts import *
sys.path.append(MMSDK_PATH)
from mmsdk import mmdatasdk
sys.path.append(STANDARD_GRID_PATH)
import standard_grid

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils import class_weight
from notebook_util import setup_no_gpu, setup_one_gpu, setup_gpu
# setup_one_gpu()
# setup_no_gpu()

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

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

os.environ['KERAS_BACKEND'] = 'tensorflow'
from util_function import *
import warnings, os
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras.layers import TimeDistributed, Flatten, Dense, Input, Activation, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
import pickle
import sys
import argparse

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def calc_test_result(result, test_label, test_mask):
    true_label=[]
    predicted_label=[]
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]==1:
                true_label.append(np.argmax(test_label[i,j] ))
                predicted_label.append(np.argmax(result[i,j] ))
    
    acc = accuracy_score(true_label, predicted_label)
    print("Accuracy {:.4f}".format(acc))

def segmentation(text, audio, size, stride):
    # print('text',text.shape,'audio',audio.shape)
    s = stride; length = text.shape[2]
    local = int((length-size)/s) + 1
    if (length-size)%s != 0 :
        k = (length-size)%s
        pad = size - k
        text = np.concatenate((text,np.zeros([text.shape[0],text.shape[1],pad])),axis = 2)
        audio = np.concatenate((audio,np.zeros([text.shape[0],text.shape[1],pad])),axis = 2)
        local +=1
    input1 =  np.zeros([text.shape[0],text.shape[1],local,2*size])
    fusion = np.zeros([text.shape[0],text.shape[1],local,(size+1)**2])

    for i in range(local):
        text1 = text[:,:,s*i:s*i+size]
        text2 = text1
        text1 = np.concatenate((text1,np.ones([text.shape[0],text.shape[1],1])),axis = 2)
        text1 = text1[:,:,:,np.newaxis]

        audio1 = audio[:,:,s*i:s*i+size] 
        audio2 = audio1
        audio1 = np.concatenate((audio1,np.ones([text.shape[0],text.shape[1],1])),axis = 2)
        audio1 = audio1[:,:,np.newaxis,:]

        ta = np.matmul(text1,audio1)

        ta = np.squeeze(np.reshape(ta,[text.shape[0],text.shape[1],(size+1)**2,1]), axis=-1)
        fusion[:,:,i,:] = ta

        input1[:,:,i,0:size] = text2
        input1[:,:,i,size:size*2] = audio2
    return fusion, input1, local


def multimodal(unimodal_activations, args):
    #Fusion (appending) of features
        #[62 63 50] [62 63 150]

    args = {
        'segmentation_size': 2,
        'segmentation_stride': 2,
        'batch_size': 10,
        'average_type': 'macro',
        'verbose': 1,
        'overwrite_models': 1,
        'test': False,
    }

    unimodal_activations['train_mask'] = unimodal_activations['text_train_mask']
    unimodal_activations['test_mask'] = unimodal_activations['text_test_mask']
    unimodal_activations['train_label'] = unimodal_activations['text_train_label']
    unimodal_activations['test_label'] = unimodal_activations['text_test_label']

    # model_save_path = join(args['model_path'], 'hffn')
    # model_sub_save_path = join(model_save_path, 'hffn')
    train_mask=unimodal_activations['train_mask'] # 0 or 1
    test_mask=unimodal_activations['test_mask']
    train_label=onehot_initialization(unimodal_activations['train_label'])
    test_label=onehot_initialization(unimodal_activations['test_label'])
    #  concat = Lambda(lambda x: K.concatenate([x[0],x[1]],axis=-1))
    # padd = np.ones([62,63,1])

    text = unimodal_activations['text_train']
    audio = unimodal_activations['audio_train']
    fusion, _, _ = segmentation(text, audio, args['segmentation_size'], args['segmentation_stride'])

    text = unimodal_activations['text_test']
    audio = unimodal_activations['audio_test']
    fusion2, _, _ = segmentation(text, audio, args['segmentation_size'], args['segmentation_stride'])

    input_data = Input(shape=(fusion.shape[1],fusion.shape[2],fusion.shape[3]))  #???

    lstm3 = TimeDistributed(ABS_LSTM4(units=3, intra_attention=True, inter_attention=True))(input_data)  # or ABS_LSTM5
    lstm3 = TimeDistributed(Activation('tanh'))(lstm3)  #tanh
    lstm3 = TimeDistributed(Dropout(0.6))(lstm3)   #0.6
    fla = TimeDistributed(Flatten())(lstm3)
    uni = TimeDistributed(Dense(50,activation='relu'))(fla)   ####50
    uni = Dropout(0.5)(uni)
    output = TimeDistributed(Dense(4, activation='softmax'))(uni) 
    # output = TimeDistributed(Dense(args['train_label'].shape[-1], activation='linear'))(uni) 
    model = Model(input_data, output)
    # if False:
    model_save_path = '/z/abwilf/mmfusion/temp/model'
    model_sub_save_path = '/z/abwilf/mmfusion/temp/model/weights'
    if exists(model_save_path) and not args['overwrite_models']: # can't use exists b/c load and save weights
        print('Using saved hffn model')
        model.load_weights(model_sub_save_path)
    else:
        print('Training hffn...')
        model.compile(optimizer='RMSprop', loss='cosine_similarity', weighted_metrics=['categorical_accuracy'], sample_weight_mode='temporal')
        # model.compile(optimizer='RMSprop', loss='mse', sample_weight_mode='temporal')
        # model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        
        # class_weight_mask = np.argmax(train_label, axis=-1).astype('float32')
        # for class_val, weight in args['class_weights'].items():
        #     class_weight_mask[class_weight_mask==class_val] = weight
        # class_weight_mask = class_weight_mask * train_mask

        train_history = model.fit(fusion, train_label,
            epochs=1 if args['test'] else 1000,
            steps_per_epoch=20 if args['test'] else None,
            batch_size=10,
            # sample_weight=class_weight_mask if args['average_type'] == 'macro' else train_mask,
            sample_weight=train_mask,
            shuffle=True, 
            callbacks=[early_stopping],
            validation_split=0.2,
            verbose=args['verbose'],
        ).history
        print('Saving model...')
        model.save_weights(model_sub_save_path)
    
    eval_history = model.evaluate(
        fusion2,
        test_label,
        sample_weight=test_mask,
    )
    result = model.predict(fusion2)
    calc_test_result(result, test_label, test_mask)
    res = {
        'train_losses': ar(train_history['loss']),
        'train_accs': ar(train_history['categorical_accuracy']),
        'val_losses': ar(train_history['val_loss']),
        'val_accs': ar(train_history['val_categorical_accuracy']),
        'test_acc': eval_history[1],
        'test_loss': eval_history[0],
    }
    return res

