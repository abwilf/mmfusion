######### models.py #########
# We built support for different kinds of models - ones that considered within utterance or cross utterance context unimodally (audio xor text) and multimodally (audio + text).  This 

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

def train_cross_multi(train, val, test):
    # hffn
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
    clf = Dense(args['num_labels'], activation='softmax')(drop2)

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
    dense2 = TimeDistributed(Dense(args['num_labels'], activation='softmax'))(drop5)

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
    dense2 = Dense(args['num_labels'], activation='softmax')(drop5)

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
    clf = TimeDistributed(Dense(args['num_labels'], activation='softmax'))(drop2)

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
    clf = Dense(args['num_labels'], activation='softmax')(drop2)

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
