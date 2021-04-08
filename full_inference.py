from main import *

default = {elt[0].replace('--',''): elt[2] for elt in params}

## valence
args = {
    **default,
    'modality': 'text,audio',
    'tensors_path': 'tensors.pk',
    'transcripts_path': 'test_data/transcripts.pk',
    'audio_path': 'test_data/mfb.pk',
    'wav_dir': 'test_data/wavs',
    'overwrite_mfbs': 1,
    'mode': 'inference',
    'evaluate_inference': 0,
    'print_transcripts': 0,
    'seq_len': 150,
    'hffn_path': 'val_model',
    'train_keys': [],
    'cross_utterance': 1,
    'num_labels': num_labels,
}
rmfile(args['audio_path'])
rmfile(args['transcripts_path'])
val_inf = main_inference(args)

# reshaping from cross utterance format
rel_idxs = np.where(val_inf['utt_masks'].flatten())[0]
preds = val_inf['predictions'].flatten()[rel_idxs]
ids = val_inf['ids']
rmfile(args['tensors_path'])

## activation
args = {
    **default,
    'modality': 'audio',
    'tensors_path': 'tensors.pk',
    'transcripts_path': 'test_data/transcripts.pk',
    'audio_path': 'test_data/mfb.pk',
    'wav_dir': 'test_data/wavs',
    'overwrite_mfbs': 0,
    'mode': 'inference',
    'evaluate_inference': 0,
    'print_transcripts': 0,
    'seq_len': 35000,
    'model_path': 'act_model',
    'train_keys': [],
    'cross_utterance': 0,
    'num_labels': num_labels,
}
act_inf = main_inference(args)

rmfile(args['tensors_path'])
rmfile(args['transcripts_path'])

res = {
    'val': preds,
    'val_ids': ids,
    'act': act_inf['predictions'],
    'act_ids': act_inf['ids'],
}
rmfile(args['audio_path'])
save_pk('output.pk', res)
