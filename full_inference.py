from main import *

def full_inference(speaker_profile):
    default = {elt[0].replace('--',''): elt[2] for elt in params}

    ## valence
    args = {
        **default,
        'modality': 'text,audio',
        'tensors_path': 'preds/tensors.pk',
        'overwrite_mfbs': 1,
        'mode': 'inference',
        'print_transcripts': 1,
        'seq_len': 150,
        'model_path': 'val_model',
        'cross_utterance': 0,
        'num_labels': 3,
        'speaker_profile': speaker_profile,
        
        ## IF predicting
        'evaluate_inference': 0,
        'labels_path': '',
        'transcripts_path': 'preds/transcripts.pk',
        'audio_path': 'preds/mfb.pk',
        'wav_dir': 'preds/wavs',

        ## IF testing
        # 'evaluate_inference': 1,
        # 'labels_path': 'test_data/val_utt_labels.json',
        # 'transcripts_path': 'test_data/transcripts.pk',
        # 'audio_path': 'test_data/mfb.pk',
        # 'wav_dir': 'test_data/wavs',
    }
    rmfile(args['audio_path'])
    val_inf = main_inference(args)

    # reshaping from cross utterance format
    val_preds = val_inf['predictions']
    val_ids = val_inf['ids']
    speaker_ver = val_inf['speaker_ver']
    segment_lengths = val_inf['segment_lengths']
    rmfile(args['tensors_path'])

    ## activation
    args = {
        **default,
        'modality': 'audio',
        'tensors_path': 'preds/tensors.pk',
        'overwrite_mfbs': 1,
        'mode': 'inference',
        'print_transcripts': 1,
        'seq_len': 35000,
        'model_path': 'act_model',
        'cross_utterance': 0,
        'num_labels': 3,
        'speaker_profile': '',
        'evaluate_inference': 1,

        ## IF predicting
        'evaluate_inference': 0,
        'labels_path': '',
        'transcripts_path': 'preds/transcripts.pk',
        'audio_path': 'preds/mfb.pk',
        'wav_dir': 'preds/wavs',

        ## IF testing
        # 'evaluate_inference': 1,
        # 'labels_path': 'test_data/act_utt_labels.json',
        # 'transcripts_path': 'test_data/transcripts.pk',
        # 'audio_path': 'test_data/mfb.pk',
        # 'wav_dir': 'test_data/wavs',

    }
    act_inf = main_inference(args)

    rmfile(args['tensors_path'])
    # rmfile(args['transcripts_path'])


    def get_three_bin(arr):
        return lmap(lambda elt: ';'.join(elt.astype(str)), np.round(arr, decimals=4))

    res = {
        'val': get_three_bin(val_preds),
        'val_ids': val_ids,
        'act': get_three_bin(act_inf['predictions']),
        'act_ids': act_inf['ids'],
        'speaker_ver': speaker_ver,
        'segment_lengths': segment_lengths,
    }

    # WIPE PREDS
    shutil.rmtree('preds')
    mkdirp('preds/wavs')

    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker_profile', type=str, help='Speaker profile id', required=True)
    args = vars(parser.parse_args())
    full_inference(args['speaker_profile'])

