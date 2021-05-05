from main import *

def full_inference(speaker_profile):
    default = {elt[0].replace('--',''): elt[2] for elt in params}

    ## valence
    args = {
        **default,
        'modality': 'text,audio',
        'tensors_path': 'tensors.pk',
        'overwrite_mfbs': 1,
        'mode': 'inference',
        'print_transcripts': 1,
        'seq_len': 150,
        'model_path': 'val_model',
        'cross_utterance': 0,
        'num_labels': num_labels,
        'speaker_profile': speaker_profile,
        
        ## IF predicting
        # 'evaluate_inference': 0,
        # 'labels_path': '',
        # 'transcripts_path': 'preds/transcripts.pk',
        # 'audio_path': 'preds/mfb.pk',
        # 'wav_dir': 'preds/wavs',

        ## IF testing
        'labels_path': 'test_data/val_utt_labels.json',
        'evaluate_inference': 1,
        'transcripts_path': 'test_data/transcripts.pk',
        'audio_path': 'test_data/mfb.pk',
        'wav_dir': 'test_data/wavs',
    }
    rmfile(args['audio_path'])
    val_inf = main_inference(args)

    # reshaping from cross utterance format
    preds = val_inf['predictions']
    ids = val_inf['ids']
    speaker_ver = val_inf['speaker_ver']
    rmfile(args['tensors_path'])

    ## activation
    args = {
        **default,
        'modality': 'audio',
        'tensors_path': 'tensors.pk',
        'transcripts_path': 'test_data/transcripts.pk',
        'audio_path': 'test_data/mfb.pk',
        'wav_dir': 'test_data/wavs',
        'overwrite_mfbs': 1,
        'mode': 'inference',
        'evaluate_inference': 0,
        'print_transcripts': 1,
        'seq_len': 35000,
        'model_path': 'act_model',
        'cross_utterance': 0,
        'num_labels': num_labels,
        'speaker_profile': '',
        
        'labels_path': 'test_data/act_utt_labels.json',
        'evaluate_inference': 1,
    }
    act_inf = main_inference(args)

    rmfile(args['tensors_path'])
    # rmfile(args['transcripts_path'])

    res = {
        'val': preds,
        'val_ids': ids,
        'act': act_inf['predictions'],
        'act_ids': act_inf['ids'],
        'speaker_ver': speaker_ver
    }
    rmfile(args['audio_path'])
    save_pk('output.pk', res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker_profile', type=str, help='Speaker profile id', required=True)
    args = vars(parser.parse_args())
    full_inference(args['speaker_profile'])
