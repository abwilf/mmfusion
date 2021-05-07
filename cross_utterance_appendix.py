
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
