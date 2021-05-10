# Multimodal Fusion
This package offers unimodal and multimodal approaches to within-utterance and cross-utterance classification problems. In essence, this module takes in a transcripts file and a path to a directory containing wavs, generates MFB's from those wavs, aligns the MFB's with the transcripts, extracts word embeddings using BERT, packages everything together in a single `tensors` object (which is saved to disk in `--tensors_path`), then trains and tests on this object according to some parameters you pass in.  In this README, I first describe the requirements, then the data format for how to add new datasets, then finally the parameters required to run the program.

## Requirements
1. `conda` is installed
2. If you wish to use a GPU, `CUDA` must be compatible with `tensorflow-2.3.0`: e.g.: `CUDA Version 10.1.243; CUDNN 7.6`

## Usage

### Depdendencies
Create and activate a new environment
```
conda create -y -n mmfusion python=3.7
conda activate mmfusion
```

Install dependencies
```
pip install -r requirements.txt
```

### Constants
`git clone` these two repositories wherever you'd like
* [Standard-Grid](https://github.com/abwilf/standard-grid)
* [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK/)

Modify `consts.py` to reflect the paths you've chosen
```python
STANDARD_GRID_PATH = '/z/abwilf/Standard-Grid/'
MMSDK_PATH = '/z/abwilf/CMU-MultimodalSDK/'
BASE_PATH = '/z/abwilf/mmf' # the path to this directory
```

Create and save a file called `azure_secrets.json`, with the following code inside.  If you decide to use azure for speech recognition in the future, this is where your API information will go.  If you are using this from the CHAI lab, contact me for our keys.
```
{
    "cognitive_key": "...",
    "service_region": "...",
    "speaker_verification_key": "..."
}
```

### Enrolling Users for Speaker Verification
Ask the user to speak into a microphone for at least 30 seconds in English (preferably more). Noise should be minimal. Save this as a wav file (mono, signed 16-bit PCM encoding; Audacity is free and does this perfectly). Name this file `enrollment.wav`, then run the enrollment program.

```python
python3 speaker_verification.py
```

You should see the resulting profile ID in a few seconds.
```
Profile id is:
8c4af779-771f-4f5e-9dcf-7be56b28cdd2
Enrollment successful!
```

When you add the person to the database, make sure this is in their information, and is passed to the inference program as their `speaker_profile` in `full_inference.py` when they upload a wav (this should have been implemented by Owen).

If you need to list users, remove users, or identify users, there are helper functions in `speaker_verification.py` to do so.


### Training Data Format
The three main inputs to our training pipeline are the transcripts, the wav directory, and the labels.

#### Transcripts
The transcripts must be a dictionary, saved in `--transcripts_path` as a pickle file and formatted as follows:
```python
{
    'wav_id': {
        'features': [['word1'], ['word2']..., ['wordn']], # np array of shape (n,1) for n words in the recording (across multiple utterances), stored in utf8 encoding (standard strings) or binary encoding
        'intervals': [ [0,1.2], [1.2, 2.3]...], # np array of shape (n,2) of floats describing the start and end time of each word in seconds from the start of the wav file
    }
}
```

For example, to see our transcripts for IEMOCAP, open up this file using `pickle`. I use helper functions `load_pk` and `save_pk` from `utils.py` to do this quickly.
```
python3
from utils import *
tran = load_pk('data/iemocap/IEMOCAP_TimestampedWords.pk')
k = lkeys(tran)[0]
print(tran[k])
```

#### Wavs
The wavs must be in a directory specified by `--wav_dir` with their names given as `{wav_id}.wav`.  For example, our IEMOCAP wavs look like this (CHAI members: all paths are on `lotus`)

```
ls /z/abwilf/iemocap_wavs/clean

Ses01F_impro01.wav     Ses01M_impro04.wav     Ses02F_impro07.wav     Ses02M_script01_1.wav  Ses03F_script01_3.wav  Ses03M_script01_3.wav  Ses04F_script02_2.wav  Ses04M_script03_2.wav	Ses05M_impro02.wav
Ses01F_impro02.wav     Ses01M_impro05.wav     Ses02F_impro08.wav     Ses02M_script01_2.wav  Ses03F_script02_1.wav  Ses03M_script02_1.wav  Ses04F_script03_1.wav  Ses05F_impro01.wav	Ses05M_impro03.wav
Ses01F_impro03.wav     Ses01M_impro06.wav     Ses02F_script01_1.wav  Ses02M_script01_3.wav  Ses03F_script02_2.wav  Ses03M_script02_2.wav  Ses04F_script03_2.wav  Ses05F_impro02.wav	Ses05M_impro04.wav
Ses01F_impro04.wav     Ses01M_impro07.wav     Ses02F_script01_2.wav  Ses02M_script02_1.wav  Ses03F_script03_1.wav  Ses03M_script03_1.wav  Ses04M_impro01.wav	 Ses05F_impro03.wav	Ses05M_impro05.wav
Ses01F_impro05.wav     Ses01M_script01_1.wav  Ses02F_script01_3.wav  Ses02M_script02_2.wav  Ses03F_script03_2.wav  Ses03M_script03_2.wav  Ses04M_impro02.wav	 Ses05F_impro04.wav	Ses05M_impro06.wav
Ses01F_impro06.wav     Ses01M_script01_2.wav  Ses02F_script02_1.wav  Ses02M_script03_1.wav  Ses03M_impro01.wav	   Ses04F_impro01.wav	  Ses04M_impro03.wav	 Ses05F_impro05.wav	Ses05M_impro07.wav
Ses01F_impro07.wav     Ses01M_script01_3.wav  Ses02F_script02_2.wav  Ses02M_script03_2.wav  Ses03M_impro02.wav	   Ses04F_impro02.wav	  Ses04M_impro04.wav	 Ses05F_impro06.wav	Ses05M_impro08.wav
Ses01F_script01_1.wav  Ses01M_script02_1.wav  Ses02F_script03_1.wav  Ses03F_impro01.wav     Ses03M_impro03.wav	   Ses04F_impro03.wav	  Ses04M_impro05.wav	 Ses05F_impro07.wav	Ses05M_script01_1.wav
Ses01F_script01_2.wav  Ses01M_script02_2.wav  Ses02F_script03_2.wav  Ses03F_impro02.wav     Ses03M_impro04.wav	   Ses04F_impro04.wav	  Ses04M_impro06.wav	 Ses05F_impro08.wav	Ses05M_script01_2.wav
Ses01F_script01_3.wav  Ses01M_script03_1.wav  Ses02M_impro01.wav     Ses03F_impro03.wav     Ses03M_impro05a.wav    Ses04F_impro05.wav	  Ses04M_impro07.wav	 Ses05F_script01_1.wav	Ses05M_script01_3.wav
Ses01F_script02_1.wav  Ses01M_script03_2.wav  Ses02M_impro02.wav     Ses03F_impro04.wav     Ses03M_impro05b.wav    Ses04F_impro06.wav	  Ses04M_impro08.wav	 Ses05F_script01_2.wav	Ses05M_script02_1.wav
Ses01F_script02_2.wav  Ses02F_impro01.wav     Ses02M_impro03.wav     Ses03F_impro05.wav     Ses03M_impro06.wav	   Ses04F_impro07.wav	  Ses04M_script01_1.wav  Ses05F_script01_3.wav	Ses05M_script02_2.wav
Ses01F_script03_1.wav  Ses02F_impro02.wav     Ses02M_impro04.wav     Ses03F_impro06.wav     Ses03M_impro07.wav	   Ses04F_impro08.wav	  Ses04M_script01_2.wav  Ses05F_script02_1.wav	Ses05M_script03_1.wav
Ses01F_script03_2.wav  Ses02F_impro03.wav     Ses02M_impro05.wav     Ses03F_impro07.wav     Ses03M_impro08a.wav    Ses04F_script01_1.wav  Ses04M_script01_3.wav  Ses05F_script02_2.wav	Ses05M_script03_2.wav
Ses01M_impro01.wav     Ses02F_impro04.wav     Ses02M_impro06.wav     Ses03F_impro08.wav     Ses03M_impro08b.wav    Ses04F_script01_2.wav  Ses04M_script02_1.wav  Ses05F_script03_1.wav	temp
Ses01M_impro02.wav     Ses02F_impro05.wav     Ses02M_impro07.wav     Ses03F_script01_1.wav  Ses03M_script01_1.wav  Ses04F_script01_3.wav  Ses04M_script02_2.wav  Ses05F_script03_2.wav
Ses01M_impro03.wav     Ses02F_impro06.wav     Ses02M_impro08.wav     Ses03F_script01_2.wav  Ses03M_script01_2.wav  Ses04F_script02_1.wav  Ses04M_script03_1.wav  Ses05M_impro01.wav

```

#### Labels
Labels must be stored similarly to transcripts, in a pickle file in `--labels_path` containing a python object of the form:
```python
{
    'wav_id': {
        'features': [[0], [1], [0], [2]...] # np array of type int32 (for categorical tasks) and shape (m,1) for m utterances in this wav_id
        'intervals': [[6.2, 7.1], ...] # np array of shape (m,2) containing start / end times for each utterance of type float
    }
}
```

To see our IEMOCAP labels, pop them open in a python shell.
```
python3
from utils import *
labels = load_pk('data/iemocap/IEMOCAP_ValenceLabels.pk')
k = lkeys(labels)[0]
print(labels[k])
```

#### Other Relevant Inputs to Data Formatting / Creation
* `--tensors_path`: if `unique`, the program will recreate tensors (by aligning and creating embeddings) each time the program is run in training mode (the default, or specified with `--mode train`). If a tensors path is specified and the tensor exists, the program will use those tensors and skip data preprocessing.  This is a useful feature because if, for example, you'd like to extract features & embeddings on one machine, then train on another (e.g., to extract MFB's and BERT embeddings on ARMIS, then port over a tensor of embeddings to train on a lab machine), you can run the program in training mode with `--tensors_path my_tensors.pk`, end the program when it begins training (or add an `exit()` command at the end of `load_data()` in `main.py` after it saves the tensors), and port `my_tensors.pk` over to the lab machine using `rsync`.  Then, if you specify `--tensors_path my_tensors.pk` in training on the lab machine, you can skip data processing on the lab machines.
* `--audio_path, --overwrite_mfbs`: the path to the MFB's.  If this exists and `--overwrite_mfbs` is 0, the program will not generate new MFB's from `--wav_dir`, else it will.
* `--seq_len`: the legnth of the sequence you'd like to wrap to.  When the text modality is present, this is the number of words you'd like to wrap to.  For example, if the `seq_len` is 10, and an utterance has the words "one two three...eleven", the last word would be chopped off.  If an utterance had the words "one, two, three", there would be zero padding for the last seven slots. If just the audio modality, this is the number of MFB frames you'd like to take.  This number is usually much higher if audio only, in 30-50,000 range depending on the length of your utterances and sampling rate.
* `--keys_path`: Here you can specify keys for the program to train / validate / test against (in case you want to do leave-one-speaker-out validation, or something similar).  See the form in `data/iemocap/utt_keys.json` for details.

### Training
In training, the relevant arguments are:
* `--model_path`: where the trained model should be saved
* `--modality`: which modalities to use. Options are `text` (unimodal lexical), `audio` (unimodal acoustic), or `text,audio` (multimodal).
* `--mode`: this defaults to `train`.

To train a model saved in `val_model` on **valence** labels using both modalities, you can run the following (if you have problems because of the CUDA_VISIBLE_DEVICES line, just remove it - it just selects a GPU)
```
CUDA_VISIBLE_DEVICES=1 python3 main.py --modality text,audio --tensors_path unique --labels_path data/iemocap/IEMOCAP_ValenceLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --audio_path data/iemocap/mfb.pk --wav_dir /z/abwilf/iemocap_wavs/clean --overwrite_mfbs 0 --model_path val_model --seq_len 150 --keys_path data/iemocap/utt_keys.json --mode train
```

To train a model saved in `act_model` on **activation** labels using only the acoustic modality, you can run
```
CUDA_VISIBLE_DEVICES=1 python3 main.py --modality audio --tensors_path unique --labels_path data/iemocap/IEMOCAP_ActivationLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --audio_path data/iemocap/mfb.pk --wav_dir /z/abwilf/iemocap_wavs/clean --seq_len 35000 --overwrite_mfbs 0 --model_path act_model --keys_path data/iemocap/utt_keys.json --mode train
```
    
### Inference
Inference can be broken down into two cases.  First, where labels are known, and we wish to test our models' performance on an unseen test set. Second, where labels are unknown and we simply desire the predictions.

#### When Labels are Known: Evaluate Inference
The flag `--evaluate_inference` allows you to control whether the pipeline will evaluate how well the model performs, or just yield the prediction.

For example, to test the **valence** model on wavs in `test_data/wavs` with labels in `test_data/val_utt_labels.json`, you could use the following command.
```
rm tensors.pk
CUDA_VISIBLE_DEVICES=1 python3 main.py --modality text,audio --tensors_path tensors.pk --labels_path test_data/val_utt_labels.json --transcripts_path test_data/transcripts.pk --audio_path test_data/mfb.pk --wav_dir test_data/wavs --overwrite_mfbs 1 --mode inference --evaluate_inference 1 --print_transcripts 1 --seq_len 150 --model_path val_model
```

A quick aside: above, I said that labels had to be stored as pickle files.  That is preferable, but I've actually built in flexibility for loading / saving labels files as `json` objects so you can edit them directly.  To load / save a python object as `json`, use the `load_json`, `save_json` functions in `utils.py`.  The labels file must be of the following form:
```python
{
    'wav_id': {
        'features': [0,1,2], # shape (m,) for m utterances in wav_id
        'intervals': null # None if python object
    }
}
```

If you wanted to add more wavs to `test_data/wavs` to expand the test set and further validate the performance of the model, it is a bit tricky to add labels, because you first need to know how the wavs will be segmented by the VAD.  To do this, you can run the above program until you hit an error telling you the labels are not all accounted for.  Then you can look in `test_data/wavs_segments` to listen to the individual utterances and manually add the labels.  This is cumbersome and time consuming, but unfortunately I don't see a way around it, as your labels will need to be at the utterance level, but you won't know how utterances are defined until you see how the VAD has split up the wav.


If you wanted to test the **activation** model on the same wavs with labels in `test_data/act_utt_labels.json`, you would use the following command.
```
rm tensors.pk
CUDA_VISIBLE_DEVICES=1 python3 main.py --modality audio --tensors_path rm tensors.pk --labels_path test_data/act_utt_labels.json --transcripts_path test_data/transcripts.pk --audio_path test_data/mfb.pk --wav_dir test_data/wavs --overwrite_mfbs 1 --mode inference --evaluate_inference 1 --print_transcripts 1 --seq_len 35000 --model_path act_model --keys_path data/iemocap/utt_keys.json
```

#### When Labels are Unknown
When labels are unknown, we are performing full inference.   This means that we will also be performing speaker verification on each of the utterances.  To do this, we need to pass the speaker verification profile_id of the participant.  This happens automatically as part of Owen's code, fetching from the database.  To keep everything self contained and secure for production, we store everything we save to the disk in a folder for predictions, `preds`.  So, to see this in action add some wav files to `preds/wavs`, then run `test.py`.  It should output predictions in the form we store in the database by calling `full_inference.py`, including speaker verification output from a hardcoded profile id in `test.py` (in production, this is pulled from the database).
```
rm -rf preds/wavs
mkdir -p preds/wavs
cp -R test_data/wavs/* preds/wavs
python3 test.py
```

As a note: there are two cases where the program will intentionally throw a `ValueError`: (1) when there are no wavs in preds/wavs, (2) when there is no speech recognized in an entire wav.  This is by design, so that the handling function will know not to add anything to the database.

## Documentation & Important Files
If you are planning to read the code, I would start with `main.py`.  That is where all functions are called from, and it is well commented.  Other files are listed below.

* `main.py`: This is the main file, handling the data processing, and training the models.
* `full_inference.py`: This runs full inference, including speaker verification.
* `test.py`: This is a top level example calling `full_inference` within a try/except block to account for ValueErrors.  This is a simple version of the code Owen will use to call `full_inference` on the back end.
* `models.py`: Models are defined here.
* `utils.py`: Utility functions.  You should not have to read this unless you are curious about how something works.
* `speaker_verification.py`: Where speaker verification functions come from, used for identification and enrollment in the pipeline
* `ds_utils.py`: Deepspeech utilities used for VAD segmentation
* `interpret.py, generate.py, status.py`: Utilities for grid search using Standard-Grid.
* `transcribe.py`: Azure transcription
* `util_function.py, hffn.py`: Supporting functions for HFFN (deprecated).
