# Multimodal Fusion
This package offers unimodal and multimodal approaches to within-utterance and cross-utterance classification problems. In essence, this module takes in a transcripts file and a path to a directory containing wavs, generates MFB's from those wavs, aligns the MFB's with the transcripts, extracts word embeddings using BERT, packages everything together in a single `tensors` object (which is saved to disk in `--tensors_path`), then trains and tests on this object according to some parameters you pass in.  In this README, I first describe the requirements, then the data format for how to add new datasets, and the parameters required to run the program.

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
    "cognitive_key": "temp",
    "service_region": "temp"
}
```



### Training Data Format
The three main inputs to our training pipeline are the transcripts, the wav directory, and the labels.

#### Transcripts
The transcripts must be a dictionary, saved in `--transcripts_path` as a pickle file and formatted as such:
```python
{
    'wav_id': {
        'features': [['word1'], ['word2']..., ['wordn']], # np array of shape (n,1) for n words, stored in utf8 encoding (standard strings) or binary encoding
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
`--tensors_path`: if `unique`, the program will recreate tensors (by aligning and creating embeddings) each time the program is run in training mode (the default, or specified with `--mode train`). If a tensors path is specified and the tensor exists, the program will use those tensors and skip data preprocessing.  This is a useful feature because if, for example, you'd like to extract features & embeddings on one machine, then train on another (e.g. extract MFB's and BERT embeddings on ARMIS, then port over a tensor of embeddings to train on a lab machine), you can run the program in training mode with `--tensors_path my_tensors.pk`, end the program when it begins training (or add an `exit()` command at the end of `load_data()` after it saves the tensors), and port the tensors over to the lab machine.  Then, if you specify `--tensors_path my_tensors.pk` and it exists, you can skip data preprocessing during training in the future.
`--audio_path, --overwrite_mfbs`: the path to the MFB's.  If this exists and `--overwrite_mfbs` is 0, the program will not generate new MFB's from `--wav_dir`, else it will.
`--seq_len`: the legnth of the sequence you'd like to wrap to.  When the text modality is present, this is the number of words you'd like to wrap to.  For example, if the `seq_len` is 10, and an utterance has the words "one two three...eleven", the last word would be chopped off.  If an utterance had the words "one, two, three", there would be zero padding for the last seven slots. If just the audio modality, this is the number of MFB frames you'd like to take.  This number is usually much higher.
`--keys_path`: Here you can specify keys for the program to train / validate / test against (in case you want to do leave-one-speaker-out validation, or something similar).  See the form in `data/iemocap/utt_keys.json` for details.


### Training
--model_path val_model

### Inference