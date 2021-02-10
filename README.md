# Multimodal Fusion
This package offers unimodal and multimodal approaches to within-utterance and cross-utterance classification problems.

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
BASE_PATH = '/z/abwilf/mmfusion2' # the path to this directory
```

Create and save a file called `azure_secrets.json`, with the following code inside.  If you decide to use azure for speech recognition in the future, this is where your API information will go.
```
{
    "cognitive_key": "temp",
    "service_region": "temp"
}
```

### Unimodal Lexical Classifier
Let's start off by getting a simple **unimodal lexical classifier** working on the IEMOCAP dataset.

```
python3 main.py --modality text --tensors_path tensors/tensors.pk --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --trials 1
```

You'll notice that a lot of the time was spent loading and aligning the dataset.  If you rerun the command, you'll see that we skip this step because we've cached our tensors in tensors_path.  If you don't want this to happen, use `--overwrite_tensors 1`.  If you're running a bunch of tests that involve data preprocessing and don't want to have to worry about renaming tensors each time (or having them overlap across tests), set `--tensors_path unique` and the program will choose a random hash as the tensors name.  This will be useful for us later on.

Running the program this way doesn't give you control over which portion of the dataset you train vs test on.  To change that, simply pass in the path to a json file containing an object of the form:
```
{
    'train_keys': [list of train keys],
    'test_keys': [list of test keys],
}
```

For example on IEMOCAP:
```
python3 main.py --modality text --tensors_path unique --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --trials 1 --keys_path data/iemocap/utt_keys.json
```

### Unimodal Audio Classifier
Before we get started with our classifier, we'll download the data.
```
cd data/iemocap
gdown https://drive.google.com/uc?id=1eGj8DSau66NiklH30UIGab55cUWR_qw9
tar -xvf ie_wavs.tar
cd ../..
```

Now we can run our unimodal acoustic classifier. 
```
python3 main.py --modality audio --tensors_path unique --audio_path data/iemocap/mfb.pk --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --trials 1
```

First, you'll see that we map our wav files to mfbs.  Then, we align the dataset with the labels and begin classification.  We cache mfbs, but if you'd like to overwrite them, just set `--overwrite_mfbs 1`.

### Cross Utterance Classification
So far, we've classified each utterance independently from neighboring utterances.  However, utterances take place within the context of a larger conversation.  We'll leverage that here with a **cross utterance unimodal lexical classifier**.

```
python3 main.py --modality text --cross_utterance 1 --tensors_path unique --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --trials 1
```

We can specify train/test keys the same way, but it is important to note that **the keys must now be video ids instead of utterance ids**. See `data/iemocap/vid_keys.json` vs `data/iemocap/utt_keys.json` for an example of the difference. A note on terminology: we use the term "video" to describe a temporally ordered grouping of utterances - the visual modality does not need to be present.
```
python3 main.py --modality text --cross_utterance 1 --tensors_path unique --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --trials 1 --keys_path data/iemocap/vid_keys.json
```

### Multimodal Classification
So far, we've only performed unimodal classification on audio xor text.  Now, we will combine them.

To perform **within-utterance multimodal classification**, we'll modify `--modality` to contain `audio,text` (order doesn't matter) while keeping `--cross_utterance 0`.
```
python3 main.py --modality audio,text --cross_utterance 0 --tensors_path unique --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --audio_path data/iemocap/mfb.pk --trials 1
```

We can get significant performance improvements by using HFFN to implement **cross-utterance multimodal classification**.
```
python3 main.py --modality audio,text --cross_utterance 1 --tensors_path unique --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --audio_path data/iemocap/mfb.pk --trials 1
```

### Inference
As you train your models, they are automatically saved to `BASE_PATH/models` (`BASE_PATH` from `consts.py`). You can specify this path with the `--model_path` flag (or, in the case of cross utterance multimodal training with HFFN, with the `--hffn_path` flag, which requires that unimodal and multimodal models be created). Once a model is saved, we can use it to generate predictions by feeding in data in the correct format

Let's first train a within utterance unimodal text model on IEMOCAP, saving it to `models/model`
```
python3 main.py --modality text --tensors_path unique --mode train --cross_utterance 0
```

Now, we will run inference on a wav file, `test2.wav`.  `--mode inference` will automatically transcribe the wavs in `--wav_dir` if they do not yet exist in `--transcripts_path` using the azure cognitive services credentials `azure_secrets.json`. 
```
python3 main.py --modality text --mode inference --cross_utterance 0 --print_transcripts 1 --wav_dir demo_data/wavs --transcripts_path demo_data/transcripts.pk
```

The transcripts are available in the pickle file:
```
python3
from utils import *
load_pk('demo_data/transcripts.pk')
```

You can run this with different `--modality` and `--cross_utterance` flags as well, provided you have previously trained a model equipped to handle the type of modality and within/cross utterance type and that model exists in `--model_path` (or `--hffn_path` if cross utterance multimodal).  Your results will be in `output/inference.pk`.


### Grid Searches
Grid searching over hyperparameters can be essential to maximizing performance.  With Amir Zadeh's [Standard-Grid](https://github.com/A2Zadeh/Standard-Grid) this becomes easy.  We'll be using my fork of the project to access a few nice features.

To demonstrate these capabilities, we'll grid search over different approaches and modalities. In `generate.py`, you'll see our grid:
```python
'cross_utterance': [0,1],
'modality': ['text', 'audio', 'text,audio'],
'tensors_path': ['unique'],
```

To get a search started, simply type `python3 generate.py`, then copy the resulting output into the shell.
```bash
hash='06819'
p start_time.py $hash; root=$(pwd); attempt='0'; cd $root/results/${hash}/central/attempt_${attempt}/; chmod +x main.sh; ./main.sh; cd $root; p status.py ${hash}; p interpret.py ${hash};
```

When your test has finished, you can see your results in `results/{hash}/csv_results.csv`. Here's a short script to visualize the results of our grid search together.
```
python3 vis_results.py
```

```
Results of grid search:
      acc  cross_utterance    modality
1  0.4873                0       audio
4  0.5570                0        text
3  0.6618                0  text,audio
2  0.4566                1       audio
0  0.6885                1        text
```

### Other Datasets
To train / test models on **MOSEI**, first download the data
```
cd data
gdown https://drive.google.com/uc?id=12mJyK7w9NLJ88q7FHy2vtkT4pKqnzL5S
tar -xvf mosei.tar
rm mosei.tar
cd ..
```

Train a model
```
python3 main.py --modality text --transcripts_path data/mosei/CMU_MOSEI_TimestampedWords.csd --tensors_path tensors/mosei_text.pk --labels_path data/mosei/CMU_MOSEI_All_Labels.csd --trials 1 --mode train
```

Realistically, you would probably want to run inference on an unseen test set, but if you'd like, you can run inference on the whole dataset like this.
```
python3 main.py --modality text --transcripts_path data/mosei/CMU_MOSEI_TimestampedWords.csd --labels_path data/mosei/CMU_MOSEI_All_Labels.csd --mode inference
```

Since we are passing in real labels, we can set the `--evaluate_inference` flag and see how we would perform on the full dataset.  Again, this is not a realistic test, because we trained on a significant part of this dataset.  This is just to demonstrate how you would run inference on a dataset you choose to pass in.
```
python3 main.py --modality text --transcripts_path data/mosei/CMU_MOSEI_TimestampedWords.csd --labels_path data/mosei/CMU_MOSEI_All_Labels.csd --mode inference --evaluate_inference 1
```

This model only includes the text modality, as the audio files for MOSEI are quite large (14G).  If you would like those, you can download them here.

```
cd data/mosei
gdown https://drive.google.com/uc?id=1gslTSa9O9UbACuPWc-lAE770GyxsjI-c
tar -xvf mosei_wavs.tar
rm mosei_wavs.tar
cd ../..
```

### Notes
* **When moving to another dataset with different labels than IEMOCAP**, you'll need to modify `label_map_fn` in `main.py` which maps labels from their original form (e.g., [`hap`, `neu`, `ang`, `sad`]) to indeces we can use for classification (e.g., [0,1,2,3]).  You'll also need to modify `num_labels`, which defines the dimensionality of the output classifiers (i.e., the number of nodes in the last dense layer). I would recommend opening a debugging session or using print statements to see the exact form of your data, and how you'll need to change it.
    
Here is the definition for `label_map_fn`:
```python
def label_map_fn(labels):
    '''
    input: a one-dimensional array of labels (e.g., shape (10,...))
    output: a one-dimensional array of labels as integers
    '''
```

If you would like to use vscode to debug, here is a `launch.json` file you can use to start the debugger. If you haven't done this before, you'll need to select your python interpreter as the one tied to the `conda` environment you're using.
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": ["--cross_utterance=0", "--modality=text", "--tensors_path=unique", "--seq_len=50", "--keys_path=data/iemocap/utt_keys.json"]
        }
    ]
}
```

* The `--seq_len` flag controls the max sequence length during alignment.  If `text` is one of the modalities, `seq_len` is the max number of words per utterance (wrapped if more, padded if less). If only audio, this controls the max number of mfbs to allow per utterance.  When involving text, `seq_len=50` is usually sufficient, but only audio should be more like `250`, as mfbs are sampled every .1 seconds.
* When grid searching, make sure to use tensors_path=unique so the different runs don't alter each other's data.
* Transcripts files **must** end in `.pk`.
* For a full list of arguments accepted by `main.py`, run `python3 main.py --help` or look at the bottom of the file where the arguments and their descriptions are defined.
* In the transcripts and labels files, `intervals` are measured in seconds
* To change how MFBs are extracted, see the `mfb_util.py` file and modify the global variables there.
* When grid searching (especially on audio classification tasks), watch out for the number of parallel threads you run per GPU.  The tensors are very large, and you may encounter OOM errors.  If this happens, simply set `num_parallel` in `generate.py` to 1. For details on this behavior, see the [Standard-Grid](https://github.com/abwilf/standard-grid) repository.

## Roadmap (to be implemented)
* HFFN inference
* FMT support for within utterance classification
* MOSEI links

## Credits
Made for and with support of the CHAI Lab at the University of Michigan, advised by Professor Emily Mower Provost. The HFFN code (`hffn.py` and `util_function.py`) came from Mai et al., the [MMSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) from Zadeh et al.