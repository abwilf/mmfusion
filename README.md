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
Models are automatically saved to `BASE_PATH/models` (`BASE_PATH` from `consts.py`), but you can specify this path with the `--model_path` flag. Once a model is saved, we can use it to generate predictions by feeding in data in the correct format.

The required format for **label** files is:
```
{'<vid_key>':
    'features': dummy array (doesn't matter what's in here) of size (num_utterances,1),
    'intervals': array of floats of shape (num_utterances,2) with start,end for each utterance.  this is essential to split and align the data across modalities
}
```

To see an example, pop open the IEMOCAP emotion labels file.
```
python3
from utils import *
a = load_pk('data/iemocap/IEMOCAP_EmotionLabels.pk')
dict_at(a)
```

The required format for **transcript** files is
```
{'<vid_key>':
    'features': array of words in the video of shape (num_words,),
    'intervals': array of floats of shape (num_words,2) with start,end for each word
}
```
For example, IEMOCAP:
```
from utils import *
a = load_pk('data/iemocap/IEMOCAP_TimestampedWords.pk')
dict_at(a)
```

The required format for **audio** is a wav directory where the same video keys from the labels are the names of the wavs. e.g.: data/iemocap/wavs

We'll first train a model, which is saved by default in `models/` (you can modify this with the `--model_path` flag).
```
python3 main.py --mode train --modality audio,text --cross_utterance 0 --tensors_path tensors/tensors_inf.pk --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --audio_path data/iemocap/mfb.pk --trials 1
```

To run inference from some `labels_path`, `transcripts_path`, and/or `audio_path`, set the `--mode inference` flag and the output will be sent to `inference.pk`.  For example:

```
python3 main.py --mode inference --modality audio,text --cross_utterance 0 --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --audio_path data/iemocap/mfb.pk --trials 1
```

As a sanity check on whether the model is performing as we think it should, we can add `--evaluate_inference 1`, which treats the labels passed in as real (not dummy) and evaluates the saved model on those labels.
```
python3 main.py --mode inference --modality audio,text --cross_utterance 0 --labels_path data/iemocap/IEMOCAP_EmotionLabels.pk --transcripts_path data/iemocap/IEMOCAP_TimestampedWords.pk --audio_path data/iemocap/mfb.pk --trials 1 --evaluate_inference 1
```

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

### Notes
* **When moving to another dataset with different labels than IEMOCAP**, you'll need to modify `label_map_fn` which maps labels from their original form (e.g., [`hap`, `neu`, `ang`, `sad`]) to indeces we can use for classification (e.g., [0,1,2,3]).  You'll also need to modify `num_labels`, which defines the dimensionality of the output classifiers (i.e., the number of nodes in the last dense layer). I would recommend opening a debugging session or using print statements to see the exact form of your data, and how you'll need to change it.
    
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

## Roadmap (to be implemented)
* HFFN inference
* FMT support for within utterance classification
* MOSEI links

## Credits
Made for and with support of the CHAI Lab at the University of Michigan, advised by Professor Emily Mower Provost. The HFFN code (`hffn.py` and `util_function.py`) came from Mai et al., the [MMSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) from Zadeh et al.