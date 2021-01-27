# HFFN
An end to end Multimodal Sentiment Analysis system taking [MOSEI](https://www.aclweb.org/anthology/P18-1208/) data from the [SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK), extracting utterance level features by modality and adding cross utterance context as in [Poria et al (2017)](https://www.aclweb.org/anthology/P17-1081.pdf), and running the output through [HFFN](https://www.aclweb.org/anthology/P19-1046.pdf).

## Requirements

## Usage
1. Install `python3.7`
2. Create two virtual environments (conda works fine), `one` with `tensorflow-gpu==2.0.0`, `two` with `tensorflow==2.2.0`
3. Call preprocess, e.g.
```
conda create -n one python=3.7
conda activate one
pip install numpy tensorflow==2.2.0 argparse pandas sklearn tqdm twilio
```
conda activate md
python3 preprocess.py --data mosei --classes 2 --idx 1 --base_path /z/abwilf/hffn/
```
4. Call hffn, e.g.
```
conda activate autoencoder
python3 hffn.py --data mosei --classes 2 --idx 1
```

I built a simple texting utility to send me updates.  If you don't want to include this, coment out all lines containing `from ModelTexter`, `t = Texter()` and `t.send()`.  If you want to include this, create a json file of the following form with information from your twilio account.
```
{
    "ACCOUNT_SID": "<...>",
    "AUTH_TOKEN": "<...>",
    "TWILIO_PHONE": "<usually +1><...>",
    "MY_PHONE": "<usually +1><...>"
}
```

## Credits
The HFFN code (`mosi_acl.py` and `util_function.py`)from Sijie Mai, the [MOSEI SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK), and the BC-LSTM from [Poria et al](https://github.com/soujanyaporia/multimodal-sentiment-analysis).




Download iemocap wavs
```
pip install gdown
cd data/iemocap
gdown https://drive.google.com/uc?id=1eGj8DSau66NiklH30UIGab55cUWR_qw9
tar -xvf ie_wavs.tar
```



CUDA Version 10.1.243
CUDNN 7.6


