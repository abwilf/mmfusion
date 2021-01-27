from utils import *
import librosa
import soundfile as sf
from python_speech_features import fbank
import scipy.io.wavfile as wav
import subprocess

n_mels = 40
n_fft = 2048
hop_length = 160
fmin = 0
fmax = None
SR = 16000
MFB_WIN_STEP = 0.01
n_iter = 32

def resampleAudioFile(inFile, outFile, outFs):
    cmdParts = ['sox', inFile, '-r', str(outFs), outFile]
    subprocess.call(cmdParts, stdout=open(os.devnull), stderr=open(os.devnull))
    
def normAudioAmplitude(sig):
    return sig.astype(np.float)/-np.iinfo(np.int16).min

def extractMfbs(x, fs):
    # normalize
    # x = normAudioAmplitude(x)
    
        # Constants
    MFB_DIM = 40
    MFB_WIN_LEN = 0.025
    MFB_NFFT = 2048
    MFB_LOW_FREQ = 0
    MFB_HIGH_FREQ = None
    MFB_PREEMPH = 0.97

    # Extract
    x = fbank(x, samplerate=fs, winlen=MFB_WIN_LEN,
                     winstep=MFB_WIN_STEP, nfilt=MFB_DIM, nfft=MFB_NFFT,
                     lowfreq=MFB_LOW_FREQ, highfreq=MFB_HIGH_FREQ,
                     preemph=MFB_PREEMPH, winfunc=np.hanning)
    x = np.log(x[0] + 1e-8)
    
    # z-Normalize utterance
    stdVal = np.std(x)
    if stdVal == 0:
       stdVal = 1
    x = (x-np.mean(x))/stdVal
    
    # Clamp to 3 std
    clampVal = 3.0
    x[x>clampVal] = clampVal
    x[x<-clampVal] = -clampVal
    
    return x    

def get_mfbs(audio_path, sample_rate=None):
    if sample_rate is None: # leave unchanged
        tmpPath = audio_path
        rate, sig = wav.read(tmpPath)
    else:
        tmpPath = join(audio_path, '.tmp')
        resampleAudioFile(audio_path, tmpPath, sample_rate)
        rm_file(tmpPath)
        rate, sig = wav.read(tmpPath)

    mfbs = extractMfbs(sig, rate)
    return mfbs

def new_get_mfbs(wav_file):
    y, sr = librosa.load(wav_file, sr=SR)
    y = librosa.effects.preemphasis(y, coef=0.97)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax, htk=False)

    mel_spec, mean, std = z_norm(np.log(mel_spec + 1e-6))
    clampVal = 3.0
    mel_spec[mel_spec>clampVal] = clampVal
    mel_spec[mel_spec<-clampVal] = -clampVal
    return mel_spec.T, mean, std
    
def write_recon(out_path, mfb, mean, std):
    mfb = np.e**(un_z_norm(mfb.T, mean, std))

    melspec_recon = librosa.feature.inverse.mel_to_audio(mfb, sr=SR, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter, win_length=None, window='hann', center=True, power=2.0, length=None)
    sf.write(out_path, melspec_recon, SR)

def get_mfb_intervals(end, step):
    end = trunc(end*100, decs=2)
    step = trunc(step*100, decs=2)

    a = np.arange(0, end, step)

    a = trunc(a / 100, decs=2)
    end = trunc(end/100, decs=2)
    step = trunc(step/100, decs=2)


    b = np.concatenate([a[1:], [a[-1] + step]], axis=0)
    return np.vstack([a,b]).T