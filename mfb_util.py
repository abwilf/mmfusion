from utils import *
import librosa
import soundfile as sf
import scipy.io.wavfile as wav
import subprocess

n_mels = 40
n_fft = 2048
hop_length = 160 # mfbs are extracted in intervals of .1 second
fmin = 0
fmax = None
SR = 16000
n_iter = 32
MFB_WIN_STEP = .01

def resampleAudioFile(inFile, outFile, outFs):
    cmdParts = ['sox', inFile, '-r', str(outFs), outFile]
    subprocess.call(cmdParts, stdout=open(os.devnull), stderr=open(os.devnull))
    
def normAudioAmplitude(sig):
    return sig.astype(np.float)/-np.iinfo(np.int16).min

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