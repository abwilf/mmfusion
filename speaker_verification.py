import requests, json, librosa, argparse
import soundfile as sf

def lmap(fn, iterable):
    return list(map(fn, iterable))

def load_json(file_stub):
    filename = file_stub
    with open(filename) as json_file:
        return json.load(json_file)

def create_profile(api_key, region='eastus'):
    '''
    With curl:
            curl --location --request POST 'https://westus.api.cognitive.microsoft.com/speaker/identification/v2.0/text-independent/profiles/' \
            --header 'Ocp-Apim-Subscription-Key: 0c5427b4a05e4abfbc37c0dcdf649ff3' \
            --header 'Content-Type: application/json' \
            --data-raw '{
                '\''locale'\'':'\''en-us'\''
            }'
    '''
    res = requests.post(
        f'https://{region}.api.cognitive.microsoft.com/speaker/identification/v2.0/text-independent/profiles/',
        headers={
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': api_key,
        },
        data="{'locale': 'en-us'}"
    )
    obj = json.loads(res.text)

    assert 'error' not in obj, obj
    return obj['profileId']

def enroll_user(api_key, wav_path, profile_id, region='eastus'):
    '''
    With curl:
                curl --location --request POST 'https://westus.api.cognitive.microsoft.com/speaker/identification/v2.0/text-independent/profiles/a6c80127-cb3b-4c7e-8d24-a6d83b888f51/enrollments' \
            --header 'Ocp-Apim-Subscription-Key: 0c5427b4a05e4abfbc37c0dcdf649ff3' \
            --header 'Content-Type: application/json' \
            --data-binary '@/Users/abwilf/file.wav'
    '''

    resample_audio(wav_path, wav_path)

    with open(wav_path, 'rb') as f:
        data_binary = f.read()
    
    res = requests.post(
        f'https://{region}.api.cognitive.microsoft.com/speaker/identification/v2.0/text-independent/profiles/{profile_id}/enrollments/',
        headers={
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': api_key,
        },
        data=data_binary,
    )
    obj = json.loads(res.text)
    assert 'error' not in obj, obj
    return obj

def remove_user(api_key, profile_id, region='eastus'):
    res = requests.delete(
        f'https://{region}.api.cognitive.microsoft.com/speaker/identification/v2.0/text-independent/profiles/{profile_id}',
        headers={
            'Ocp-Apim-Subscription-Key': api_key,
        },
    )
    if res.text != '':
        assert False

def list_users(api_key, region='eastus'):
    res = requests.get(
        f'https://{region}.api.cognitive.microsoft.com/speaker/identification/v2.0/text-independent/profiles/',
        headers={
            'Ocp-Apim-Subscription-Key': api_key,
        },
    )
    obj = json.loads(res.text)
    assert 'error' not in obj, obj
    return lmap(lambda elt: elt['profileId'], obj['profiles'])

def identify_user(api_key, wav_path, profile_ids, region='eastus'):
    '''profile_ids can be a single profile_id or a list of profile ids'''
    resample_audio(wav_path, wav_path)

    if type(profile_ids) == list:
        profile_ids = ','.join(profile_ids)

    with open(wav_path, 'rb') as f:
        data_binary = f.read()
    
    res = requests.post(
        f'https://{region}.api.cognitive.microsoft.com/speaker/identification/v2.0/text-independent/profiles/identifySingleSpeaker?profileIds={profile_ids}',
        headers={
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': api_key,
        },
        data=data_binary,
    )
    obj = json.loads(res.text)
    print(obj)
    assert 'error' not in obj, obj
    return obj

def resample_audio(wav_path, temp_path):
    '''
    Makes wav conform with requirements of speaker verification API:
        * turns wav into mono
        * samples at 16khz
        * writes PCM_32 wav
    '''
    sr = 16000
    y, _ = librosa.load(wav_path, sr=sr)
    y = librosa.to_mono(y)
    sf.write(temp_path, y, sr, subtype='PCM_32')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wav_path', type=str, default='./enrollment.wav', help='The path to the enrollment wav file')
    parser.add_argument('--temp_path', type=str, default='./temp.wav', help='Internally used to hold the resampled file')
    parser.add_argument('--secrets_path', type=str, default='./secrets.json', help='Contains api_key for Azure speech service')
    args = parser.parse_args()

    resample_audio(args.wav_path, args.temp_path)

    api_key = load_json(args.secrets_path)['speaker_verification_key']
    region = 'eastus'

    profile_id = create_profile(api_key, region)

    print(f'Profile id is:\n{profile_id}')

    res = enroll_user(api_key, args.wav_path, profile_id)
    print(f'Enrollment successful!')


