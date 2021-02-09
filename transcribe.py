import azure.cognitiveservices.speech as speechsdk
from utils import *

# create a cognitive services resouce on azure, select 
# cognitive_key = '1f57046b9db6453684916bd00fddd6ce'
# service_region = 'eastus'

# emily grant
cognitive_key = '3b02f4c5953546d6944a0eccaed9a4bf'
service_region = 'eastus'

# cognitive_key = '5100ec9ba72a498cbe77ff5ac9df4905'
# service_region = 'westus'

speech_config = speechsdk.SpeechConfig(subscription=cognitive_key, region=service_region)
speech_config.set_service_property(name='format', value='detailed', channel=speechsdk.ServicePropertyChannel.UriQueryParameter)
speech_config.request_word_level_timestamps()

# initial_silence_timeout = 6 # seconds
# end_silence_timeout = 6
# speech_config.set_service_property(name='initialSilenceTimeoutMs', value=str(initial_silence_timeout*1e3), channel=speechsdk.ServicePropertyChannel.UriQueryParameter)
# speech_config.set_service_property(name='endSilenceTimeoutMs', value=str(end_silence_timeout*1e3), channel=speechsdk.ServicePropertyChannel.UriQueryParameter)

def get_transcript(wav_path):
    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        confidence = json.loads(list(result.properties.values())[0])['NBest'][0]['Confidence']
        stt = json.loads(result.json)
        confidences_in_nbest = [item['Confidence'] for item in stt['NBest']]
        best_index = confidences_in_nbest.index(max(confidences_in_nbest))
        
        words = stt['NBest'][best_index]['Words']

        features, intervals = [], []
        for elt in words:
            features.append(elt['Word'])
            start = elt['Offset'] * 1e-7
            end = start + elt['Duration']*1e-7
            intervals.append([start, end])
        
        features, intervals = ar(features), ar(intervals)

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print('No speech could be recognized: {}'.format(result.no_match_details))
        features, intervals, confidence = ar([]), ar([]), 0

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print('Speech Recognition canceled: {}'.format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print('Error details: {}'.format(cancellation_details.error_details))
        assert False
    
    return features, intervals, confidence

path = 'blah3.wav'
print(get_transcript(path))
# save_pk('/z/abwilf/mmfusion/blah.pk', get_transcript(path))
