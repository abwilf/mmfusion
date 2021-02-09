from utils import *
from consts import *
import azure.cognitiveservices.speech as speechsdk

cognitive_key, service_region = LD(load_json(join(BASE_PATH, 'azure_secrets.json')))[['cognitive_key', 'service_region']]
speech_config = speechsdk.SpeechConfig(subscription=cognitive_key, region=service_region)
speech_config.set_service_property(name='format', value='detailed', channel=speechsdk.ServicePropertyChannel.UriQueryParameter)
speech_config.request_word_level_timestamps()

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
    
    return np.expand_dims(features, axis=-1), intervals, confidence

# # path = 'test_wavs/test.wav'
# path='/z/abwilf/mmfusion2/temp_wavs/test__-__2.wav'
# print(get_transcript(path))
# # save_pk('/z/abwilf/mmfusion/blah.pk', get_transcript(path))
