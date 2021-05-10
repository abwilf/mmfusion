from full_inference import full_inference

speaker_profile='7411f391-e8f1-4bf8-acd4-f11dafde406d'
try:
    res = full_inference(speaker_profile)
    print(res)
except ValueError:
    print('No result!  Either no speech recognized or no wavs in preds/wavs')