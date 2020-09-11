import time
import numpy as np
from utils import *
from scipy.io import wavfile
from keras.models import load_model
import soundfile as sf
import sounddevice as sd

def main():
    chime, fs = sf.read("audios/chime.wav", dtype='float32')
    model = load_model("models/trigger word detection model.h5")
    ###create the pre interfence :-
    audio_arr = np.zeros((441000, ))
    audio_arr = np.int16(audio_arr / np.max(np.abs(audio_arr)) * 32767)
    spec_arr = rec_to_spec(audio_arr).swapaxes(0, 1)
    model.predict(np.expand_dims(spec_arr, axis = 0))
    while True:
        try:
            rate, new_aud_arr = wavfile.read("audios/current_recording.wav")
        except:
            time.sleep(0.05)
            continue
        new_spec_arr = rec_to_spec(new_aud_arr).swapaxes(0, 1)
        spec_arr = np.vstack([spec_arr, new_spec_arr])[-5511:, :]
        prediction = model.predict(np.expand_dims(spec_arr, axis = 0))
        if has_new_triggerword(np.squeeze(prediction, axis = 0), 2.5, 10, 0.5)==True:
            sd.play(chime, fs)
            time.sleep(2.6)
if __name__=="__main__":
    main()