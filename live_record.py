import time
import pyaudio
import numpy as np
from scipy.io import wavfile

def record_main():
    p = pyaudio.PyAudio()
    CHANNELS = 1
    RATE = 44100


    def callback(in_data, frame_count, time_info, flag):
        global data, audio_data
        audio_data = np.fromstring(in_data, dtype=np.float32)
        data = np.append(data, audio_data)
        return (audio_data, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    output=False,
                    input=True,
                    stream_callback=callback)
    while True:
        global data
        data = np.array([])
        stream.start_stream()
        time.sleep(2.5)
        stream.stop_stream()
        data = np.int16(data / np.max(np.abs(data)) * 32767)
        wavfile.write("audios/current_recording.wav", 44100, data)

    stream.close()
    p.terminate()

if __name__=="__main__":
    record_main()



