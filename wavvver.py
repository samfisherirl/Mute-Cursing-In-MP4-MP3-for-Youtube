import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows
import soundfile as sf


def apply_fadein(audio, sr, duration=0.01):
    # convert to audio indices (samples)
    length = int(duration * sr)
    start = 0
    end = start + length

    # compute fade in curve
    # linear fade
    fade_curve = np.linspace(0.0, 1.0, length)

    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve

def apply_fadeout(audio, sr, duration=0.01):
    # convert to audio indices (samples)
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length

    # compute fade out curve
    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)

    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve


def apply_combined_fades(audio, sr, start_time, stop_time, fade_duration=0.01):
    # Convert times to samples
    fade_length = int(fade_duration * sr)
    start_sample = int(start_time * sr)
    stop_sample = int(stop_time * sr)

    # Apply fade out
    fade_out_end = start_sample + fade_length
    if fade_out_end > audio.shape[0]:
        fade_out_end = audio.shape[0]
    fade_out_curve = np.linspace(1.0, 0.0, fade_out_end - start_sample)
    audio[start_sample:fade_out_end] *= fade_out_curve
    # Apply fade in
    fade_in_start = stop_sample - fade_length
    if fade_in_start < 0:
        fade_in_start = 0
    fade_in_curve = np.linspace(0.0, 1.0, stop_sample - fade_in_start)
    audio[fade_in_start:stop_sample] *= fade_in_curve
    # Ensure silence between the fades
    audio[fade_out_end:fade_in_start] = 0


path = librosa.ex('brahms')
orig, sr = librosa.load('S.wav')
out = orig.copy()
apply_combined_fades(out, sr, 2, 3, 0.01)


sf.write('S.wav', orig, samplerate=sr)
sf.write('Sfaded.wav', out, samplerate=sr)
