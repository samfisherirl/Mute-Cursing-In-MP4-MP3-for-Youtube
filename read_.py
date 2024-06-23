from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import write


def read_audio_file(file_path):
    """
    Reads an audio file (.wav or .mp3) and returns it in a mono format,
    normalized between -1.0 and 1.0.
    """
    if file_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(file_path)
    elif file_path.endswith(".wav"):
        audio = AudioSegment.from_wav(file_path)
    else:
        raise Exception("Unsupported file format. Please use WAV or MP3.")
    audio_mono = audio.set_channels(1)
    samples = np.array(audio_mono.get_array_of_samples(), dtype=np.float32) / 32768.0
    return samples, audio_mono.frame_rate


def audio_to_numpy(audio_mono):
    """
    Converts mono audio data to a numpy array, normalized between -1.0 and 1.0.
    """
    samples = np.array(audio_mono.get_array_of_samples())
    # Normalize float32 array to range -1.0 to 1.0
    return samples.astype(np.float32) / (2**15)


def numpy_to_wav(np_array, sample_rate, file_name):
    """
    Correctly converts a numpy array (expected to be in float32 format) to a .wav file with int16 format,
    ensuring proper normalization to prevent clipping and maintain audio quality.
    """
    # First step is to find the maximum absolute value to use for normalization
    max_val = np.abs(np_array).max()
    if max_val > 0:  # Prevent division by zero
        # Normalize the array to -1.0 to 1.0 if not already
        normalized_array = np_array / max_val
    else:
        # In case the array is silent (all zeros), no normalization is needed
        normalized_array = np_array

    # Now, convert normalized array to int16. This scales the -1.0 to 1.0 range to -32767 to 32767.
    int_samples = np.int16(normalized_array * 32767)

    # Write the int16 samples to a WAV file
    write(file_name, sample_rate, int_samples)


class NumpyMono:
    def __init__(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.np_array, self.sample_rate = read_audio_file(audio_file_path)
        self.output_file_name = audio_file_path + "_clean_.wav"

    def numpy_to_wav(self):
        # Exporting numpy array to a .wav file
        numpy_to_wav(
            self.np_array, self.sample_rate, self.output_file_name)
        print('File saved as:', self.output_file_name)


if __name__ == "__main__":
    audio_file_path = 'path/to/your/audiofile.mp3'  # or .wav
    output_file_name = 'output_filename'
    NumpyMono(audio_file_path, output_file_name)


"""def read_audio_file(file_path):
Reads an audio file(.wav or .mp3) and returns it in a mono format.
    if file_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(file_path)
    elif file_path.endswith(".wav"):
        audio = AudioSegment.from_wav(file_path)
    else:
        raise Exception("Unsupported file format. Please use WAV or MP3.")
    audio_mono = audio.set_channels(1)
    return audio_mono


def audio_to_numpy(audio_mono):
Converts mono audio data to a numpy array.
    samples = np.array(audio_mono.get_array_of_samples())
    return samples.astype(np.float32)


def numpy_to_wav(np_array, sample_rate, file_name):
Converts a numpy array to a .wav file.
    audio_bytes = np_array.astype(np.int16).tobytes()

    # Creating an audio segment with the raw audio data
    audio = AudioSegment(data=audio_bytes, sample_width=2,
                         frame_rate=sample_rate, channels=1)
    # Exporting the audio segment to a .wav file
    audio.export(file_name, format="wav")


class NumpyMono:
    def __init__(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.audio_mono = read_audio_file(audio_file_path)
        self.np_array = audio_to_numpy(self.audio_mono)
        # The sample_rate needs to be obtained from the original file to accurately convert back to .wav
        self.sample_rate = self.audio_mono.frame_rate
        self.output_file_name = audio_file_path + "_clean_.wav"

    def numpy_to_wav(self):
        # Exporting numpy array to a .wav file
        numpy_to_wav(
            self.np_array, self.sample_rate, self.output_file_name)"""
