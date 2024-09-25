import numpy as np
from scipy.io import wavfile
from scipy.signal import windows


def read_modify_write_wav(input_filename, output_filename):
    sample_rate, audio_data = wavfile.read(input_filename)

    # Calculate number of samples for 0.2 second silence
    silence_samples = int(0.2 * sample_rate)
    silence_data = np.zeros(silence_samples, dtype=audio_data.dtype)

    # Calculate positions for silences
    first_silence_start = 5 * sample_rate
    second_silence_start = 10 * sample_rate

    # Insert silences
    audio_data = np.insert(audio_data, first_silence_start, silence_data)
    audio_data = np.insert(audio_data, second_silence_start +
                           silence_samples, silence_data)  # Adjust for inserted silence

    # Fade function (1 second fade-in and fade-out)
    def apply_fade(audio, sample_rate, start_pos, end_pos, fade_length=1):
        fade_in_samples = int(fade_length * sample_rate)

        # Create fade in window (first half of a Hann window)
        fade_in_window = windows.hann(fade_in_samples * 2)[:fade_in_samples]

        # Apply fade in
        audio[start_pos:start_pos + fade_in_samples] *= fade_in_window

        # Create fade out window (second half of a Hann window)
        fade_out_window = windows.hann(fade_in_samples * 2)[fade_in_samples:]

        # Apply fade out
        audio[end_pos - fade_in_samples:end_pos] *= fade_out_window

    # Apply fade in and fade out around silences
    apply_fade(audio_data, sample_rate, first_silence_start -
               silence_samples, first_silence_start)
    apply_fade(audio_data, sample_rate, second_silence_start,
               second_silence_start + silence_samples)

    wavfile.write(output_filename, sample_rate, audio_data)


# Example usage:
input_filename = 'Sequence 01(1).wav'
output_filename = 'your_output.wav'
read_modify_write_wav(input_filename, output_filename)
