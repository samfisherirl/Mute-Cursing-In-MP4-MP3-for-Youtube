import os
from pydub import AudioSegment
from datetime import timedelta
from pathlib import Path

def split_audio(audio_file, output_dir, segment_duration=30):
    """
    Split an audio file into segments of specified duration and save them in the output directory.

    Args:
        audio_file (str): Path to the input audio file (WAV or MP3)
        output_dir (str): Path to the output directory
        segment_duration (int, optional): Duration of each segment in minutes (default: 30)

    Returns:
        list: List of paths to the generated audio segments
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the audio file
    audio = AudioSegment.from_file(
        audio_file, format=audio_file.split(".")[-1])

    # Calculate the duration of the audio file in minutes
    total_duration = audio.duration_seconds / 60

    # Split the audio into segments
    segments = []
    fil_name = Path(audio_file).stem
    segment_start = 0
    segment_number = 1
    while segment_start < total_duration:
        segment_end = min(segment_start + segment_duration, total_duration)
        segment = audio[segment_start * 60000:(segment_end * 60000)]
        segment_filename = os.path.join(
            output_dir, f"{fil_name}_{segment_number}.{audio_file.split('.')[-1]}")
        segment.export(segment_filename, format=audio_file.split(".")[-1])
        segments.append(segment_filename)
        segment_start = segment_end
        segment_number += 1

    return segments


if __name__ == "__main__":
    audio_file = input("Enter the path to the audio file: ")
    output_dir = os.path.dirname(os.path.abspath(audio_file))

    segments = split_audio(audio_file, output_dir)
    print("Generated segments:")
    for segment in segments:
        print(segment)
