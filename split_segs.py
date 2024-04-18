import subprocess
import os
from datetime import datetime

# Assuming segment_duration is defined globally
segment_duration = 1800  # Example: 1800 seconds (30 minutes)


def split_audio(audio_file, output_dir, segment_duration=120):
    """
    Splits an audio file into segments of a specified duration using ffmpeg,
    and saves them in the provided output directory. Returns a list of paths
    to the generated segments.
    
    Args:
        audio_file (str): Path to the input audio file.
        output_dir (str): Path to the directory where the segments will be saved.
        segment_duration (int): Duration of each audio segment in seconds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_pattern = os.path.join(output_dir, f"segment_{timestamp}_%03d.wav")

    cmd = [
        'ffmpeg',
        '-i', audio_file,
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-c', 'copy',
        '-vn',  # Exclude video
        output_pattern
    ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
        print(f"Audio has been successfully split and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to split audio: {e.stderr}")
        return []

    # Generate the list of file paths for the new audio segments
    segment_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                     if f.startswith(f"segment_{timestamp}") and f.endswith('.wav')]
    return segment_files
    
if __name__ == "__main__":
    # Example usage
    input_audio = "C:/Users/dower/Videos/Why Context Matters When Bridges Burn..._1.mp3"
    output_dir = "C:\\Users\\dower\\Videos\\16-04-112949-Why Context Matters When Bridges Burn..._1"
    segment_duration = 1800  # For example, 1800 seconds (30 minutes)

    split_audio(input_audio, output_dir, segment_duration)
