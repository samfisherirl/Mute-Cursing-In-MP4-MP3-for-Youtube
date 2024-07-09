import subprocess
import re
import os
import tkinter as tk
from tkinter import filedialog

GAP = 300 # ms

def final_segment_non_silence(input_video, last_silence_end, cut_files):
    final_segment = "final_segment.mp4"
    # Use ffmpeg to get to the end without specifying duration
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-ss", last_silence_end.replace(',', '.'),
        "-c", "copy",
        final_segment
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if "Conversion failed" not in result.stderr:
        cut_files.append(final_segment)

        
def select(formats="*.srt", format_name="SRT files"):
    root = tk.Tk()
    root.withdraw()  # Hide main window
    root.call('wm', 'attributes', '.', '-topmost', '1')

    # File dialog to select an audio or video file
    srt_path = filedialog.askopenfilename(
        title='Select SRT files',
        filetypes=[(formats, format_name)]
    )
    root.destroy()
    return srt_path


# Parse subtitles to find silences
def parse_subtitles(subtitle_text):
    timestamps = re.findall(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', subtitle_text)
    silences = []

    for i in range(len(timestamps) - 1):
        current_end = timestamps[i][1]
        next_start = timestamps[i + 1][0]

        # Convert times to milliseconds
        current_end_ms = convert_to_ms(current_end)
        next_start_ms = convert_to_ms(next_start)

        # Check if silence is more than 300ms
        if next_start_ms - current_end_ms > GAP:
            silences.append((current_end, timestamps[i + 1][0]))

    return silences

def convert_to_ms(timestamp):
    hours, minutes, seconds_milliseconds = re.split('[:]', timestamp)
    seconds, milliseconds = (seconds_milliseconds + ",0").split(',')[:2]
    return int(hours) * 3600000 + int(minutes) * 60000 + int(seconds) * 1000 + int(milliseconds)

def process_video(silences, input_video):
    cut_files = []
    silence_end_previous = "0"

    for i, (start_cut, end_cut) in enumerate(silences, start=1):
        output_file = f"segment_{i}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-ss", silence_end_previous.replace(',', '.'),
            "-to", start_cut.replace(',', '.'),
            "-c", "copy",
            output_file
        ]
        subprocess.run(cmd, check=True)
        silence_end_previous = end_cut
        cut_files.append(output_file)
    
    # Always attempt to include a final segment if it exists
    final_segment_non_silence(input_video, silence_end_previous.replace(',', '.'), cut_files)
    
    combine_videos(cut_files, "final_output.mp4")

def combine_videos(video_files, output_file):
    with open('file_list.txt', 'w') as f:
        for file in video_files:
            f.write(f"file '{file}'\n")
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "file_list.txt",
        "-c", "copy",
        output_file
    ]
    subprocess.run(cmd, check=True)

    # Cleanup
    for file in video_files:
        os.remove(file)
    os.remove("file_list.txt")

def read_subtitles(inputfile):
    with open(inputfile, "r") as f:
        contents = f.read()
    return contents


def main():
    silences = parse_subtitles(read_subtitles(select()))
    if not silences:
        print("No silences detected, processing complete.")
    else:
        process_video(silences, select("*.mp4", "mp4"))
        print("Video processing complete.")

if __name__ == "__main__":
    main()