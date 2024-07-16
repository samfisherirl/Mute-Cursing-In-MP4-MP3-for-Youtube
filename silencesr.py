from tkinter.messagebox import showinfo
from tkinter import Tk
import subprocess
import re
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

GAP = 80  # ms


def _msgbox(title="", message=""):
    """
    Displays a messagebox with the specified title and message.
    
    Args:
    - title: A string representing the title of the messagebox.
    - message: A string representing the message to be displayed.
    Example usage:
    # create_messagebox("Greeting", "Hello, World!")
    """
    root = Tk()
    root.withdraw()  # Hide the main Tk window
    showinfo(title, message)
    root.destroy()





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
    _msgbox(formats, format_name)
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


def process_pair(pair):
    current_end, next_start = pair
    # Assuming convert_to_ms function is defined elsewhere
    current_end_ms = convert_to_ms(current_end) + 20
    next_start_ms = convert_to_ms(next_start) - 20

    if next_start_ms - current_end_ms > GAP:
        return (current_end, next_start)
    return None


def find_silences(timestamps):
    silences = []

    # Preparing pairs of current end and next start timestamps
    pairs = [(timestamps[i][1], timestamps[i + 1][0])
             for i in range(len(timestamps) - 1)]

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_pair, pairs)

    # Filter out None results and append to silences
    silences.extend(filter(None, results))

    return silences


def convert_to_ms(time_str):
    """Converts time string to milliseconds."""
    hours, minutes, seconds = map(float, time_str.replace(',', ':').split(':'))
    return ((hours * 60 + minutes) * 60 + seconds) * 1000


def process_pair(pair):
    """Process a single pair to find silence."""
    current_end, next_start = pair
    current_end_ms = convert_to_ms(current_end) + 20
    next_start_ms = convert_to_ms(next_start) - 20

    if next_start_ms - current_end_ms > GAP:
        return (current_end, next_start)
    return None


def find_silences(timestamps):
    """Finds silences based on given timestamps."""
    pairs = [(timestamps[i][1], timestamps[i + 1][0])
             for i in range(len(timestamps) - 1)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_pair, pairs)

    # Filtering None values and return the list of silences
    return list(filter(None, results))


def parse_subtitles(subtitle_text):
    """Parse subtitles to find silences using timestamps."""
    # Find all timestamps in the subtitle text
    timestamps = re.findall(
        r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', subtitle_text)

    # Use the find_silences function to find silences
    silences = find_silences(timestamps)

    return silences

def convert_to_ms(timestamp):
    hours, minutes, seconds_milliseconds = re.split('[:]', timestamp)
    seconds, milliseconds = (seconds_milliseconds + ",0").split(',')[:2]
    
    return int(hours) * 3600000 + int(minutes) * 60000 + int(seconds) * 1000 + int(milliseconds)
    

def process_video(silences, input_video):
    cut_files = []
    silence_end_previous = "0"
    out = Path(input_video).parent / "output"
    out.mkdir(exist_ok=True)
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
    #  for file in video_files:
    #     os.remove(file)
    print(f'video_files: {video_files}')
    os.remove("file_list.txt")

def read_subtitles(inputfile):
    with open(inputfile, "r") as f:
        contents = f.read()
    return contents


def main():
    silences = parse_subtitles(
        read_subtitles(
            select()
            )
        )
    if not silences:
        print("No silences detected, processing complete.")
    else:
        process_video(silences, select("*.mp4", "mp4"))
        print("Video processing complete.")

if __name__ == "__main__":
    main()