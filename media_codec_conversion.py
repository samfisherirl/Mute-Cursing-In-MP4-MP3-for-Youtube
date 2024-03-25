import ffmpeg
import tkinter as tk
from tkinter import filedialog

# Function to get video file details


# Function to get video file details
def get_video_details(filepath):
    try:
        probe = ffmpeg.probe(filepath)

        # Assuming the first streams of video and audio are what we're matching
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)

        details = {
            'video_codec': video_stream['codec_name'],
            'resolution': f"{video_stream['width']}x{video_stream['height']}",
            # Convert string fraction to float
            'frame_rate': eval(video_stream['r_frame_rate']),
            'pix_fmt': video_stream.get('pix_fmt'),
            'audio_codec': audio_stream['codec_name'] if audio_stream else None,
            'audio_sample_rate': audio_stream['sample_rate'] if audio_stream else None,
            # Defaulting to 2 if not found
            'audio_channels': audio_stream['channels'] if audio_stream else '2',
        }
        return details
    except Exception as e:
        print(f"Error getting video details: {e}")
        return None

# Function to convert second video to match first video's details


def convert_video(input_file, output_file, video_details):
    try:
        conversion_cmd = (
            ffmpeg
            .input(input_file)
            .output(output_file,
                    **{'c:v': video_details['video_codec'],
                       'vf': f"scale={video_details['resolution']},format={video_details['pix_fmt']}",
                       'r': video_details['frame_rate'],
                       'c:a': video_details['audio_codec'],
                       'ar': video_details['audio_sample_rate'],
                       'ac': video_details['audio_channels']}
                    )
            # For experimental codecs, if necessary
            .global_args('-strict', '-2')
            .overwrite_output()
        )

        conversion_cmd.run()
    except Exception as e:
        print(f"Error converting video: {e}")


def main():
    root = tk.Tk()
    root.withdraw()  # Do not show the root window

    file_path_1 = filedialog.askopenfilename(title='Select the first video')
    file_path_2 = filedialog.askopenfilename(
        title='Select the second video to be converted')

    if file_path_1 and file_path_2:
        video1_details = get_video_details(file_path_1)
        if video1_details:
            # Convert the filename of the second video for output
            output_file = file_path_2.rsplit(
                '.', 1)[0] + '_converted.' + file_path_2.rsplit('.', 1)[1]
            convert_video(file_path_2, output_file, video1_details)
            print(f"Conversion completed. Output file: {output_file}\n{file_path_1}\n{file_path_2}")
        else:
            print("Error: Could not get details of the first video.")
    else:
        print("Operation cancelled or files not selected.")


if __name__ == "__main__":
    main()
