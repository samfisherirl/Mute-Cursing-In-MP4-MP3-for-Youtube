from pathlib import Path
import json
from datetime import datetime

cwd = Path(__file__).parent

def make_dirs():
    """returns (transcript_folder, export_folder)"""
    return (Path(cwd / "tscript").mkdir(parents=True, exist_ok=True), Path(cwd / "exports").mkdir(parents=True, exist_ok=True))
    

def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript

# Function to recursively convert all strings in a JSON object to lowercase

def main_file_audio(wav_file_path):
    # Read the audio file
    audio_data, sample_rate = sf.read(wav_file_path, dtype='float32')

    # Ensure that the audio file is mono
    if audio_data.ndim > 1:
        # Average the channels if more than one channel (i.e., stereo)
        audio_data = np.mean(audio_data, axis=1)

    # 'soundfile' already reads into 'float32', and the data is typically normalized
    # If you need to ensure normalization between -1.0 and 1.0, uncomment the following lines:
    # peak = np.max(np.abs(audio_data))
    # if peak > 1:
    #     audio_data /= peak
    return audio_data, sample_rate


def to_lowercase(input):
    if isinstance(input, dict):
        return {k.lower().strip("',.\"-_/`"): to_lowercase(v) for k, v in input.items()}
    elif isinstance(input, list):
        return [to_lowercase(element) for element in input]
    elif isinstance(input, str):
        return input.lower().strip()
    else:
        return input


def process_json(infile):
    # Read the original JSON file
    with open(infile, 'r') as file:
        data = json.load(file)
    # Convert all strings to lowercase
    words = []
    words = [{'word': word['word'].strip("',.\"-_/`").lower(), 'start': word['start'], 'end': word['end']}
             for segment in data['segments'] for word in segment['words']]

        # Write the modified JSON to a new file
    with open(infile, 'w') as file:
        json.dump(words, file, indent=4)
    # Read the original JSON file
    return words


def read_curse_words_from_csv(CURSE_WORD_FILE):
    curse_words_list = []
    with open(CURSE_WORD_FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Assuming curse words are in column A
            curse_words_list.append(row[0])
    return curse_words_list
# Function to mute curse words in the audio

