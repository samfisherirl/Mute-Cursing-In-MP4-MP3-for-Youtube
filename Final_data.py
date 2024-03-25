import sys
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
import argparse
import numpy as np
import torch
from queue import Queue
import speech_recognition as sr
import pyaudio
import openai
from threading import Thread
import stable_whisper

# Ensure you securely manage your API key
openai.api_key = 'your-openai-api-key'


def warn_distutils_present():
    if 'distutils' not in sys.modules:
        return
    import warnings

    warnings.warn(
        "Distutils was imported before Setuptools, but importing Setuptools "
        "also replaces the `distutils` module in `sys.modules`. This may lead "
        "to undesirable behaviors or errors. To avoid these issues, avoid "
        "using distutils directly, ensure that setuptools is installed in the "
        "traditional way (e.g. not an editable install), and/or make sure "
        "that setuptools is always imported before distutils."
    )


def clear_distutils():
    if 'distutils' not in sys.modules:
        return
    import warnings

    warnings.warn("Setuptools is replacing distutils.")
    mods = [
        name
        for name in sys.modules
        if name == "distutils" or name.startswith("distutils.")
    ]
    for name in mods:
        del sys.modules[name]


def enabled():
    """
    Allow selection of distutils by environment variable.
    """
    which = os.environ.get('SETUPTOOLS_USE_DISTUTILS', 'local')
    return which == 'local'


def ensure_local_distutils():
    import importlib

    clear_distutils()

    # With the DistutilsMetaFinder in place,
    # perform an import to cause distutils to be
    # loaded from setuptools._distutils. Ref #2906.
    with shim():
        importlib.import_module('distutils')

    # check that submodules load as expected
    core = importlib.import_module('distutils.core')
    assert '_distutils' in core.__file__, core.__file__
    assert 'setuptools._distutils.log' not in sys.modules


def do_override():
    """
    Ensure that the local copy of distutils is preferred over stdlib.

    See https://github.com/pypa/setuptools/issues/417#issuecomment-392298401
    for more motivation.
    """
    if enabled():
        warn_distutils_present()
        ensure_local_distutils()


class _TrivialRe:
    def __init__(self, *patterns):
        self._patterns = patterns

    def match(self, string):
        return all(pat in string for pat in self._patterns)


class DistutilsMetaFinder:
    def find_spec(self, fullname, path, target=None):
        # optimization: only consider top level modules and those
        # found in the CPython test suite.
        if path is not None and not fullname.startswith('test.'):
            return None

        method_name = 'spec_for_{fullname}'.format(**locals())
        method = getattr(self, method_name, lambda: None)
        return method()

    def spec_for_distutils(self):
        if self.is_cpython():
            return None

        import importlib
        import importlib.abc
        import importlib.util

        try:
            mod = importlib.import_module('setuptools._distutils')
        except Exception:
            # There are a couple of cases where setuptools._distutils
            # may not be present:
            # - An older Setuptools without a local distutils is
            #   taking precedence. Ref #2957.
            # - Path manipulation during sitecustomize removes
            #   setuptools from the path but only after the hook
            #   has been loaded. Ref #2980.
            # In either case, fall back to stdlib behavior.
            return None

        class DistutilsLoader(importlib.abc.Loader):
            def create_module(self, spec):
                mod.__name__ = 'distutils'
                return mod

            def exec_module(self, module):
                pass

        return importlib.util.spec_from_loader(
            'distutils', DistutilsLoader(), origin=mod.__file__
        )

    @staticmethod
    def is_cpython():
        """
        Suppress supplying distutils for CPython (build and tests).
        Ref #2965 and #3007.
        """
        return os.path.isfile('pybuilddir.txt')

    def spec_for_pip(self):
        """
        Ensure stdlib distutils when running under pip.
        See pypa/pip#8761 for rationale.
        """
        if sys.version_info >= (3, 12) or self.pip_imported_during_build():
            return
        clear_distutils()
        self.spec_for_distutils = lambda: None

    @classmethod
    def pip_imported_during_build(cls):
        """
        Detect if pip is being imported in a build script. Ref #2355.
        """
        import traceback

        return any(
            cls.frame_file_is_setup(frame) for frame, line in traceback.walk_stack(None)
        )

    @staticmethod
    def frame_file_is_setup(frame):
        """
        Return True if the indicated frame suggests a setup.py file.
        """
        # some frames may not have __file__ (#2940)
        return frame.f_globals.get('__file__', '').endswith('setup.py')

    def spec_for_sensitive_tests(self):
        """
        Ensure stdlib distutils when running select tests under CPython.

        python/cpython#91169
        """
        clear_distutils()
        self.spec_for_distutils = lambda: None

    sensitive_tests = (
        [
            'test.test_distutils',
            'test.test_peg_generator',
            'test.test_importlib',
        ]
        if sys.version_info < (3, 10)
        else [
            'test.test_distutils',
        ]
    )


for name in DistutilsMetaFinder.sensitive_tests:
    setattr(
        DistutilsMetaFinder,
        f'spec_for_{name}',
        DistutilsMetaFinder.spec_for_sensitive_tests,
    )


DISTUTILS_FINDER = DistutilsMetaFinder()


def add_shim():
    DISTUTILS_FINDER in sys.meta_path or insert_shim()


class shim:
    def __enter__(self):
        insert_shim()

    def __exit__(self, exc, value, tb):
        _remove_shim()


def insert_shim():
    sys.meta_path.insert(0, DISTUTILS_FINDER)


def _remove_shim():
    try:
        sys.meta_path.remove(DISTUTILS_FINDER)
    except ValueError:
        pass


if sys.version_info < (3, 12):
    # DistutilsMetaFinder can only be disabled in Python < 3.12 (PEP 632)
    remove_shim = _remove_shim()


class WhisperAudioProcessor:
    """Handles audio processing using the Whisper model."""
    def __init__(self, model_name, non_english):
        try:
            self.model = self.load_whisper_model(model_name, non_english)
        except Exception as e:
            raise Exception(f"Failed to initialize Whisper model: {e}")

    def load_whisper_model(self, model_name, non_english):
        """Load and return the Whisper model based on the specified options."""
        try:
            model_suffix = ".en" if not non_english else ""
            full_model_name = f"{model_name}{model_suffix}"
            return stable_whisper.load_model(full_model_name)
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {e}")

    def transcribe_audio(self, audio_data):
        """Transcribe audio data to text using the Whisper model."""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
            return result['text'].strip()
        except Exception as e:
            raise Exception(f"Transcription failed: {e}")


class AudioFactCheckerApp:
    """Main application class for the audio fact checker."""
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.running = False
        self.processor = WhisperAudioProcessor(args.model, args.non_english)
        self.transcription_queue = Queue()
        self.setup_ui()

    def setup_ui(self):
        """Initialize UI components for the application."""
        self.start_button = tk.Button(self.root, text='Start', command=self.start_async_process)
        self.start_button.pack()
        self.stop_button = tk.Button(self.root, text='Stop', command=self.stop_process)
        self.stop_button.pack()
        self.status_label = tk.Label(self.root, text="Not recording", fg="red")
        self.status_label.pack()
        self.feedback_text = scrolledtext.ScrolledText(self.root, height=10)
        self.feedback_text.pack()

        self.language_var = tk.StringVar(value="English")
        languages = ["English"]  # Extend based on model capabilities
        self.language_menu = tk.OptionMenu(self.root, self.language_var, *languages)
        self.language_menu.pack()

        self.audio_device_var = tk.StringVar(value="Default")
        self.input_devices = self.get_audio_input_devices()
        self.input_device_menu = tk.OptionMenu(self.root, self.audio_device_var, *self.input_devices)
        self.input_device_menu.pack()

        # Sound bar
        self.sound_bar = tk.Label(self.root, bg="green", width=20)
        self.sound_bar.pack(fill=tk.X, padx=10, pady=5)

        self.welcome_label = tk.Label(self.root, text="Welcome to Audio Fact Checker!", font=("Arial", 12, "bold"))
        self.welcome_label.pack()
        self.gpt_notification_label = tk.Label(self.root, text="", font=("Arial", 10, "italic"))
        self.gpt_notification_label.pack()

    def capture_audio(self):
        """Capture and process audio in a separate thread."""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.args.energy_threshold
        with sr.Microphone(sample_rate=16000, device_index=self.get_audio_device_index()) as source:
            recognizer.adjust_for_ambient_noise(source)
            while self.running:
                try:
                    audio = recognizer.listen(source, timeout=self.args.record_timeout, phrase_time_limit=self.args.phrase_timeout)
                    audio_data = audio.get_raw_data()
                    self.update_sound_bar(True)  # Update sound bar when sound is detected
                    self.show_gpt_notification("GPT-4 is reviewing the information...")  # Show GPT-4 notification
                    transcription = self.processor.transcribe_audio(audio_data)
                    self.transcription_queue.put(transcription)
                    self.root.after(0, self.update_feedback, transcription)
                    self.fact_check(transcription)  # Fact-check the transcription
                except Exception as e:
                    self.update_sound_bar(False)  # Reset sound bar if no sound detected
                    self.root.after(0, self.handle_error, f"Error: {e}")

    def update_sound_bar(self, sound_detected):
        """Update the sound bar based on whether sound is detected."""
        if sound_detected:
            self.sound_bar.config(bg="green")
        else:
            self.sound_bar.config(bg="red")

    def update_feedback(self, message):
        """Update the feedback text box with a new message."""
        self.feedback_text.insert(tk.END, message + "\n")
        self.feedback_text.see(tk.END)

    def start_async_process(self):
        """Start the asynchronous audio processing."""
        if not self.running:
            self.running = True
            self.status_label.config(text="Recording...", fg="green")
            self.audio_thread = Thread(target=self.capture_audio, daemon=True)
            self.audio_thread.start()

    def stop_process(self):
        """Stop the audio processing."""
        self.running = False
        self.status_label.config(text="Not recording", fg="red")

    def fact_check(self, statement):
        """Fact-check a statement using GPT-4."""
        try:
            prompt = f"Given the context: {statement}, evaluate the accuracy of the following statement and explain its implications or compare it with known data or sources:\n\nStatement: \"{statement}\""
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",  # Specify the GPT-4 model
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant tasked with evaluating the accuracy of statements within a given context, explaining their implications, and comparing them with known data or sources."},
                    {"role": "user", "content": prompt}
                ]
            )
            fact_check_result = response.choices[0].message['content']
            self.update_feedback(f"Fact-check result for \"{statement}\": {fact_check_result}")
        except Exception as e:
            self.handle_error(f"Fact-checking failed for \"{statement}\": {e}")

    def handle_error(self, error_message):
        """Handle and display errors."""
        messagebox.showerror("Error", error_message)

    def get_default_audio_device_index(self):
        """Get the index of the default audio input device."""
        return sr.Microphone().device_index

    def get_audio_input_devices(self):
        """Get a list of available audio input devices."""
        devices = ["Default"]
        for name in sr.Microphone.list_microphone_names():
            devices.append(name)
        return devices

    def get_audio_device_index(self):
        """Get the index of the selected audio input device."""
        selected_device = self.audio_device_var.get()
        if selected_device == "Default":
            return None
        else:
            return self.input_devices.index(selected_device) - 1

    def show_gpt_notification(self, message):
        """Show a notification indicating GPT-4 is reviewing the information."""
        self.gpt_notification_label.config(text=message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true', help="Use a non-English model")
    parser.add_argument("--energy_threshold", default=1000, type=int, help="Energy threshold for the microphone")
    parser.add_argument("--record_timeout", default=5, type=float, help="Timeout for recording, in seconds")
    parser.add_argument("--phrase_timeout", default=3, type=float, help="Timeout for phrases, in seconds")
    args = parser.parse_args()

    do_override()  # Apply the distutils override

    root = tk.Tk()
    app = AudioFactCheckerApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
