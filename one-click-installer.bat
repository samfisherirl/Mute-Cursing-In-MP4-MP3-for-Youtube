@echo off
SET venv_dir=venv
SET batch_commands_file=pytorch_url.txt

REM Check if the virtual environment directory exists
IF EXIST "%venv_dir%\Scripts\activate.bat" (
    ECHO Virtual environment found. Activating...
) ELSE (
    ECHO Creating virtual environment...
    python -m venv %venv_dir%
)

REM Activate the virtual environment
CALL %venv_dir%\Scripts\activate.bat

REM Upgrade pip and install requirements if needed
ECHO Installing dependencies...
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt

REM Check if the batch commands file exists
IF NOT EXIST "%batch_commands_file%" (
    ECHO Batch commands file not found. Exiting...
    EXIT /B
)

REM Read and execute each line from the batch commands file
FOR /F "tokens=*" %%A IN (%batch_commands_file%) DO (
    %%A
)


REM Run the Python script
python audio_or_video_censor.py

REM close this window on completion

REM Pause the command window
pause