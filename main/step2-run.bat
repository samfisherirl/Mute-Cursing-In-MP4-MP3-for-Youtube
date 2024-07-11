@echo off
SET venv_dir=venv
SET pyfile=mute_cursing_wav.py
SET requirements_file=requirements.txt

REM Create or activate the virtual environment
IF EXIST "%venv_dir%\Scripts\activate.bat" (
    ECHO Virtual environment found. Activating...
) ELSE (
    ECHO Creating virtual environment...
    python -m venv %venv_dir%
)

CALL %venv_dir%\Scripts\activate.bat

REM Upgrade pip and install dependencies
ECHO Upgrading pip...
%venv_dir%\Scripts\python.exe -m pip install --upgrade pip

ECHO Installing dependencies from %requirements_file%...
%venv_dir%\Scripts\python.exe -m pip install --upgrade -r %requirements_file%

REM Check for the specific Python file's existence or find another suitable one
IF EXIST "%pyfile%" (
    ECHO Found specified Python script: %pyfile%
) ELSE (
    ECHO Specified Python script not found. Looking for an alternative...
    FOR /F "delims=" %%i IN ('DIR *.py /B /A:-D /O:N 2^>nul') DO (
        IF NOT "%%i" == "__init__.py" (
            SET "pyfile=%%i"
            ECHO Found alternative script: %%i
            GOTO ExecutePyFile
        )
    )
    ECHO No suitable Python file found. Exiting...
    GOTO End
)

:ExecutePyFile
REM Run the Python script
ECHO Running Python script: %pyfile%
%venv_dir%\Scripts\python.exe %pyfile%

:End
REM Keep the command window open
pause