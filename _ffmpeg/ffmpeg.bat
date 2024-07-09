@echo off
setlocal

:: Define paths
set "workDir=%~dp0"
set "ffmpegDir=%userprofile%\Documents\ffmpeg_"
set "archive=%workDir%ffmpeg.7z"
set "sevenZip=%workDir%7za.exe"

:: Define paths
set "ffmpegBin=%userprofile%\Documents\ffmpeg_\bin"

set FFMPEG_PATH=%USERPROFILE%\Documents\ffmpeg_\bin
Powershell -Command "[Environment]::SetEnvironmentVariable('Path', $Env:Path + ';%FFMPEG_PATH%', [System.EnvironmentVariableTarget]::Machine)"

:: 1. Check if ffmpeg_ exists in Documents
if not exist "%ffmpegDir%\" (
    :: 2. If not, create it
    echo Creating directory "%ffmpegDir%"...
    mkdir "%ffmpegDir%"
)

:: 3. Use 7za.exe in working dir to extract ffmpeg.7z to ffmpeg_
if exist "%sevenZip%" (
    if exist "%archive%" (
        echo Extracting "%archive%" to "%ffmpegDir%"...
        "%sevenZip%" x "%archive%" -o"%ffmpegDir%"
        if %ERRORLEVEL% == 0 (
            echo Successfully extracted ffmpeg.
        ) else (
            echo Failed to extract ffmpeg.
        )
    ) else (
        echo Archive ffmpeg.7z not found.
    )
) else (
    echo 7za.exe not found in working directory.
)

:: Check if ffmpeg bin folder is in the PATH
echo %PATH% | findstr /C:"%ffmpegBin%" > nul
if %ERRORLEVEL% == 0 (
    echo ffmpeg bin directory is already in the PATH.
) else (
    echo Adding ffmpeg bin directory to PATH...
    setx PATH "%PATH%;%ffmpegBin%"
)
@echo off
set WORKING_DIR=%~dp0
PowerShell -Command "Start-Process cmd.exe -ArgumentList '/c ""%WORKING_DIR%setPath.bat""' -Verb RunAs"
endlocal