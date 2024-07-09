@echo off
setlocal
set "outputFile=main\output.txt"

:menu
cls
echo Select an option:
echo 1. Fresh Install / First Time 
echo 2. Only Install
echo 
echo 3. (Run) CENSOR or GENERATE SUBTITLES FOR VIDEO. 
echo 4. Exit
echo.

set /p choice="Enter your choice: "
if "%choice%"=="1" goto runAll
if "%choice%"=="2" goto step1
if "%choice%"=="3" goto step2
if "%choice%"=="4" exit

echo Invalid choice
goto menu

:runAll
    echo Running dependency installer...
    call "main\step1_dependency_installer.bat" >> "%outputFile%"
    echo Running installer check - then running...
    call "_ffmpeg\ffmpeg.bat" >> "%outputFile%"
    call "_ffmpeg\setPath.bat" >> "%outputFile%"
echo Running installer check - then running...
call "main\step2-run.bat" >> "%outputFile%"
echo Completed.
goto end

:step1
echo Running dependency installer...
call "main\step1_dependency_installer.bat" >> "%outputFile%"
goto end

:step2
echo Running installer check - then running...
call "main\step2-run.bat" >> "%outputFile%"
goto end

:end
echo Press any key to return to the menu...
pause > nul
goto menu