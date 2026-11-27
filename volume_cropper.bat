@echo off
rem Start Volume Cropper (the napari GUI).
rem Double-click this file, or drop a .zarr folder onto it to open that image.
rem
rem The napari plugin (0.13.0), from a terminal:
rem   volume_cropper.bat --install-plugin     add Volume Cropper to napari's menus
rem   volume_cropper.bat --uninstall-plugin   remove it again
rem   volume_cropper.bat --napari             start napari itself
rem (or double-click install_napari_plugin.bat).
rem
rem It finds conda in the usual places (or set VOLUME_CROPPER_CONDA to its folder),
rem and uses the conda environment "volume_cropper" if it exists, otherwise
rem "napari-env" (or set VOLUME_CROPPER_ENV to another environment name).
rem The plugin is only installed into "volume_cropper" (or VOLUME_CROPPER_ENV).
setlocal

set "MODE=gui"
if /i "%~1"=="--install-plugin" set "MODE=install"
if /i "%~1"=="--uninstall-plugin" set "MODE=uninstall"
if /i "%~1"=="--napari" set "MODE=napari"

set "CONDA_BASE="
for %%B in ("%VOLUME_CROPPER_CONDA%" "%LOCALAPPDATA%\miniforge3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if not defined CONDA_BASE if exist "%%~B\condabin\conda.bat" set "CONDA_BASE=%%~B"
)
if not defined CONDA_BASE (
    echo Could not find conda. Install Miniforge, or set VOLUME_CROPPER_CONDA to its folder.
    pause
    exit /b 1
)
rem Use conda itself (condabin\conda.bat), never mamba: in newer Miniforge
rem installs, Scripts\activate.bat runs "mamba activate", which fails when this
rem is started from a terminal where conda is already active.
set "CONDA_EXE=%CONDA_BASE%\Scripts\conda.exe"
set "CONDA_BAT=%CONDA_BASE%\condabin\conda.bat"
set "_CE_M="
set "_CE_CONDA="

set "ENV_NAME=%VOLUME_CROPPER_ENV%"
if not defined ENV_NAME (
    call "%CONDA_BAT%" env list | "%SystemRoot%\System32\findstr.exe" /b /c:"volume_cropper " >nul && set "ENV_NAME=volume_cropper"
)
if "%MODE%"=="install" goto :plugin_env
if "%MODE%"=="uninstall" goto :plugin_env
if not defined ENV_NAME set "ENV_NAME=napari-env"
goto :activate

:plugin_env
rem The plugin goes only into the environment made for Volume Cropper.
if defined ENV_NAME goto :activate
echo The conda environment "volume_cropper" was not found. Create it first with:
echo   conda env create -f "%~dp0env\environment.yml"
pause
exit /b 1

:activate
call "%CONDA_BAT%" activate %ENV_NAME%
if errorlevel 1 (
    echo Could not activate the conda environment "%ENV_NAME%".
    echo Create it with: conda env create -f "%~dp0env\environment.yml"
    pause
    exit /b 1
)

rem (0.23.0) Python by its full path in the environment, below: cmd.exe
rem would run a python.bat or python.exe in the current folder first.
rem Don't write Python cache files into the shared tool folder.
set "PYTHONDONTWRITEBYTECODE=1"
if "%MODE%"=="install" goto :install
if "%MODE%"=="uninstall" goto :uninstall
if "%MODE%"=="napari" goto :napari

rem The package's __main__.py, run as a file: it finds the package itself,
rem so the working folder stays the user's (relative image paths work).
"%CONDA_PREFIX%\python.exe" "%~dp0volume_cropper\__main__.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto :done

:napari
rem napari itself; Volume Cropper is in its menus if the plugin is installed.
"%CONDA_PREFIX%\python.exe" -m napari
set "EXIT_CODE=%ERRORLEVEL%"
goto :done

:install
echo Installing the Volume Cropper napari plugin into the conda environment
echo %ENV_NAME%, from "%~dp0" (nothing is written into that folder).
echo.
rem pip needs setuptools to install it; it comes with the environment
rem (napari's Qt needs it), but check, so that pip does not try to download it.
"%CONDA_PREFIX%\python.exe" -c "import setuptools" 2>nul
if errorlevel 1 goto :no_setuptools
rem -e: "editable", the environment points to this folder (new releases here
rem reach the plugin without installing again). --no-build-isolation and
rem --no-index: use what the environment has, download nothing. The "."
rem keeps the folder's last backslash from escaping the closing quote.
"%CONDA_PREFIX%\python.exe" -m pip install --no-deps --no-build-isolation --no-index --disable-pip-version-check -e "%~dp0."
if errorlevel 1 goto :install_failed
rem napari started on its own (not from this file) must not write Python
rem cache files into the tool folder either.
call "%CONDA_BAT%" env config vars set PYTHONDONTWRITEBYTECODE=1 -n %ENV_NAME% >nul
echo.
echo Done. Start napari: "conda activate %ENV_NAME%", then "napari"
echo (or: volume_cropper.bat --napari). Then use Plugins ^> Volume Cropper,
echo or Layers ^> Data ^> Crop selected layers. If %ENV_NAME% was already
echo active in your terminal, type "conda activate %ENV_NAME%" once more.
pause
exit /b 0

:no_setuptools
echo The environment %ENV_NAME% has no setuptools, which the install needs.
echo Add it with:  conda install -n %ENV_NAME% -c conda-forge setuptools
echo and then run this again.
pause
exit /b 1

:install_failed
echo.
echo The install failed (see the messages above).
pause
exit /b 1

:uninstall
"%CONDA_PREFIX%\python.exe" -m pip uninstall -y --disable-pip-version-check napari-volume-cropper
call "%CONDA_BAT%" env config vars unset PYTHONDONTWRITEBYTECODE -n %ENV_NAME% >nul 2>nul
echo.
echo Done: Volume Cropper is no longer in napari's menus (after napari is
echo started again). The tool itself still works as before.
pause
exit /b 0

:done
rem Keep the window open if something went wrong, so the error can be read.
rem (A crash gives a negative exit code, which "if errorlevel 1" misses.)
if not "%EXIT_CODE%"=="0" (
    echo Volume Cropper stopped with exit code %EXIT_CODE%.
    pause
)
