@echo off
rem Add Volume Cropper to napari's menus (the napari plugin, 0.13.0):
rem double-click this file. It installs the plugin into the conda
rem environment "volume_cropper", from this folder; see
rem "volume_cropper.bat --install-plugin". To remove it again, type in a
rem terminal: volume_cropper.bat --uninstall-plugin
call "%~dp0volume_cropper.bat" --install-plugin
