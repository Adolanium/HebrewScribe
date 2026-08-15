@echo off
:: HebrewScribe — Windows dev launcher (double-click in Explorer to run)
:: Activates the local venv if present, then launches the app without a console window.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

start "" pythonw -m hebrewscribe
