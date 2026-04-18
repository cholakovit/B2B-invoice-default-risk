@echo off
setlocal
cd /d "%~dp0"
uv run python main_advanced.py %*
