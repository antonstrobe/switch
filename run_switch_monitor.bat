@echo off
setlocal
cd /d "%~dp0"

if exist "%LocalAppData%\Programs\Python\Python312\pythonw.exe" (
    start "" "%LocalAppData%\Programs\Python\Python312\pythonw.exe" "%~dp0app.pyw"
) else (
    start "" pyw -3 "%~dp0app.pyw"
)

endlocal
