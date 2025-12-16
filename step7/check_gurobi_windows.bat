@echo off
echo [1/2] Checking license file...
dir "%USERPROFILE%\gurobi.lic" || (echo [ERROR] gurobi.lic not found & exit /b 1)

echo.
echo [2/2] Checking gurobi_cl --license ...
"C:\gurobi1300\win64\bin\gurobi_cl.exe" --license
