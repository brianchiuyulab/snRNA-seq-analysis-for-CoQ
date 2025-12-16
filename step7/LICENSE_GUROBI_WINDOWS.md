# Gurobi license setup (Windows)

## 1. Verify Gurobi installation
Locate executables (example):
- C:\gurobi1300\win64\bin\gurobi_cl.exe
- C:\gurobi1300\win64\bin\grbgetkey.exe

## 2. Install license (one-time)
Run in Windows Terminal / Anaconda Prompt:

C:\gurobi1300\win64\bin\grbgetkey.exe <YOUR_KEY>

This will create:
%USERPROFILE%\gurobi.lic

## 3. Confirm license file exists
dir %USERPROFILE%\gurobi.lic

## 4. Verify Gurobi can read the license
C:\gurobi1300\win64\bin\gurobi_cl.exe --license

Expected output includes something like:
"Using license file C:\Users\<user>\gurobi.lic"
"Academic license ... expires YYYY-MM-DD"

## Optional: set persistent env var (recommended)
setx GRB_LICENSE_FILE "%USERPROFILE%\gurobi.lic"

Close and reopen terminal after running setx.
