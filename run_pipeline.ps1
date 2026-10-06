$code   = "C:\Users\sammxie\OneDrive - Boston University\Howelab\Code\Calcium_2p_preprocessing"
$pyexe  = "C:\Users\sammxie\anaconda3\envs\suite2p\python.exe"   # adjust
$logdir = "$code\logs"
New-Item -ItemType Directory -Force $logdir | Out-Null
$log = "$logdir\pipeline_$(Get-Date -Format yyyy-MM-dd).log"

function Log($m) { "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m | Tee-Object -FilePath $log -Append }

Log "===== MATLAB: batch_2p_Calcium ====="
# cmd /c merges stderr into stdout; piping makes matlab.exe stay attached to the console
cmd /c "matlab -batch ""run('$code\batch_2p_Calcium.m')"" 2>&1" | Tee-Object -FilePath $log -Append
$matlabExit = $LASTEXITCODE
Log "MATLAB exit code: $matlabExit"
if ($matlabExit -ne 0) { Log "MATLAB failed, skipping Python"; exit 1 }

Log "===== PYTHON: run_suite2p_on_MC ====="
cmd /c """$pyexe"" -u ""$code\python\run_suite2p_on_MC.py"" 2>&1" | Tee-Object -FilePath $log -Append
Log "Python exit code: $LASTEXITCODE"