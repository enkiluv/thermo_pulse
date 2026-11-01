@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo =======================================================================
echo      THERMO PULSE V14 MECHANISM AND UTILITY BENCHMARK (Qwen2.5-3B, GPU)
echo      Main experiment only: 200 samples x 5 seeds, M1-M9
echo =======================================================================
echo.

rem [GPU Environment] Dedicated environment; do not modify the existing benchz Python or M9 chat environment.
set "BENCH_VENV=%~dp0.venv_benchz_gpu"
set "BENCH_PY=%BENCH_VENV%\Scripts\python.exe"
if not exist "%BENCH_PY%" (
    echo [Setup] Creating isolated benchmark virtual environment...
    py -3.10 -m venv "%BENCH_VENV%"
    if errorlevel 1 (
        echo [ERROR] Could not create Python 3.10 virtual environment.
        pause
        exit /b 1
    )
)

rem Install dependencies only if absent; never silently switch the GPU run to CPU.
"%BENCH_PY%" -c "import torch; assert torch.cuda.is_available(); print('[GPU] torch:', torch.__version__, '/ CUDA:', torch.cuda.get_device_name(0))" >nul 2>&1
if errorlevel 1 (
    echo [Setup] Installing NVIDIA CUDA PyTorch in benchmark-only environment...
    "%BENCH_PY%" -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 goto :install_failed
)

"%BENCH_PY%" -c "import transformers, accelerate, pandas, numpy, statsmodels" >nul 2>&1
if errorlevel 1 (
    echo [Setup] Installing benchmark Python dependencies...
    "%BENCH_PY%" -m pip install "transformers>=4.51,<5" "accelerate>=1.0,<2" "numpy<2" pandas statsmodels
    if errorlevel 1 goto :install_failed
)

echo.
"%BENCH_PY%" -c "import torch; print('[GPU] PyTorch:', torch.__version__); print('[GPU] CUDA:', torch.cuda.is_available()); print('[GPU] Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if errorlevel 1 goto :cuda_failed
"%BENCH_PY%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto :cuda_failed

echo.
rem -----------------------------------------------------------------------
rem [1/3] Step 1: Smoke Test (1x1) - DISABLED (kept for future reuse)
rem -----------------------------------------------------------------------
rem python benchz.py --model qwen --macro --sample 1 --seeds 1
rem if errorlevel 1 goto :benchmark_failed
rem echo [Cooldown] Resting GPU for 30 seconds...
rem timeout /t 30 /nobreak >nul

echo.
rem -----------------------------------------------------------------------
rem [2/3] Step 2: Pilot Run (10x2) - DISABLED (kept for future reuse)
rem -----------------------------------------------------------------------
rem python benchz.py --model qwen --macro --sample 10 --seeds 2
rem if errorlevel 1 goto :benchmark_failed
rem echo [Cooldown] Resting GPU for 30 seconds...
rem timeout /t 30 /nobreak >nul

echo -----------------------------------------------------------------------
echo [3/3] Main Benchmark: 200 samples x 5 seeds, M1-M9, Logic + Math
echo -----------------------------------------------------------------------
echo [Notice] Fast protocol: Qwen2.5-3B; require all layers on GPU.
echo [Notice] Short-reasoning prompt; 256-token cap; M1-M9 unchanged; mechanism logging.
set "BENCH_MAX_NEW_TOKENS=256"
set "BENCH_REQUIRE_FULL_CUDA=1"
echo.
"%BENCH_PY%" "%~dp0benchz.py" --model qwen-3b --macro --sample 200 --seeds 5 --start M1 --output "%~dp0qwen3b_v14_results_s200.json" --html_output "%~dp0qwen3b_v14_report_s200.html"
if errorlevel 1 goto :benchmark_failed
"%BENCH_PY%" "%~dp0post_v14.py" "%~dp0qwen3b_v14_results_s200.json"
if errorlevel 1 goto :benchmark_failed

echo.
echo [SUCCESS] Main benchmark completed.
echo [Saved] qwen3b_v14_results_s200.json
echo [Saved] qwen3b_v14_report_s200.html
pause
exit /b 0

:install_failed
echo [ERROR] Dependency installation failed in .venv_benchz_gpu.
pause
exit /b 1

:cuda_failed
echo [ERROR] CUDA is not available in benchmark environment. Stopping to avoid CPU-only test.
pause
exit /b 1

:benchmark_failed
echo [ERROR] Main benchmark failed. Existing output files are not deleted.
pause
exit /b 1
