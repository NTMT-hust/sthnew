@echo off
REM Run experiments with both split strategies

echo ==========================================
echo Running Training Experiments
echo ==========================================
echo.

REM Experiment 1: Site Split with Focal Loss (gamma=2.0)
echo Experiment 1: Site Split Strategy (Train A, Val DC, Test B)
echo ----------------------------------------------------------
python train.py ^
    --split-strategy site_split ^
    --output outputs/site_split_gamma2 ^
    --epochs 50 ^
    --focal-gamma 2.0 ^
    --learning-rate 1e-3 ^
    --batch-size 512

echo.
echo.

REM Experiment 2: Single Site Split with Focal Loss (gamma=2.0)
echo Experiment 2: Single Site Strategy (All from Site A)
echo ----------------------------------------------------------
python train.py ^
    --split-strategy single_site ^
    --output outputs/single_site_gamma2 ^
    --epochs 50 ^
    --focal-gamma 2.0 ^
    --learning-rate 1e-3 ^
    --batch-size 512

echo.
echo.

REM Experiment 3: Site Split with Cross-Entropy (gamma=0.0)
echo Experiment 3: Site Split with Cross-Entropy (Baseline)
echo ----------------------------------------------------------
python train.py ^
    --split-strategy site_split ^
    --output outputs/site_split_ce ^
    --epochs 50 ^
    --focal-gamma 0.0 ^
    --learning-rate 1e-3 ^
    --batch-size 512

echo.
echo.
echo ==========================================
echo All experiments completed!
echo ==========================================
echo.
echo Results saved to:
echo   - outputs/site_split_gamma2/
echo   - outputs/single_site_gamma2/
echo   - outputs/site_split_ce/
echo.
echo Visualizations have been automatically generated in each directory.
pause
