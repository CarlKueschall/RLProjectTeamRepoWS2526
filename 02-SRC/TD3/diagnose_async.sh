#!/bin/bash

# Quick diagnostic script to test async training and capture issues

echo "=========================================="
echo "ASYNC TRAINING DIAGNOSTIC"
echo "=========================================="
echo ""

# Check Python version
echo "[CHECK] Python version:"
python --version
echo ""

# Check if hockey-env is installed
echo "[CHECK] hockey-env module:"
python -c "import hockey.hockey_env; print('✓ hockey-env installed')" 2>&1 || echo "✗ hockey-env NOT installed"
echo ""

# Check PyTorch
echo "[CHECK] PyTorch:"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')" 2>&1 || echo "✗ PyTorch NOT installed"
echo ""

# Check if we're in the right directory
echo "[CHECK] Working directory:"
if [ -f "train_hockey_async.py" ]; then
    echo "✓ In TD3 directory"
else
    echo "✗ train_hockey_async.py not found"
    exit 1
fi
echo ""

# Show platform
echo "[CHECK] Platform:"
python -c "import sys; print(f'  Platform: {sys.platform}')"
echo ""

echo "=========================================="
echo "RUNNING 30-SECOND TEST"
echo "=========================================="
echo ""

# Run the debug test
python test_async_debug.py --num_workers 2 --run_duration 30

echo ""
echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="
