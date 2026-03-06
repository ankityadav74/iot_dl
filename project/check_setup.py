"""
Setup Checker - Verify your environment before training
"""

import sys

def check_setup():
    print("="*70)
    print("CHECKING SETUP FOR TEMPORAL GRAPH TRANSFORMER")
    print("="*70)

    errors = []
    warnings = []

    # Check Python version
    print("\n1. Python Version...")
    if sys.version_info >= (3, 8):
        print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        errors.append("Python 3.8+ required")

    # Check PyTorch
    print("\n2. PyTorch...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"   ✓ CUDA Available: {torch.version.cuda}")
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")

            # Check for H100
            gpu_name = torch.cuda.get_device_name(0).lower()
            if 'h100' in gpu_name:
                print(f"   🚀 H100 Detected! Optimizations enabled.")

            # Check memory
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✓ GPU Memory: {mem_gb:.1f} GB")
        else:
            warnings.append("CUDA not available - will run on CPU (very slow)")
    except ImportError:
        errors.append("PyTorch not installed")

    # Check PyTorch Geometric
    print("\n3. PyTorch Geometric...")
    try:
        import torch_geometric
        print(f"   ✓ PyG {torch_geometric.__version__}")
    except ImportError:
        errors.append("PyTorch Geometric not installed")

    # Check DGL
    print("\n4. DGL...")
    try:
        import dgl
        print(f"   ✓ DGL {dgl.__version__}")
    except ImportError:
        warnings.append("DGL not installed (optional)")

    # Check other dependencies
    print("\n5. Other Dependencies...")
    deps = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'tqdm']
    for dep in deps:
        try:
            __import__(dep)
            print(f"   ✓ {dep}")
        except ImportError:
            errors.append(f"{dep} not installed")

    # Check files
    print("\n6. Required Files...")
    import os
    files = ['tgt_model.py', 'graph_builder.py', 'train_tgt.py']
    for file in files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            errors.append(f"{file} not found")

    # Check dataset
    print("\n7. Dataset...")
    if os.path.exists('data/raw/WSN-DS.csv'):
        print(f"   ✓ data/raw/WSN-DS.csv found")
        import pandas as pd
        df = pd.read_csv('data/raw/WSN-DS.csv')
        print(f"   ✓ Shape: {df.shape}")
    else:
        warnings.append("data/raw/WSN-DS.csv not found - make sure dataset is in correct location")

    # Summary
    print("\n" + "="*70)
    if errors:
        print("❌ ERRORS FOUND:")
        for error in errors:
            print(f"   - {error}")
        print("\nFix these errors before training!")
    elif warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n✅ Setup OK, but consider fixing warnings")
    else:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to start training:")
        print("   python3 train_tgt.py")
    print("="*70)

if __name__ == "__main__":
    check_setup()
