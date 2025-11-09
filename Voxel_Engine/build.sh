#!/bin/bash
# Build script for ARES Voxel Engine

set -e  # Exit on error

echo "=== Building ARES Voxel Engine ==="

# Create bin directory if it doesn't exist
mkdir -p bin

# Common source files (no main function)
COMMON_SOURCES="src/Camera.cpp src/PixelMotion.cpp src/Scenario.cpp src/Target.cpp src/VoxelMotion.cpp src/SparseVoxelGrid.cpp"

# Common compiler flags
COMMON_FLAGS="-std=c++17 -O3 -march=native -I./third_party"

# Add OpenMP if available
if command -v gcc &> /dev/null; then
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    if [ "$GCC_VERSION" -ge 4 ]; then
        COMMON_FLAGS="$COMMON_FLAGS -fopenmp"
        echo "OpenMP support: enabled"
    else
        echo "OpenMP support: disabled (GCC too old)"
    fi
else
    echo "OpenMP support: disabled (GCC not found)"
fi

# Build main executable (no ZeroMQ needed)
echo ""
echo "Building VoxelEngine (main)..."
g++ $COMMON_FLAGS src/main.cpp $COMMON_SOURCES -o bin/VoxelEngine
echo "✓ VoxelEngine built successfully: bin/VoxelEngine"

# Build inference executable (requires ZeroMQ)
echo ""
echo "Building VoxelEngineInfer (with inference bridge)..."
if ldconfig -p | grep -q libzmq; then
    g++ $COMMON_FLAGS src/main_infer.cpp $COMMON_SOURCES -o bin/VoxelEngineInfer -lzmq
    echo "✓ VoxelEngineInfer built successfully: bin/VoxelEngineInfer"
else
    echo "⚠ Skipping VoxelEngineInfer: libzmq not found"
    echo "  Install with: sudo apt-get install libzmq3-dev"
fi

# Build test executable
echo ""
echo "Building test executable..."
g++ $COMMON_FLAGS src/test.cpp $COMMON_SOURCES -o bin/test -lzmq 2>/dev/null && echo "✓ test built: bin/test" || echo "⚠ test build skipped (requires ZeroMQ)"

echo ""
echo "=== Build complete ==="
echo ""
echo "Usage:"
echo "  ./bin/VoxelEngine <scenario.json> [frame1] [frame2] [output.json]"
echo ""
echo "Example:"
echo "  ./bin/VoxelEngine ../Scenarios/Scenario2/scenario.json 0 10"
