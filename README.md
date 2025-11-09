# ARES - Aerial Reconnaissance Estimation System

A multi-camera voxel-based system for 3D trajectory estimation and motion tracking using sparse voxel processing and deep learning inference.

## Overview

ARES is a high-performance 3D computer vision system that uses multiple camera feeds to:
1. Build sparse 3D voxel representations of a scene
2. Track motion between temporal frames using voxel change detection
3. Estimate object trajectories using 4D sparse convolutional neural networks
4. Generate synthetic training scenarios using Blender automation

The system processes images from multiple synchronized cameras, performs ray-casting into a 3D voxel grid, detects motion between frames, and uses a MinkowskiEngine-based 4D U-Net for trajectory prediction.

## System Architecture

### Core Components

```
ARES/
├── Voxel_Engine/          # Primary C++ voxel processing engine
├── Voxel_Engine2/         # Alternative/experimental voxel engine
├── CNN/                   # Neural network training and inference
├── Scenario_Generator/    # Blender-based synthetic data generation
├── Scenarios/             # Pre-generated scenario datasets
├── Frames/                # Camera frame data storage
└── CameraUnit/            # Camera hardware integration designs
```

### Processing Pipeline

```
Camera Images (Multiple Views)
        ↓
[Sparse Voxel Engine]
  - Ray-casting from each camera
  - Sparse voxel grid construction
  - Multi-camera intersection tracking
        ↓
[Motion Analysis]
  - Frame-to-frame voxel comparison
  - Change detection and quantification
  - Motion vector computation
        ↓
[4D CNN Inference]
  - Sparse tensor conversion
  - 4D U-Net processing (x,y,z,t)
  - Trajectory probability heatmap
        ↓
Output: 3D position probability map
```

## Main Entry Points

### 1. Voxel Processing (C++)

**Primary executable**: `Voxel_Engine/src/main.cpp`

```bash
cd Voxel_Engine
./compile.sh  # Linux/WSL
# Or compile.bat on Windows

# Run voxel processing
./bin/VoxelEngine <scenario_file> [frame1] [frame2] [output_json]
```

**Example**:
```bash
./bin/VoxelEngine ../Scenarios/Scenario2/scenario.json 100 125 output.json
```

**What it does**:
- Loads camera images and metadata from a scenario
- Performs sparse ray-casting to build 3D voxel grids for two time frames
- Computes motion changes between frames
- Exports voxel data as JSON for ML/MATLAB processing

**Key parameters**:
- `frame1`, `frame2`: Frame numbers to compare (default: 100, 125)
- `topPercentage`: Brightness threshold for ray selection (default: 1%)
- Output: `unified_scene_data.json` with voxel grid and change data

---

### 2. Voxel Processing with CNN Inference (C++)

**Inference executable**: `Voxel_Engine/src/main_infer.cpp`

```bash
# Terminal 1: Start Python inference service
cd Voxel_Engine
python infer_service.py

# Terminal 2: Run voxel engine with inference
./bin/VoxelEngineInfer <scenario_file> [frame1] [frame2]
```

**What it does**:
- Same voxel processing as `main.cpp`
- Additionally sends voxel data to Python service via ZeroMQ + shared memory
- Receives trajectory probability predictions
- Outputs `inference_heatmap.json` with per-voxel probabilities

**Communication**: Uses ZeroMQ IPC and platform-specific shared memory (mmap/Windows file mapping)

---

### 3. CNN Training (C++)

**Training executable**: `CNN/train.cpp`

```bash
cd CNN
g++ -std=c++17 -O3 -fopenmp train.cpp -o train.exe
./train.exe <frames_dir> <model_output.bin>
```

**What it does**:
- Implements a custom 4D sparse convolutional U-Net in C++
- Trains on 10-frame temporal segments from voxel data
- Uses synthetic gradients and Adam optimizer
- Saves model weights to binary format
- **Note**: Primary training is done in Python (see below)

---

### 4. CNN Inference (C++)

**Inference executable**: `CNN/infer.cpp`

```bash
cd CNN
./infer.exe model.bin <frames_dir> <output_dir>
```

**What it does**:
- Loads trained 4D U-Net model weights
- Processes 10-frame segments from voxel data
- Outputs per-voxel trajectory probability heatmaps as CSV files

---

### 5. Python Inference Service

**Service**: `Voxel_Engine/infer_service.py`

```bash
cd Voxel_Engine
python infer_service.py
```

**What it does**:
- Runs a ZeroMQ server on `tcp://127.0.0.1:5555`
- Loads `model_final.pt` (PyTorch MinkowskiEngine 4D U-Net)
- Receives sparse voxel coordinates and features via shared memory
- Performs GPU/CPU inference
- Returns probability predictions to C++ client

**Model Architecture**: `Simple4DUNet_Slim` (16-channel 4D sparse CNN)

**Dependencies**: PyTorch, MinkowskiEngine, ZeroMQ, NumPy

---

### 6. Scenario Generation (Blender)

**Blender addon**: `Scenario_Generator/blender_addon/generate.py`

1. Install the addon in Blender (Edit → Preferences → Add-ons → Install)
2. Open Blender and navigate to the 3D Viewport sidebar
3. Find "ARES Scenario Generator" panel

**Features**:
- Procedural camera placement from pre-defined nodes
- Animated target trajectory generation
- Configurable FOV, speeds, and target counts
- Exports multi-camera synchronized frames with JSON metadata
- Supports drone mesh models or simple sphere targets

**Workflow**:
1. Generate camera visibility bounding box
2. Generate target navigation nodes within box
3. Configure cameras and targets
4. Generate scenario
5. Render animation (via Blender's render engine)
6. Export frames to `Scenarios/` directory

---

## Key Technologies

### Sparse Voxel Grid (`SparseVoxelGrid`)

- **File**: `Voxel_Engine/src/SparseVoxelEngine.h`
- Hash-based sparse storage (only occupied voxels stored)
- Custom fast hash function (3x faster than std::hash)
- OpenMP parallelized ray-casting
- Typical memory savings: 95-99% vs dense grids
- Voxel size: 5m (configurable)

### Ray-Casting Algorithm

- **Method**: DDA (Digital Differential Analyzer) voxel traversal
- Casts rays from each camera pixel through 3D space
- Brightness-based filtering (top N% brightest pixels)
- Accumulates camera intersections per voxel
- Adaptive block-size processing for cache efficiency

### Motion Detection (`SparseVoxelMotion`)

- Compares two temporal voxel grids
- Computes per-voxel change metrics:
  - Absolute change (intensity delta)
  - Relative change (percentage)
  - Change type (increase/decrease/stable)
- Thread-local accumulation to minimize mutex contention

### 4D Convolutional Neural Network

- **Architecture**: Sparse 4D U-Net with residual blocks
- **Dimensions**: (X, Y, Z, Time)
- **Input features** (7 channels):
  - Ray intersection count
  - Number of cameras seeing voxel
  - 3D position (x, y, z)
  - Composite features
- **Output**: Per-voxel trajectory probability (sigmoid)
- **Framework**: MinkowskiEngine (Python) or custom C++ implementation

---

## Data Flow

### Scenario Structure

```
Scenarios/Scenario2/
├── scenario.json                 # Scenario metadata
├── Cam1_Top/
│   ├── 0000.json                # Frame metadata (camera pose, etc)
│   ├── 0000.png                 # Camera image
│   └── ...
├── Cam2_Top/
│   └── ...
└── target_drone.json            # Target ground truth trajectory
```

### Frame JSON Format

```json
{
  "frame": 0,
  "position": [x, y, z],           // Camera position in world coords
  "rotation_matrix": [...],         // 3x3 rotation matrix
  "sensor_size_mm": 6.287,
  "fov_degrees": 60,
  "image_path": "0000.png",
  "mask_path": "0000_mask.png"     // Optional segmentation mask
}
```

### Scenario JSON Format

```json
{
  "scenario_name": "Scenario2",
  "frame_rate": 30,
  "cameras": [
    {"name": "Cam1_Top", "frames_dir": "path/to/frames"},
    ...
  ],
  "targets": [
    {"name": "drone", "file": "target_drone.json"}
  ]
}
```

---

## Visualization Tools

### 1. Voxel Grid Viewer

```bash
cd Voxel_Engine
python VoxelViewer.py unified_scene_data.json
```

Interactive 3D visualization of voxel grids with intensity mapping.

### 2. Heatmap Viewer

```bash
cd Voxel_Engine
python HeatmapViewer.py inference_heatmap.json
```

Visualizes CNN inference probability heatmaps in 3D.

### 3. CNN Heatmap Viewer

```bash
cd CNN
python view_heatmap.py output/0-9_heatmap.csv
```

Displays per-voxel probabilities from C++ inference output.

---

## Dependencies

### C++ Components
- **Compiler**: GCC/Clang with C++17 support, MSVC 2022
- **OpenMP**: For parallel processing (optional but recommended)
- **ZeroMQ**: Inter-process communication (`libzmq`)
- **stb_image**: Header-only image loading (included in `third_party/`)
- **nlohmann/json**: JSON parsing (included in `third_party/`)

### Python Components
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **MinkowskiEngine**: Sparse convolution library
- **ZeroMQ**: `pyzmq`
- **NumPy**: Array operations
- **Matplotlib**: For visualization tools
- **Blender**: 4.0+ (for scenario generation)

### Build System
- CMake (for Voxel_Engine on Windows)
- Direct g++ compilation (for Linux/WSL)

---

## Build Instructions

### Linux/WSL

```bash
# Voxel Engine - Use the provided build script
cd Voxel_Engine
./build.sh

# This builds:
#   - bin/VoxelEngine (main executable, no ZeroMQ required)
#   - bin/VoxelEngineInfer (inference executable, requires ZeroMQ)
#   - bin/test (test executable)

# If ZeroMQ is not installed (for inference support):
sudo apt-get install libzmq3-dev

# CNN Training
cd ../CNN
g++ -std=c++17 -O3 -fopenmp train.cpp -o train.exe
```

### Windows (Visual Studio)

```batch
cd Voxel_Engine
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### Python Environment

```bash
pip install torch torchvision
pip install MinkowskiEngine
pip install pyzmq numpy matplotlib
```

---

## Usage Examples

### End-to-End Pipeline

```bash
# 1. Generate scenario in Blender (manual)
# Use ARES Scenario Generator addon to create cameras and targets

# 2. Process voxel grids
cd Voxel_Engine
./bin/VoxelEngine ../Scenarios/Scenario2/scenario.json 0 10 output.json

# 3. Start inference service (in separate terminal)
python infer_service.py

# 4. Run voxel processing with inference
./bin/VoxelEngineInfer ../Scenarios/Scenario2/scenario.json 0 10

# 5. Visualize results
python HeatmapViewer.py inference_heatmap.json
```

### Training a Model

```bash
# Prepare training data (generate scenarios in Blender)
# Process scenarios to voxel format using VoxelEngine

# Train (Python version recommended)
cd Voxel_Engine
python infer_service.py --train --data-dir ../Frames/

# Or use C++ trainer
cd CNN
./train.exe ../Frames/training_data/ model.bin
```

---

## Performance Characteristics

### Voxel Engine
- **Typical grid size**: 100×100×60 voxels (5m voxel size)
- **Sparsity**: 95-99% (only 1-5% of voxels occupied)
- **Memory usage**: 10-50 MB per frame (vs 500+ MB for dense)
- **Processing time**: 500-2000ms per frame pair (6 cameras, OpenMP enabled)

### Ray-Casting
- **Rays per camera**: ~10-50k (depends on brightness threshold)
- **Brightness filtering**: Top 1% of pixels by default
- **Parallel efficiency**: Near-linear scaling up to 8 threads

### CNN Inference
- **Input size**: Typically 10k-100k active voxels
- **Inference time**: 20-100ms per 10-frame segment (GPU)
- **Model size**: ~18 MB (PyTorch), ~6 MB (C++ binary)

---

## Troubleshooting

### "No camera_node empties found"
The scenario generator requires pre-placed camera nodes. Generate them using the "Generate Box" and "Generate Target Nodes" tools in the Blender addon.

### "ZMQ connection refused"
Ensure `infer_service.py` is running before starting `VoxelEngineInfer`. Check that port 5555 is not blocked by firewall.

### "OpenMP not available"
Install OpenMP support for your compiler or remove `-fopenmp` flag (single-threaded fallback).

### Empty voxel grids
Check that camera images are not completely black. Verify `topPercentage` parameter is not too restrictive.

### Memory errors
Reduce voxel grid size or increase voxel size (default 5m). Check available RAM.

---

## Project Status

**Current State**: Research/Development

- ✅ Sparse voxel engine with multi-camera support
- ✅ Motion detection between temporal frames
- ✅ 4D CNN architecture (both C++ and Python)
- ✅ ZeroMQ-based inference bridge
- ✅ Blender scenario generator
- ✅ Visualization tools
- ⚠️ No pre-trained models included (requires training on scenarios)
- ⚠️ Documentation is minimal (this README is the primary documentation)
- ⚠️ No automated tests

**Voxel_Engine vs Voxel_Engine2**: Two implementations exist. `Voxel_Engine` is the primary, production version. `Voxel_Engine2` appears to be an experimental/alternative implementation.

---

## Camera Compatibility

The system expects:
- Synchronized multi-camera feeds
- Known camera intrinsics (FOV, sensor size)
- Known camera extrinsics (position, rotation matrix)
- Grayscale or RGB images (auto-converted to grayscale)

**Camera models**: Generic pinhole camera model. The `CameraUnit/` directory contains hardware designs for custom camera units (Orange Pi Zero 3 based).

No camera driver code is currently implemented - the system works with pre-rendered or pre-captured image sequences.

---

## License

Not specified in repository.

---

## Contributing

This repository appears to be a research project. No contributing guidelines are provided.

---

## Contact

No contact information provided in repository.

---

## References

- **MinkowskiEngine**: Sparse tensor library for deep learning (https://github.com/NVIDIA/MinkowskiEngine)
- **ZeroMQ**: High-performance messaging library (https://zeromq.org/)
- **Blender**: 3D creation suite (https://www.blender.org/)

---

## Acknowledgments

Based on sparse voxel reconstruction and 4D convolutional neural network techniques for multi-view 3D tracking.

---

**Last Updated**: Auto-generated based on repository state
