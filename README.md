# NN_for_NOVA4

A specialized neural network implementation optimized for deployment on SPAR (Systolic Array) hardware architecture, focusing on MNIST digit classification with post-training quantization (PTQ) for hardware deployment. [1](#0-0) 

## Overview

NN_for_NOVA4 implements **SPARNet**, a neural network architecture designed to maximize Processing Element (PE) utilization on systolic array hardware. The project bridges high-level PyTorch development with low-level hardware implementation by providing tools to export trained, 8-bit quantized parameters into memory initialization files (`.mem`) compatible with FPGA or ASIC workflows. [2](#0-1) 

### Hardware Context

The model uses a **row-reshape strategy** that treats the 784-pixel MNIST image as four rows of 196 elements (`[B, 4, 196]`), enabling efficient mapping of matrix multiplications to the systolic grid for 100% PE utilization. [3](#0-2) 

## Project Structure

```
NN_for_NOVA4/
├── nova.ipynb                    # Main development notebook
└── 97.06_96.93_model4*196/      # Exported .mem files (train_acc/test_acc_input_dim)
```

## Installation

### Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

### Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib
```
3. Open `nova.ipynb` in Jupyter Notebook or Google Colab

## Usage

### Training the Model

The main training pipeline is implemented in the `SPARNet` class and `train_model` function: [4](#0-3) 

```python
model = SPARNet().to(device)
history, best_acc = train_model(model, 'SPARNet', epochs=15)
```

### Model Architecture

SPARNet consists of the following components: [1](#0-0) 

- **Input Reshape**: `[B, 1, 28, 28]` → `[B, 4, 196]`
- **FC1**: Linear layer `196 → 64` with bias
- **FC2**: Linear layer `64 → 64` with bias  
- **Transform Block**: 4×4 transform per slice (`W2` parameter)
- **Learned Reducer**: Weighted reduction to final class scores
- **Activation**: ReLU non-linearity

### Quantization Pipeline

The project includes a complete post-training quantization pipeline: [5](#0-4) 

1. **Range Inspection**: Profile activation ranges using `inspect_ranges()`
2. **Model Quantization**: Build quantized model with `build_quantized_model()`
3. **Shift Search**: Grid search for optimal bit-shifts to minimize precision loss
4. **INT8 Evaluation**: Validate quantized model accuracy with `evaluate_int8()`

### Export to Hardware Files

Export quantized weights and biases to memory initialization files: [2](#0-1) 

```python
# Export weights as decimal .mem files
save_mem_file(qmodel["fc1_w"], "fc1_weights.mem")
save_mem_file(qmodel["fc2_w"], "fc2_weights.mem")
save_mem_file(qmodel["W2"], "w2_weights.mem")
save_mem_file(qmodel["reducer"], "reducer.mem")

# Export biases as hexadecimal .mem files
save_hex_mem(qmodel["fc1_b"], "fc1_bias.mem")
save_hex_mem(qmodel["fc2_b"], "fc2_bias.mem")
save_hex_mem(qmodel["b2"], "w2_bias.mem")
```

## Results

The SPARNet architecture achieves approximately 97% training accuracy and 97% test accuracy on MNIST while being optimized for systolic array hardware constraints, pre and post quantization. [6](#0-5) 

## Key Features

- **Hardware-Optimized Architecture**: Designed specifically for systolic array dataflow
- **Post-Training Quantization**: Complete PTQ pipeline for INT8 deployment
- **Memory Export**: Direct export to hardware-compatible .mem files
- **Accuracy Preservation**: Maintains high accuracy after quantization

## Notes

The repository primarily consists of the `nova.ipynb` notebook which contains all model definitions, training logic, quantization pipeline, and export functionality. The exported `.mem` files are organized in directories named with accuracy metrics and input dimensions (e.g., `97.06_96.93_model4*196/`). [7](#0-6) 

Wiki pages you might want to explore:
- [Project Overview (rithikreddypalla/NN_for_NOVA4)](/wiki/rithikreddypalla/NN_for_NOVA4#1)
