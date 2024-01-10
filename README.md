# CFU Playground - AAML Final Project - Group 4

## Setup Guide

### 1. Prepare a supported board (We use Nexys A7-100T) and install required toolchains. See [Setup Guide](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html) for more details

### 2. Clone the CFU-Playground Repository from the github

``` bash
git clone https://github.com/scott306lr/AAML_Final.git
```

### 3. Run the setup script

``` bash
cd AAML_Final
./scripts/setup
```

### 4. Build Program and Start System

``` bash
# Automized script for building the project.
# Equivalent to "make build && make load", while fixing multiple definition of a non-constant variable.

# Enter the project directory
cd proj/AAML_final_proj

# Building project with the default model
bash run.sh

# Building project with our custom model
bash run.sh -m "model_compression/final_0.875_qat_model.tflite"

# Verbose mode, for debugging
bash run.sh -v
``````

After the build process is finished, press enter and type 'reboot' to reboot and start the system.

### 5. Run Golden Test

After the program started, type ```11g``` to run test.

### 6. Run Evaluation Test

After the program started, press ctrl+c to leave litex-term, then run the following command:

``` bash
python evaluation.py
```

## Final Results
2sec/img -> 0.75sec/img
