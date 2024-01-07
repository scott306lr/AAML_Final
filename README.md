# CFU Playground - AAML Final Project - Group 4

## Setup Guide

### 1. Prepare a supperted board (We use Nexys A7-100T) and install required toolchains. See [Setup Guide](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html) for more details

### 2. Clone the CFU-Playground Repository from the github

``` bash
git clone https://github.com/scott306lr/AAML_Final.git
```

### 3. Run the setup script

``` bash
cd AAML_Final
./scripts/setup
```

### 4. Enter project directory and run the build script

After the build process is finished, press enter and type 'reboot' to reboot and start the system.

``` bash
# Automized script for building the project.
# Equivalent to "make build && make load", while fixing multiple definition of a non-constant variable.

# enter the project directory
cd proj/AAML_final_proj

# building project with the default model
bash run.sh

# building project with our custom model
bash run.sh -m "../../model_compression/final_0.875_qat_model.tflite"

# verbose mode, for debugging
bash run.sh -v
``````

### 5. Run Golden Test

Using keyboard, type ```1, 1, g``` after the program started.

### 6. Run Evaluation Test

After the program started, press ctrl+c to stop the program, then run the following command:

``` bash
python3 evaluation.py
```

## Some Test Results

...
