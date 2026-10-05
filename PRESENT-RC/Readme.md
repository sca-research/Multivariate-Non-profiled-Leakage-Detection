# PRESENT-RC trace preparation

This folder contains the dataset configuration and converter used to prepare
the PRESENT randomized-clock traces for the leakage-detection experiments.

## Requirements

Use the repository's supported Python environment and install dependencies from
the repository root:

```bash
python -m pip install -r requirements.txt
```

The converter uses NumPy and PyYAML, which are included in the root
`requirements.txt`.

## Download and extract the dataset

Download `FPGA_PRESENT_RANDOMIZED_CLOCK.zip` from the
[DL-LA dataset link](https://drive.google.com/file/d/1gYerTpnJF_u-BrP5ny0c6rcgbuKQlE4h/view)
and extract it into this directory. The default configuration expects:

```text
PRESENT-RC/
├── FPGA_PRESENT_RANDOMIZED_CLOCK/
│   └── Traces_1.dat
├── DUT.py
└── traces.yml
```

The binary files are not included in this repository. Check that the extracted
directory and file names match `path` and `file_pattern` in `traces.yml`.

## Convert the traces

From the repository root, run:

```bash
python PRESENT-RC/DUT.py
```

The converter validates each binary file against the configured record size
and writes `PRESENT-RC/Traces_PRESENT_RC.npy`. The resulting array has one row
per trace: the first 5,000 columns contain the signed 8-bit sample values and
the last column contains the trace group label (`0` or `1`). NumPy stores the
combined array as signed 16-bit integers to represent both fields, so the
default 100,000-trace output is approximately 1 GB. The converter memory-maps
the output file to avoid holding the full result in RAM; make sure sufficient
disk space is available.

To select another dataset configured in `traces.yml`, or to choose another
output/configuration path, use:

```bash
python PRESENT-RC/DUT.py --dataset FPGA_PRESENT_TI_MISALIGNED
python PRESENT-RC/DUT.py --output /path/to/Traces_PRESENT_RC.npy
python PRESENT-RC/DUT.py --config /path/to/traces.yml
```

The `FPGA_PRESENT_TI_MISALIGNED` configuration contains 50 files and would
produce an output of approximately 50 GB with the current settings.

Relative dataset paths in the YAML file are resolved relative to that file.
Relative output paths are resolved relative to the current working directory.

## Run leakage detection

The experiments load `Traces_PRESENT_RC.npy` from the current working
directory. From the repository root, run them from `PRESENT-RC`:

```bash
cd PRESENT-RC
python ../Code/out_of_the_box_exp.py --exp present_pointwise
python ../Code/out_of_the_box_exp.py --exp present_multivariate
```

The pointwise experiment reproduces Figure 8a. The multivariate experiment
compares the configured multivariate D-test and univariate G-test; its trial
count and enabled tests can be changed in
[`Code/out_of_the_box_exp.py`](../Code/out_of_the_box_exp.py).
