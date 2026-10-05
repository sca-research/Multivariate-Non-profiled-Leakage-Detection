import argparse
import ast
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "traces.yml"


def convert_traces(dataset_name, config_path=DEFAULT_CONFIG, output_path=None):
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        datasets = yaml.safe_load(config_file)

    if not isinstance(datasets, dict):
        raise ValueError(f"Expected a mapping of datasets in {config_path}")
    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets))
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; choose one of: {available}"
        )

    settings = datasets[dataset_name]
    if not isinstance(settings, dict):
        raise ValueError(f"Dataset {dataset_name!r} must have mapping settings")

    try:
        directory = Path(settings["path"])
        file_count = int(settings["nrfiles"])
        traces_per_file = int(settings["tracesinfile"])
        trace_length = int(settings["tracelen"])
        record_dtype = np.dtype(ast.literal_eval(settings["struct"]))
    except (KeyError, TypeError, ValueError, SyntaxError) as error:
        raise ValueError(
            f"Invalid configuration for dataset {dataset_name!r}"
        ) from error

    if file_count < 1 or traces_per_file < 1 or trace_length < 1:
        raise ValueError(
            "File count, traces per file, and trace length must be positive"
        )

    fields = record_dtype.fields or {}
    if "trace" not in fields or "group" not in fields:
        raise ValueError("The record structure must define 'trace' and 'group' fields")
    trace_field_dtype = fields["trace"][0]
    group_field_dtype = fields["group"][0]
    if trace_field_dtype.shape != (trace_length,):
        raise ValueError(
            f"The configured trace field must contain {trace_length} samples"
        )
    if group_field_dtype.shape:
        raise ValueError("The configured group field must be a scalar")

    if not directory.is_absolute():
        directory = config_path.parent / directory
    filename_pattern = settings.get("file_pattern", "Traces_{file_number}.dat")
    expected_size = traces_per_file * record_dtype.itemsize
    trace_files = []
    for file_number in range(1, file_count + 1):
        trace_file = directory / filename_pattern.format(file_number=file_number)
        actual_size = trace_file.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"{trace_file} has {actual_size} bytes; expected {expected_size} "
                f"({traces_per_file} records)"
            )
        trace_files.append(trace_file)

    if output_path is None:
        output_path = SCRIPT_DIR / "Traces_PRESENT_RC.npy"
    output_path = Path(output_path)
    if output_path.suffix != ".npy":
        output_path = Path(f"{output_path}.npy")

    result_dtype = np.result_type(trace_field_dtype.base, group_field_dtype)
    traces = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=result_dtype,
        shape=(file_count * traces_per_file, trace_length + 1),
    )
    for file_number, trace_file in enumerate(trace_files, start=1):
        records = np.fromfile(trace_file, dtype=record_dtype)
        if records.size != traces_per_file:
            raise ValueError(
                f"Could not read {traces_per_file} records from {trace_file}"
            )
        if not np.isin(records["group"], (0, 1)).all():
            raise ValueError(f"{trace_file} contains group labels other than 0 and 1")

        start = (file_number - 1) * traces_per_file
        stop = start + traces_per_file
        traces[start:stop, :-1] = records["trace"]
        traces[start:stop, -1] = records["group"]

    traces.flush()
    del traces
    print(
        f"Saved {file_count * traces_per_file} traces "
        f"({trace_length} samples + group) to {output_path}"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert configured PRESENT-RC binary traces to a NumPy array."
    )
    parser.add_argument(
        "--dataset",
        default="FPGA_PRESENT_RANDOMIZED_CLOCK",
        help="Dataset key from traces.yml (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Path to the dataset configuration "
            "(default: traces.yml beside this script)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path (default: Traces_PRESENT_RC.npy beside this script)",
    )
    args = parser.parse_args()
    convert_traces(args.dataset, args.config, args.output)


if __name__ == "__main__":
    main()
