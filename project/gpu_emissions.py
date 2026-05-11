from codecarbon import OfflineEmissionsTracker
import subprocess
import os

CWD = "./"
PERF_DIR = "performance"
BUILD_DIR = "build"
FILE = os.path.join(PERF_DIR, "emissions.csv")

os.makedirs(PERF_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)

def measure_subprocess_emissions(command):
    tracker = OfflineEmissionsTracker(
        country_iso_code="BEL", 
        log_level="error",
        measure_power_secs=1,
        save_to_file=False,
    )
    tracker.start()
    subprocess.run(command)
    tracker.stop()

    return tracker.final_emissions_data

with open(FILE, "w") as file:
    file.write("method,emission,duration_sec,gpu_power_watts,gpu_energy_kwh\n")

    subprocess.run([
        "g++",
        "-DDIM_COUNT=19",
        "-DCL_MUL_METHOD=0",
        "-o", os.path.join(BUILD_DIR, "opencl_matrix_perf_naive"), "src/opencl_matrix_perf.cpp", "src/matrix_opencl.cpp",
        "-std=c++17",
        "-Wall", "-Wextra", 
        "-O2", 
        "-Iinclude", "-lOpenCL",
    ])
    
    cmd_naive = [os.path.join(CWD, BUILD_DIR, "opencl_matrix_perf_naive"), "--no-output"]
    data = measure_subprocess_emissions(cmd_naive)

    file.write(f"opencl_naive,{data.emissions},{data.duration},{data.gpu_power},{data.gpu_energy}\n")

    for tile in (4, 8, 16):
        subprocess.run([
            "g++",
            "-DDIM_COUNT=19",
            "-DCL_MUL_METHOD=1", f"-DTILE_SIZE={tile}",
            "-o", os.path.join(BUILD_DIR, f"opencl_matrix_perf_tiled_{tile}"), "src/opencl_matrix_perf.cpp", "src/matrix_opencl.cpp",
            "-std=c++17",
            "-Wall", "-Wextra", 
            "-O2", 
            "-Iinclude", "-lOpenCL",
        ])

        cmd_tiled = [os.path.join(CWD, BUILD_DIR, f"opencl_matrix_perf_tiled_{tile}"), "--no-output"]
        data_tiled = measure_subprocess_emissions(cmd_tiled)
        file.write(f"opencl_tiled_{tile},{data_tiled.emissions},{data_tiled.duration},{data_tiled.gpu_power},{data_tiled.gpu_energy}\n")
