from codecarbon import OfflineEmissionsTracker
import subprocess
import os


PERF_DIR = "performance"
FILE = os.path.join(PERF_DIR, "emissions.csv")

os.makedirs(PERF_DIR, exist_ok=True)
with open(FILE, "w") as file:
    file.write("method,emission\n")

    tracker = OfflineEmissionsTracker(country_iso_code="BEL", log_level="error")

    subprocess.run([
        "g++",
        "-DCL_MUL_METHOD=0",
        "-o", "opencl_matrix_perf_naive", "src/opencl_matrix_perf.cpp", "src/matrix_opencl.cpp",
        "-std=c++17",
        "-Wall", "-Wextra", 
        "-O2", 
        "-Iinclude", "-lOpenCL",
    ])

    tracker.start()
    subprocess.run(["./opencl_matrix_perf_naive"])
    emissions = tracker.stop()
    file.write(f"opencl_naive,{emissions}\n")

    for tile in (4, 8, 16):
        subprocess.run([
            "g++",
            "-DCL_MUL_METHOD=1", f"-DTILE_SIZE={tile}",
            "-o", f"opencl_matrix_perf_tiled_{tile}", "src/opencl_matrix_perf.cpp", "src/matrix_opencl.cpp",
            "-std=c++17",
            "-Wall", "-Wextra", 
            "-O2", 
            "-Iinclude", "-lOpenCL",
        ])

        tracker.start()
        subprocess.run([f"./opencl_matrix_perf_tiled_{tile}"])
        emissions = tracker.stop()
        file.write(f"opencl_tiled_{tile},{emissions}\n")
