# from codecarbon import OfflineEmissionsTracker
# import subprocess
# import os


# CWD = "./"
# PERF_DIR = "performance"
# BUILD_DIR = "build"
# FILE = os.path.join(PERF_DIR, "emissions.csv")

# os.makedirs(PERF_DIR, exist_ok=True)
# os.makedirs(BUILD_DIR, exist_ok=True)
# with open(FILE, "w") as file:
#     file.write("method,emission\n")

#     tracker = OfflineEmissionsTracker(country_iso_code="BEL", log_level="error")

#     subprocess.run([
#         "g++",
#         "-DCL_MUL_METHOD=0",
#         "-o", BUILD_DIR + "/opencl_matrix_perf_naive", "src/opencl_matrix_perf.cpp", "src/matrix_opencl.cpp",
#         "-std=c++17",
#         "-Wall", "-Wextra", 
#         "-O2", 
#         "-Iinclude", "-lOpenCL",
#     ])

#     tracker.start()
#     subprocess.run([CWD + "/" + BUILD_DIR + "/opencl_matrix_perf_naive", "--no-output"])
#     emissions = tracker.stop()
#     file.write(f"opencl_naive,{emissions}\n")

#     for tile in (4, 8, 16):
#         subprocess.run([
#             "g++",
#             "-DCL_MUL_METHOD=1", f"-DTILE_SIZE={tile}",
#             "-o", BUILD_DIR + f"/opencl_matrix_perf_tiled_{tile}", "src/opencl_matrix_perf.cpp", "src/matrix_opencl.cpp",
#             "-std=c++17",
#             "-Wall", "-Wextra", 
#             "-O2", 
#             "-Iinclude", "-lOpenCL",
#         ])

#         tracker.start()
#         subprocess.run([CWD + "/" + BUILD_DIR + f"/opencl_matrix_perf_tiled_{tile}", "--no-output"])
#         emissions = tracker.stop()
#         file.write(f"opencl_tiled_{tile},{emissions}\n")

from codecarbon import OfflineEmissionsTracker
import subprocess
import os

CWD = "./"
PERF_DIR = "performance"
BUILD_DIR = "build"
FILE = os.path.join(PERF_DIR, "emissions.csv")

os.makedirs(PERF_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)

# Helper function to ensure we get a completely fresh tracker and 
# isolated measurement for every single execution.
def measure_subprocess_emissions(command):
    # measure_power_secs=1 forces codecarbon to sample every 1 second 
    # instead of the default 15 seconds, crucial for fast C++ jobs.
    tracker = OfflineEmissionsTracker(
        country_iso_code="BEL", 
        log_level="error",
        measure_power_secs=1 
    )
    tracker.start()
    subprocess.run(command)
    return tracker.stop()

with open(FILE, "w") as file:
    file.write("method,emission\n")

    # --- 1. NAIVE ---
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
    emissions_naive = measure_subprocess_emissions(cmd_naive)
    file.write(f"opencl_naive,{emissions_naive}\n")

    # --- 2. TILED ---
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
        emissions_tiled = measure_subprocess_emissions(cmd_tiled)
        file.write(f"opencl_tiled_{tile},{emissions_tiled}\n")