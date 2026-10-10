# 2D FDTD TMz Solver - Microstrip Line & Ground Slit

This directory contains the final codebase for the 2D FDTD Electromagnetic simulation, utilizing NVIDIA CUDA for GPU acceleration and Python for generating high-quality MP4 visualizations.

## Directory Structure
- `video_pml.cu`: The core C++ CUDA FDTD solver. Computes the TMz fields and outputs binary `.bin` files.
- `make_video_fast.py` & `make_powerflow.py`: Python scripts that read the `.bin` files and generate video animations (Electric Field, Magnetic Field, Poynting Vector).
- `run_all.sh`: Bash script that orchestrates the execution of both Python scripts.
- `submit.sh` & `run_cluster.sh`: SLURM scripts for cluster execution.
- `outputs/` & `errors/`: Directories where SLURM execution logs are automatically saved with sequential numbering (e.g., `output_1.txt`, `error_1.txt`).
- `video_E/`, `video_H/`, `video_S/`: Directories where the final MP4 videos are saved.

---

## 1. Running on a SLURM Cluster

To run the full simulation and generate videos on a cluster with SLURM, simply execute:
```bash
./submit.sh
```
By default, this runs **Mode 0** (Continuous Microstrip).
If you want to run **Mode 1** (Ground Slit), execute:
```bash
./submit.sh video_pml.cu run_all.sh 1
```

**What happens automatically:**
1. The script finds the next available ID (e.g., 15) and creates `outputs/output_15.txt` and `errors/error_15.txt`.
2. It compiles the CUDA code and submits the job.
3. The binary data is temporarily written to a fast `/scratch` directory.
4. Python multiprocesses the binary data to create individual frames.
5. `ffmpeg` stitches the frames into `.mp4` videos.
6. The final videos are saved in the `video_E/`, `video_H/`, and `video_S/` folders.

---

## 2. Running Locally (Without SLURM)

If you have a local machine with an NVIDIA GPU and wish to run the simulation directly, follow these steps:

### Prerequisites:
- NVIDIA CUDA Toolkit (`nvcc` compiler)
- Python 3 with the following libraries:
  ```bash
  pip install numpy matplotlib imageio imageio[ffmpeg]
  ```

### Step-by-Step Execution:

**1. Compile the CUDA solver:**
```bash
nvcc video_pml.cu -o video_pml -O3
```

**2. Run the solver (Mode 0):**
```bash
./video_pml 0
```
*(This will compute the electromagnetic fields and save `.bin` files in the `outputs/` folder or local directory).*

**3. Generate the visualizations:**
Run the orchestration script:
```bash
bash run_all.sh 0
```
*(Alternatively, run the python scripts manually: `python3 make_video_fast.py 0` and `python3 make_powerflow.py 0`).*

Once completed, you can find your MP4 files in the `video_E/`, `video_H/`, and `video_S/` directories.
