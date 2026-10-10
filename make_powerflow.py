import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import imageio.v2 as imageio
import multiprocessing
import gc

NX, NY = 2500, 5000
DX, DY = 2e-5, 2e-5
NPML = 20
NUM_FRAMES = 60
FPS = 20

sim_mode = 0
if len(sys.argv) > 1:
    sim_mode = int(sys.argv[1])
slit_x_start = 1225
slit_x_end = 1275


# Time calculations
ER = 4.5
W = 0.1e-3
H = 0.1e-3
T_RISE = 10e-12
C0 = 299792458.0
er_eff = (ER + 1.0)/2.0 + ((ER - 1.0)/2.0) * (1.0 / np.sqrt(1.0 + 12.0 * (H / W)))
v_prop = C0 / np.sqrt(er_eff)
DT = 4e-14
dist = (int(NX * 0.8) - 50) * DX
tau = T_RISE / 2.19722
t50 = tau * 0.693147
t_end = t50 + dist / v_prop
max_time_steps = int(np.ceil(t_end / DT))
snapshot_interval = max_time_steps // (NUM_FRAMES - 1)
if snapshot_interval == 0: snapshot_interval = 1

start_x, end_x = NPML, NX - NPML
start_y, end_y = NPML, NY - NPML
x_coords = np.arange(start_x, end_x) * DX * 1e3
y_coords = np.arange(start_y, end_y) * DY * 1e3
X, Y = np.meshgrid(x_coords, y_coords)

trace_width_cells = int(W / DX)
trace_y_start = int((NY - trace_width_cells) / 2)
trace_y_end = trace_y_start + trace_width_cells

# Initialize globals
ez_data, hx_data, hy_data = None, None, None

def render_frame_s(i):
    current_time_ps = (i * snapshot_interval * DT) * 1e12
    total_time_ps = max_time_steps * DT * 1e12
    
    ez_frame = ez_data[i, start_y:end_y, start_x:end_x]
    hx_frame = hx_data[i, start_y:end_y, start_x:end_x]
    hy_frame = hy_data[i, start_y:end_y, start_x:end_x]
    
    Sx = -ez_frame * hy_frame
    Sy = ez_frame * hx_frame
    S_mag = np.sqrt(Sx**2 + Sy**2)
    
    S_mag[S_mag < 1e-20] = 1e-20
    current_max_s = np.max(S_mag)
    if current_max_s < 1e-20: current_max_s = 1e-20
    
    # Power is 10*log10 (not 20*log10 like voltage/field)
    S_db = np.clip(10 * np.log10(S_mag / current_max_s), -60, 0)
    
    time_str = f"Time: {current_time_ps:.2f} ps / {total_time_ps:.2f} ps"
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.suptitle(f"2D FDTD TMz - Poynting Vector (Power Flow)\n{time_str} | Max Power: {current_max_s:.2e} W/m²", fontsize=14, y=0.98)
    
    skip_p = 5
    c = ax.pcolormesh(X[::skip_p, ::skip_p], Y[::skip_p, ::skip_p], S_db[::skip_p, ::skip_p], cmap='jet', shading='auto', vmin=-60, vmax=0)
    fig.colorbar(c, ax=ax, label='Power Flow |S| (dB)')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(start_x * DX * 1e3, end_x * DX * 1e3)
    ax.set_ylim(start_y * DY * 1e3, end_y * DY * 1e3)
    ax.axhline(trace_y_start * DY * 1e3, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(trace_y_end * DY * 1e3, color='gray', linestyle='--', alpha=0.5)
    
    trace_x_start_mm = 50 * DX * 1e3
    trace_x_end_mm = int(NX * 0.8) * DX * 1e3
    ax.axvline(trace_x_start_mm, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(trace_x_end_mm, color='gray', linestyle=':', alpha=0.5)

    if sim_mode == 1:
        ax.axvspan(slit_x_start * DX * 1e3, slit_x_end * DX * 1e3, color='magenta', alpha=0.15, label='Ground Slit')
        ax.legend(loc='lower left')

    
    x_front_mm = trace_x_start_mm + (v_prop * current_time_ps * 1e-12) * 1e3
    if x_front_mm <= end_x * DX * 1e3:
        ax.axvline(x_front_mm, color='lime', linestyle='-', linewidth=2, label='Theoretical Wavefront')
        ax.legend(loc='upper right')
        
    # Quiver plot for direction (downsampled)
    skip = 50
    u = Sx[::skip, ::skip]
    v = Sy[::skip, ::skip]
    norm = np.sqrt(u**2 + v**2)
    norm[norm < 1e-20] = 1.0 # avoid div by zero
    u = u / norm
    v = v / norm
    
    # Only draw arrows where signal is somewhat strong (> -35 dB)
    mask = (S_db[::skip, ::skip] > -35)
    
    if np.any(mask):
        ax.quiver(X[::skip, ::skip][mask], Y[::skip, ::skip][mask], u[mask], v[mask], 
                  color='white', pivot='middle', scale=50, headwidth=4, headlength=5, alpha=0.8)
    
    plt.tight_layout()
    file_s = f"frames_s/frame_{i:03d}.png"
    plt.savefig(file_s, bbox_inches='tight')
    
    fig.clf()
    plt.close('all')
    del fig, ax, c
    gc.collect()
    
    if (i+1) % 5 == 0: print(f"PowerFlow Rendered {i+1}/{NUM_FRAMES}")
    return i

if __name__ == '__main__':
    print("Loading binary data...")
    out_dir = "/scratch/m/mpaltzak/fieldsolver_outputs" if os.path.exists("/scratch/m/mpaltzak/fieldsolver_outputs") else "outputs"
    ez_data = np.memmap(f"{out_dir}/frames_ez_mode_{sim_mode}.bin", dtype=np.float32, mode='r', shape=(NUM_FRAMES, NY, NX))
    hx_data = np.memmap(f"{out_dir}/frames_hx_mode_{sim_mode}.bin", dtype=np.float32, mode='r', shape=(NUM_FRAMES, NY, NX))
    hy_data = np.memmap(f"{out_dir}/frames_hy_mode_{sim_mode}.bin", dtype=np.float32, mode='r', shape=(NUM_FRAMES, NY, NX))
    
    os.makedirs('frames_s', exist_ok=True)
    os.makedirs('video_S', exist_ok=True)
    
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    print(f"Using {num_cores} CPU cores for Multiprocessing Rendering...")
    
    print("Generating PowerFlow frames in parallel...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(render_frame_s, range(NUM_FRAMES))
        
    print("Stitching PowerFlow MP4...")
    writer_s = imageio.get_writer(f'video_S/video_S_mode_{sim_mode}.mp4', fps=FPS, macro_block_size=2)
    for i in range(NUM_FRAMES):
        file_s = f"frames_s/frame_{i:03d}.png"
        im_s = imageio.imread(file_s)
        writer_s.append_data(im_s)
        del im_s
    writer_s.close()
    
    print("Saved video_S/video_S.mp4 successfully!")
