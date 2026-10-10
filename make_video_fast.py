import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Parse mode from argv
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
tau = T_RISE / 2.19722
t50 = tau * 0.693147
trace_x_start = 50
trace_x_end = int(NX * 0.8)
dist = (trace_x_end - trace_x_start) * DX
v_prop = C0 / np.sqrt(er_eff)
TD = dist / v_prop
t_end = t50 + TD
DT = 4e-14
max_time_steps = int(np.ceil(t_end / DT))
snapshot_interval = max_time_steps // (NUM_FRAMES - 1)
if snapshot_interval == 0: snapshot_interval = 1

time_array = np.arange(max_time_steps) * DT
V_AMP = 1.0
pulse_array = (V_AMP / tau) * np.exp(-time_array / tau)
pulse_array_norm = pulse_array / np.max(pulse_array)
time_ps_array = time_array * 1e12

start_x, end_x = NPML, NX - NPML
start_y, end_y = NPML, NY - NPML
x_coords = np.arange(start_x, end_x) * DX * 1e3
y_coords = np.arange(start_y, end_y) * DY * 1e3
X, Y = np.meshgrid(x_coords, y_coords)

trace_width_cells = int(W / DX)
trace_y_start = int((NY - trace_width_cells) / 2)
trace_y_end = trace_y_start + trace_width_cells
trace_y_center = trace_y_start + trace_width_cells // 2
x_1d = x_coords

# Initialize globals to None so they can be populated later
ez_data, h_mag_data = None, None
global_max_e, global_max_h, max_v = 0, 0, 0

def render_frame_e(i):
    current_time_ps = (i * snapshot_interval * DT) * 1e12
    total_time_ps = max_time_steps * DT * 1e12
    
    ez_frame = ez_data[i]
    ez = np.abs(ez_frame[start_y:end_y, start_x:end_x])
    ez[ez < 1e-12] = 1e-12
    current_max_e = np.max(ez)
    if current_max_e < 1e-12: current_max_e = 1e-12
    ez_db = np.clip(20 * np.log10(ez / current_max_e), -60, 0)
    voltage_1d = (ez_frame[trace_y_center, start_x:end_x] * H) * 1000.0
    
    time_str = f"Time: {current_time_ps:.2f} ps / {total_time_ps:.2f} ps"
    
    fig = plt.figure(figsize=(14, 6), dpi=150)
    gs = gridspec.GridSpec(2, 3, width_ratios=[4, 0.1, 1.5], height_ratios=[2.5, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 2])
    
    fig.suptitle(f"2D FDTD TMz - E-Field\n{time_str} | Max Ez: {current_max_e:.2e} V/m", fontsize=14, y=0.98)
    
    skip_p = 5
    c = ax1.pcolormesh(X[::skip_p, ::skip_p], Y[::skip_p, ::skip_p], ez_db[::skip_p, ::skip_p], cmap='jet', shading='auto', vmin=-60, vmax=0)
    fig.colorbar(c, cax=cax, label='Electric Field Ez (dB)')
    ax1.set_ylabel('Y (mm)')
    ax1.axhline(trace_y_start * DY * 1e3, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(trace_y_end * DY * 1e3, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(trace_x_start * DX * 1e3, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(trace_x_end * DX * 1e3, color='gray', linestyle=':', alpha=0.5)

    if sim_mode == 1:
        ax1.axvspan(slit_x_start * DX * 1e3, slit_x_end * DX * 1e3, color='magenta', alpha=0.15, label='Ground Slit')
        ax1.legend(loc='lower left')

    
    x_front_mm = (trace_x_start * DX + v_prop * current_time_ps * 1e-12) * 1e3
    if x_front_mm <= end_x * DX * 1e3:
        ax1.axvline(x_front_mm, color='lime', linestyle='-', linewidth=2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax2.plot(x_1d, voltage_1d, color='blue', linewidth=1.5)
    ax2.fill_between(x_1d, voltage_1d, alpha=0.3, color='blue')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax2.axvline(trace_x_start * DX * 1e3, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(trace_x_end * DX * 1e3, color='gray', linestyle=':', alpha=0.5)

    if sim_mode == 1:
        ax2.axvspan(slit_x_start * DX * 1e3, slit_x_end * DX * 1e3, color='magenta', alpha=0.15)

    ax2.set_ylim(-max_v * 1000 * 1.1, max_v * 1000 * 1.1)
    ax2.set_xlim(start_x * DX * 1e3, end_x * DX * 1e3)
    ax1.set_ylim(start_y * DY * 1e3, end_y * DY * 1e3)
    if x_front_mm <= end_x * DX * 1e3:
        ax2.axvline(x_front_mm, color='lime', linestyle='-', linewidth=2, label='Theoretical Wavefront')
        ax2.legend(loc='upper right')
        
    ax3.plot(time_ps_array, pulse_array_norm, color='orange', linewidth=2)
    ax3.fill_between(time_ps_array, pulse_array_norm, alpha=0.2, color='orange')
    ax3.axvline(current_time_ps, color='red', linestyle='-', linewidth=2)
    ax3.scatter([current_time_ps], [np.interp(current_time_ps, time_ps_array, pulse_array_norm)], color='red', s=50, zorder=5)
    ax3.set_title("Soft Source Injection")
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Normalized Amplitude')
    ax3.set_xlim(0, total_time_ps)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    file_e = f"frames_e/frame_{i:03d}.png"
    plt.savefig(file_e, bbox_inches='tight')
    
    fig.clf()
    plt.close('all')
    del fig, ax1, ax2, ax3, cax, c
    gc.collect()
    
    if (i+1) % 5 == 0: print(f"E-Field Rendered {i+1}/{NUM_FRAMES}")
    return i

def render_frame_h(i):
    current_time_ps = (i * snapshot_interval * DT) * 1e12
    total_time_ps = max_time_steps * DT * 1e12
    
    h_frame = np.sqrt(hx_data[i]**2 + hy_data[i]**2)
    h_cropped = h_frame[start_y:end_y, start_x:end_x]
    h_cropped[h_cropped < 1e-12] = 1e-12
    current_max_h = np.max(h_cropped)
    if current_max_h < 1e-12: current_max_h = 1e-12
    h_db = np.clip(20 * np.log10(h_cropped / current_max_h), -60, 0)
    h_1d = h_frame[trace_y_center, start_x:end_x]
    
    time_str = f"Time: {current_time_ps:.2f} ps / {total_time_ps:.2f} ps"
    
    fig = plt.figure(figsize=(14, 6), dpi=150)
    gs = gridspec.GridSpec(2, 3, width_ratios=[4, 0.1, 1.5], height_ratios=[2.5, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 2])
    
    fig.suptitle(f"2D FDTD TMz - H-Field\n{time_str} | Max |H|: {current_max_h:.2e} A/m", fontsize=14, y=0.98)
    
    skip_p = 5
    c = ax1.pcolormesh(X[::skip_p, ::skip_p], Y[::skip_p, ::skip_p], h_db[::skip_p, ::skip_p], cmap='jet', shading='auto', vmin=-60, vmax=0)
    fig.colorbar(c, cax=cax, label='Magnetic Field |H| (dB)')
    ax1.set_ylabel('Y (mm)')
    ax1.axhline(trace_y_start * DY * 1e3, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(trace_y_end * DY * 1e3, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(trace_x_start * DX * 1e3, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(trace_x_end * DX * 1e3, color='gray', linestyle=':', alpha=0.5)

    if sim_mode == 1:
        ax1.axvspan(slit_x_start * DX * 1e3, slit_x_end * DX * 1e3, color='magenta', alpha=0.15, label='Ground Slit')
        ax1.legend(loc='lower left')

    
    x_front_mm = (trace_x_start * DX + v_prop * current_time_ps * 1e-12) * 1e3
    if x_front_mm <= end_x * DX * 1e3:
        ax1.axvline(x_front_mm, color='lime', linestyle='-', linewidth=2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax2.plot(x_1d, h_1d, color='red', linewidth=1.5)
    ax2.fill_between(x_1d, h_1d, alpha=0.3, color='red')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('|H| (A/m)')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax2.axvline(trace_x_start * DX * 1e3, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(trace_x_end * DX * 1e3, color='gray', linestyle=':', alpha=0.5)

    if sim_mode == 1:
        ax2.axvspan(slit_x_start * DX * 1e3, slit_x_end * DX * 1e3, color='magenta', alpha=0.15)

    ax2.set_ylim(0, global_max_h * 1.1)
    ax2.set_xlim(start_x * DX * 1e3, end_x * DX * 1e3)
    ax1.set_ylim(start_y * DY * 1e3, end_y * DY * 1e3)
    if x_front_mm <= end_x * DX * 1e3:
        ax2.axvline(x_front_mm, color='lime', linestyle='-', linewidth=2, label='Theoretical Wavefront')
        ax2.legend(loc='upper right')
        
    ax3.plot(time_ps_array, pulse_array_norm, color='orange', linewidth=2)
    ax3.fill_between(time_ps_array, pulse_array_norm, alpha=0.2, color='orange')
    ax3.axvline(current_time_ps, color='red', linestyle='-', linewidth=2)
    ax3.scatter([current_time_ps], [np.interp(current_time_ps, time_ps_array, pulse_array_norm)], color='red', s=50, zorder=5)
    ax3.set_title("Soft Source Injection")
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Normalized Amplitude')
    ax3.set_xlim(0, total_time_ps)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    file_h = f"frames_h/frame_{i:03d}.png"
    plt.savefig(file_h, bbox_inches='tight')
    
    fig.clf()
    plt.close('all')
    del fig, ax1, ax2, ax3, cax, c
    gc.collect()
    
    if (i+1) % 5 == 0: print(f"H-Field Rendered {i+1}/{NUM_FRAMES}")
    return i

if __name__ == '__main__':
    print("Loading binary data...")
    out_dir = "/scratch/m/mpaltzak/fieldsolver_outputs" if os.path.exists("/scratch/m/mpaltzak/fieldsolver_outputs") else "outputs"
    ez_data = np.memmap(f"{out_dir}/frames_ez_mode_{sim_mode}.bin", dtype=np.float32, mode='r', shape=(NUM_FRAMES, NY, NX))
    hx_data = np.memmap(f"{out_dir}/frames_hx_mode_{sim_mode}.bin", dtype=np.float32, mode='r', shape=(NUM_FRAMES, NY, NX))
    hy_data = np.memmap(f"{out_dir}/frames_hy_mode_{sim_mode}.bin", dtype=np.float32, mode='r', shape=(NUM_FRAMES, NY, NX))
    
    global_max_e = 0
    global_max_h = 0
    for i in range(NUM_FRAMES):
        global_max_e = max(global_max_e, np.max(np.abs(ez_data[i])))
        h_frame = np.sqrt(hx_data[i]**2 + hy_data[i]**2)
        global_max_h = max(global_max_h, np.max(h_frame))
    if global_max_e == 0: global_max_e = 1e-10
    if global_max_h == 0: global_max_h = 1e-10
    max_v = global_max_e * H
    
    os.makedirs('frames_e', exist_ok=True)
    os.makedirs('frames_h', exist_ok=True)
    os.makedirs('video_E', exist_ok=True)
    os.makedirs('video_H', exist_ok=True)
    
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    print(f"Using {num_cores} CPU cores for Multiprocessing Rendering...")
    
    print("Generating E-Field frames in parallel...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(render_frame_e, range(NUM_FRAMES))
        
    print("Generating H-Field frames in parallel...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(render_frame_h, range(NUM_FRAMES))
        
    print("Stitching E-Field MP4...")
    writer_e = imageio.get_writer(f'video_E/video_E_mode_{sim_mode}.mp4', fps=FPS, macro_block_size=2)
    for i in range(NUM_FRAMES):
        file_e = f"frames_e/frame_{i:03d}.png"
        im_e = imageio.imread(file_e)
        writer_e.append_data(im_e)
        del im_e
    writer_e.close()
    
    print("Stitching H-Field MP4...")
    writer_h = imageio.get_writer(f'video_H/video_H_mode_{sim_mode}.mp4', fps=FPS, macro_block_size=2)
    for i in range(NUM_FRAMES):
        file_h = f"frames_h/frame_{i:03d}.png"
        im_h = imageio.imread(file_h)
        writer_h.append_data(im_h)
        del im_h
    writer_h.close()
    
    print("Saved video_E/video_E.mp4 and video_H/video_H.mp4 successfully!")
