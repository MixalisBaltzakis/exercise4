#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

// --- SIMULATION PARAMETERS ---
const int NX = 2500;              // Cells in X direction
const int NY = 5000;               // Cells in Y direction
const double DX = 2e-5;           // Cell size in X (10 um)
const double DY = 2e-5;           // Cell size in Y (10 um)

const double C0 = 299792458.0;    
const double MU0 = 4.0 * M_PI * 1e-7;
const double EPS0 = 1.0 / (C0 * C0 * MU0);

const double DT = 4e-14;          
const int NPML = 20;              // PML width in cells

// Trace Parameters
const double W = 0.1e-3;          
const double H = 0.1e-3;          
const double ER = 4.5;            

// Step Pulse Parameters
const double V_AMP = 1.0;         // 1 Volt amplitude
const double T_RISE = 20e-12;     // 10-90% Rise Time = 20 ps

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// --- CUDA KERNELS ---
__global__ void update_H(float* hx, float* hy, const float* ez, 
                         float* icex, float* icey,
                         const float* mhx1, const float* mhx2, const float* mhx3,
                         const float* mhy1, const float* mhy2, const float* mhy3,
                         int nx, int ny, float dx, float dy) {
    __shared__ float s_ez[33][33];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    if (x < nx && y < ny) {
        s_ez[ty][tx] = ez[y * nx + x];
    } else {
        s_ez[ty][tx] = 0.0f;
    }

    if (tx == blockDim.x - 1 && x + 1 < nx && y < ny) {
        s_ez[ty][tx + 1] = ez[y * nx + (x + 1)];
    }
    if (ty == blockDim.y - 1 && x < nx && y + 1 < ny) {
        s_ez[ty + 1][tx] = ez[(y + 1) * nx + x];
    }

    __syncthreads();

    if (x < nx && y < ny - 1) {
        int idx = y * nx + x;
        float cex = (s_ez[ty + 1][tx] - s_ez[ty][tx]) / dy;
        icex[idx] += cex;
        hx[idx] = mhx1[idx] * hx[idx] + mhx2[idx] * cex + mhx3[idx] * icex[idx];
    }

    if (x < nx - 1 && y < ny) {
        int idx = y * nx + x;
        float cey = -(s_ez[ty][tx + 1] - s_ez[ty][tx]) / dx;
        icey[idx] += cey;
        hy[idx] = mhy1[idx] * hy[idx] + mhy2[idx] * cey + mhy3[idx] * icey[idx];
    }
}

__global__ void update_E(float* dz, float* ez, const float* hx, const float* hy,
                         float* idz, 
                         const float* mdz1, const float* mdz2, const float* mdz4,
                         const float* mez1,
                         int nx, int ny, float dx, float dy) {
    __shared__ float s_hx[33][33];
    __shared__ float s_hy[33][33];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    if (x < nx && y < ny) {
        s_hx[ty + 1][tx + 1] = hx[y * nx + x];
        s_hy[ty + 1][tx + 1] = hy[y * nx + x];
    } else {
        s_hx[ty + 1][tx + 1] = 0.0f;
        s_hy[ty + 1][tx + 1] = 0.0f;
    }

    if (tx == 0 && x > 0 && y < ny) {
        s_hx[ty + 1][0] = hx[y * nx + (x - 1)];
        s_hy[ty + 1][0] = hy[y * nx + (x - 1)];
    }
    if (ty == 0 && y > 0 && x < nx) {
        s_hx[0][tx + 1] = hx[(y - 1) * nx + x];
        s_hy[0][tx + 1] = hy[(y - 1) * nx + x];
    }

    __syncthreads();

    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        int idx = y * nx + x;
        float chz = (s_hy[ty + 1][tx + 1] - s_hy[ty + 1][tx]) / dx - 
                    (s_hx[ty + 1][tx + 1] - s_hx[ty][tx + 1]) / dy;
        
        float old_dz = dz[idx];
        dz[idx] = mdz1[idx] * old_dz + mdz2[idx] * chz + mdz4[idx] * idz[idx];
        idz[idx] += old_dz;
        
        ez[idx] = mez1[idx] * dz[idx];
    }
}

__global__ void inject_source(float* dz, float* ez, const float* mez1, int src_x, int src_y, float value, int nx, int ny) {
    int idx = src_y * nx + src_x;
    dz[idx] += value; 
    ez[idx] = mez1[idx] * dz[idx];
}

int main(int argc, char* argv[]) {
    int sim_mode = 0;
    if (argc > 1) {
        sim_mode = std::atoi(argv[1]);
    }
    std::cout << "Simulation Mode: " << (sim_mode == 1 ? "GROUND SLIT" : "MICROSTRIP") << std::endl;

    std::cout << "--- 2D FDTD TMz Mode Solver (PML D-E Formulation) ---" << std::endl;

    double er_eff = (ER + 1.0)/2.0 + ((ER - 1.0)/2.0) * (1.0 / std::sqrt(1.0 + 12.0 * (H / W)));
    
    int trace_width_cells = std::round(W / DX);
    int trace_y_start = (NY - trace_width_cells) / 2;
    int trace_y_end = trace_y_start + trace_width_cells;
    int trace_x_start = 50;    
    int trace_x_end = (int)(NX * 0.8); 
    
    // Dynamic Time Calculation for an RC exponential step
    double tau = T_RISE / 2.19722; 
    double t50 = tau * 0.693147; 
    
    double v_prop = C0 / std::sqrt(er_eff);
    double dist = (trace_x_end - trace_x_start) * DX;
    double TD = dist / v_prop;
    
    // Snapshot exactly when 50% point hits far end
    double t_end = t50 + TD;
    int max_time_steps = std::ceil(t_end / DT);

    std::ofstream param_file("sim_params.txt");
    param_file << "TD_ps=" << TD * 1e12 << "\n";
    param_file << "TRISE_ps=" << T_RISE * 1e12 << "\n";
    param_file << "TAU_ps=" << tau * 1e12 << "\n";
    param_file << "DT_ps=" << DT * 1e12 << "\n";
    param_file << "MAX_STEPS=" << max_time_steps << "\n";
    param_file << "VAMP=" << V_AMP << "\n";
    param_file.close();

    int num_cells = NX * NY;
    size_t size_bytes = num_cells * sizeof(float);

    std::vector<float> h_ez(num_cells, 0.0f), h_hx(num_cells, 0.0f), h_hy(num_cells, 0.0f), h_dz(num_cells, 0.0f);
    std::vector<float> h_icex(num_cells, 0.0f), h_icey(num_cells, 0.0f), h_idz(num_cells, 0.0f);
    
    std::vector<float> mhx1(num_cells, 0.0f), mhx2(num_cells, 0.0f), mhx3(num_cells, 0.0f);
    std::vector<float> mhy1(num_cells, 0.0f), mhy2(num_cells, 0.0f), mhy3(num_cells, 0.0f);
    std::vector<float> mdz1(num_cells, 0.0f), mdz2(num_cells, 0.0f), mdz4(num_cells, 0.0f);
    std::vector<float> mez1(num_cells, 0.0f);

    std::vector<float> sigx_E(NX, 0.0f), sigx_H(NX, 0.0f);
    std::vector<float> sigy_E(NY, 0.0f), sigy_H(NY, 0.0f);
    
    float sig_max_x = EPS0 / (2.0f * DT);
    float sig_max_y = EPS0 / (2.0f * DT);
    
    for (int i = 0; i < NX; i++) {
        float dist_E = 0, dist_H = 0;
        if (i < NPML) { dist_E = NPML - i; dist_H = NPML - i - 0.5f; }
        else if (i > NX - 1 - NPML) { dist_E = i - (NX - 1 - NPML); dist_H = i + 0.5f - (NX - 1 - NPML); }
        sigx_E[i] = (dist_E > 0) ? sig_max_x * std::pow(dist_E / NPML, 3.0f) : 0.0f;
        sigx_H[i] = (dist_H > 0) ? sig_max_x * std::pow(dist_H / NPML, 3.0f) : 0.0f;
    }
    for (int j = 0; j < NY; j++) {
        float dist_E = 0, dist_H = 0;
        if (j < NPML) { dist_E = NPML - j; dist_H = NPML - j - 0.5f; }
        else if (j > NY - 1 - NPML) { dist_E = j - (NY - 1 - NPML); dist_H = j + 0.5f - (NY - 1 - NPML); }
        sigy_E[j] = (dist_E > 0) ? sig_max_y * std::pow(dist_E / NPML, 3.0f) : 0.0f;
        sigy_H[j] = (dist_H > 0) ? sig_max_y * std::pow(dist_H / NPML, 3.0f) : 0.0f;
    }

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            float sx_hx = sigx_E[x], sy_hx = sigy_H[y];
            float mhx0 = 1.0f/DT + sy_hx / (2.0f*EPS0);
            mhx1[idx] = (1.0f/DT - sy_hx / (2.0f*EPS0)) / mhx0;
            mhx2[idx] = -(C0 / 1.0f) / mhx0;
            mhx3[idx] = -(C0 * DT * sx_hx / EPS0) / mhx0;

            float sx_hy = sigx_H[x], sy_hy = sigy_E[y];
            float mhy0 = 1.0f/DT + sx_hy / (2.0f*EPS0);
            mhy1[idx] = (1.0f/DT - sx_hy / (2.0f*EPS0)) / mhy0;
            mhy2[idx] = -(C0 / 1.0f) / mhy0;
            mhy3[idx] = -(C0 * DT * sy_hy / EPS0) / mhy0;

            float sx_dz = sigx_E[x], sy_dz = sigy_E[y];
            float mdz0 = 1.0f/DT + (sx_dz + sy_dz)/(2.0f*EPS0) + (sx_dz * sy_dz * DT)/(4.0f*EPS0*EPS0);
            mdz1[idx] = (1.0f/DT - (sx_dz + sy_dz)/(2.0f*EPS0) - (sx_dz * sy_dz * DT)/(4.0f*EPS0*EPS0)) / mdz0;
            mdz2[idx] = C0 / mdz0;
            mdz4[idx] = -(DT / (EPS0*EPS0) * sx_dz * sy_dz) / mdz0;

            float local_er = 1.0f;
            if (y >= trace_y_start && y <= trace_y_end && x >= trace_x_start && x <= trace_x_end) {
                local_er = er_eff;
            }
            if (sim_mode == 1 && x >= 1225 && x <= 1275) {
                local_er = 1.0f;
            }
            mez1[idx] = 1.0f / local_er;
        }
    }

    float *d_ez, *d_hx, *d_hy, *d_dz, *d_icex, *d_icey, *d_idz;
    float *d_mhx1, *d_mhx2, *d_mhx3, *d_mhy1, *d_mhy2, *d_mhy3, *d_mdz1, *d_mdz2, *d_mdz4, *d_mez1;
    CUDA_CHECK(cudaMalloc(&d_ez, size_bytes)); CUDA_CHECK(cudaMalloc(&d_hx, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_hy, size_bytes)); CUDA_CHECK(cudaMalloc(&d_dz, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_icex, size_bytes)); CUDA_CHECK(cudaMalloc(&d_icey, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_idz, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_mhx1, size_bytes)); CUDA_CHECK(cudaMalloc(&d_mhx2, size_bytes)); CUDA_CHECK(cudaMalloc(&d_mhx3, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_mhy1, size_bytes)); CUDA_CHECK(cudaMalloc(&d_mhy2, size_bytes)); CUDA_CHECK(cudaMalloc(&d_mhy3, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_mdz1, size_bytes)); CUDA_CHECK(cudaMalloc(&d_mdz2, size_bytes)); CUDA_CHECK(cudaMalloc(&d_mdz4, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_mez1, size_bytes));

    CUDA_CHECK(cudaMemcpy(d_ez, h_ez.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hx, h_hx.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hy, h_hy.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dz, h_dz.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_icex, h_icex.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_icey, h_icey.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idz, h_idz.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mhx1, mhx1.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mhx2, mhx2.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mhx3, mhx3.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mhy1, mhy1.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mhy2, mhy2.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mhy3, mhy3.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mdz1, mdz1.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mdz2, mdz2.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mdz4, mdz4.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mez1, mez1.data(), size_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    const int NUM_FRAMES = 60;
    int snapshot_interval = max_time_steps / (NUM_FRAMES - 1);
    if(snapshot_interval == 0) snapshot_interval = 1;
    
    char name_ez[256], name_hx[256], name_hy[256];
    const char* out_dir = "/scratch/m/mpaltzak/fieldsolver_outputs";
    FILE* test_dir = fopen("/scratch/m/mpaltzak/fieldsolver_outputs/test.tmp", "w");
    if (test_dir) {
        fclose(test_dir);
        remove("/scratch/m/mpaltzak/fieldsolver_outputs/test.tmp");
    } else {
        out_dir = "outputs";
    }
    
    sprintf(name_ez, "%s/frames_ez_mode_%d.bin", out_dir, sim_mode);
    sprintf(name_hx, "%s/frames_hx_mode_%d.bin", out_dir, sim_mode);
    sprintf(name_hy, "%s/frames_hy_mode_%d.bin", out_dir, sim_mode);
    FILE* f_ez = fopen(name_ez, "wb");
    FILE* f_hx = fopen(name_hx, "wb");
    FILE* f_hy = fopen(name_hy, "wb");
    
    int frame_count = 0;

    for (int t = 0; t < max_time_steps; ++t) {
        double current_time = t * DT;
        float v_deriv = (V_AMP / tau) * std::exp(-current_time / tau);
        float dz_inject_val = v_deriv * DT * er_eff;

        update_H<<<numBlocks, threadsPerBlock>>>(d_hx, d_hy, d_ez, d_icex, d_icey, 
                                                 d_mhx1, d_mhx2, d_mhx3, 
                                                 d_mhy1, d_mhy2, d_mhy3, 
                                                 NX, NY, DX, DY);
        update_E<<<numBlocks, threadsPerBlock>>>(d_dz, d_ez, d_hx, d_hy, d_idz, 
                                                 d_mdz1, d_mdz2, d_mdz4, d_mez1, 
                                                 NX, NY, DX, DY);
        
        inject_source<<<1, 1>>>(d_dz, d_ez, d_mez1, trace_x_start, trace_y_start + trace_width_cells / 2, dz_inject_val, NX, NY);
        
        if (t % snapshot_interval == 0 && frame_count < NUM_FRAMES) {
            CUDA_CHECK(cudaMemcpy(h_ez.data(), d_ez, size_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_hx.data(), d_hx, size_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_hy.data(), d_hy, size_bytes, cudaMemcpyDeviceToHost));
            
            if (f_ez) fwrite(h_ez.data(), sizeof(float), NX * NY, f_ez);
            if (f_hx) fwrite(h_hx.data(), sizeof(float), NX * NY, f_hx);
            if (f_hy) fwrite(h_hy.data(), sizeof(float), NX * NY, f_hy);
            
            frame_count++;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Time loop executed in: " << milliseconds << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_ez.data(), d_ez, size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hx.data(), d_hx, size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hy.data(), d_hy, size_bytes, cudaMemcpyDeviceToHost));


    if (f_ez) fclose(f_ez); 
    if (f_hx) fclose(f_hx); 
    if (f_hy) fclose(f_hy);
    cudaFree(d_mhx1); cudaFree(d_mhx2); cudaFree(d_mhx3);
    cudaFree(d_mhy1); cudaFree(d_mhy2); cudaFree(d_mhy3);
    cudaFree(d_mdz1); cudaFree(d_mdz2); cudaFree(d_mdz4); cudaFree(d_mez1);
    return 0;
}
