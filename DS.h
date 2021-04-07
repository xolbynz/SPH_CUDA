#ifndef DS_H
#define DS_H

#define uint unsigned int
#define DIM 3
#define hTd cudaMemcpyHostToDevice
#define dTh cudaMemcpyDeviceToHost

typedef struct sim_param_t {
	char* fname;
	uint nframes;
	uint npframe;
	float h;
	float dt;
	float rho0;
	float k;
	float mu;
	float g;
} sim_param_t;

typedef struct sim_state_t {
	uint n;
	float mass;
	float* __restrict rho;
	float* __restrict x;
	float* __restrict vh;
	float* __restrict v;
	float* __restrict a;
} sim_state_t;

typedef struct device_info {
	uint nloop;
	uint nThreads;
	uint nBlocks;
	uint n;
	float mass;

} device_info;

#endif