#include "DS.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#include <math.h>

sim_state_t* alloc_state(uint n)
{
	sim_state_t *state = (sim_state_t*)malloc(sizeof(sim_state_t));

	state->n = n;
	state->rho = (float*)malloc(sizeof(float)*n);
	state->x = (float*)malloc(sizeof(float)*n*DIM);
	state->v = (float*)malloc(sizeof(float)*n * DIM);
	state->vh = (float*)malloc(sizeof(float)*n * DIM);
	state->a = (float*)malloc(sizeof(float)*n * DIM);
	return state;
}

void free_state(sim_state_t* s)
{
	free(s->rho);
	free(s->x);
	free(s->v);
	free(s->vh);
	free(s->a);
	free(s);
}

void compute_density(
	sim_state_t* s,
	sim_param_t* params
	)
{
	uint n = s->n;
	float* __restrict rho = s->rho;
	const float* __restrict x = s->x;

	float h = params->h;
	float h2 = h*h;
	float h8 = (h2*h2)*(h2*h2);
	float C = 4.f * s->mass / M_PI / h8;

	memset(rho, 0, n*sizeof(float));

	for (uint i = 0; i < n; ++i) {
		rho[i] += 4 * s->mass / M_PI / h2;

		for (uint j = i + 1; j < n; ++j) {
			float dx = x[DIM * i + 0] - x[DIM * j + 0];
			float dy = x[DIM * i + 1] - x[DIM * j + 1];
			float dz = x[DIM * i + 2] - x[DIM * j + 2];
			float r2 = dx*dx + dy*dy + dz*dz;
			float z = h2 - r2;
			if (z > 0) {
				float rho_ij = C*z*z*z;
				rho[i] += rho_ij;
				rho[j] += rho_ij;
			}
		}
	}
}

__global__
void d_compute_density(
		float* d_rho,
		float* d_x,
		sim_param_t* d_params,
		device_info* d_info
		)
{
	uint lo_indx = (uint)(blockDim.x * blockIdx.x + threadIdx.x);

	float mass = d_info->mass;
	float h = d_params->h;
	float h2 = h*h;

	uint n = d_info->n;
	uint n2 = n*n;
	uint nloop = d_info->nloop;
	uint nThreads = d_info->nThreads;
	uint nBlocks = d_info->nBlocks;
	uint Tthreads = nThreads*nBlocks;
	uint indx = lo_indx + nloop*Tthreads;


	if (indx < n2) {
		float h8 = h2*h2*h2*h2;
		float C = 4.f * mass / M_PI / h8;
		d_rho[indx] = 0.f;

		uint i, j;

		i = (uint)(indx / n);
		j = (uint)(indx % n);

		if (i == j) {
			d_rho[indx] = 4 * mass / M_PI / h2;
		}
		else if (i<j) {

			float dx = d_x[DIM * i + 0] - d_x[DIM * j + 0];
			float dy = d_x[DIM * i + 1] - d_x[DIM * j + 1];
			float dz = d_x[DIM * i + 2] - d_x[DIM * j + 2];

			float r2 = dx*dx + dy*dy + dz*dz;
			float z = h2 - r2;

			if (z > 0) {
				float rho_ij = C*z*z*z;
				d_rho[indx] = rho_ij;
			}
		}
	}
}

void par_compute_density(
	sim_state_t* s,
	sim_param_t* params
	)
{
	uint nBlocks, nThreads;
	nBlocks = 100;
	nThreads = 64;

	uint n = s->n;
	uint n2 = n*n;
	float* __restrict rho = s->rho;
	const float* __restrict x = s->x;

	float h = params->h;
	float h2 = h*h;
	float h8 = (h2*h2)*(h2*h2);
	float C = 4.f * s->mass / M_PI / h8;

	memset(rho, 0, n*sizeof(float));
	
	float* h_rho;
	h_rho = (float*)malloc(sizeof(float)*n2);

	float* d_rho;
	cudaMalloc((void**)&d_rho, sizeof(float)*n2);

	float* d_x;
	cudaMalloc((void**)&d_x, sizeof(float)*n*DIM);
	cudaMemcpy(d_x, x, sizeof(float)*n*DIM, hTd);

	sim_param_t* d_param;
	cudaMalloc((void**)&d_param, sizeof(sim_param_t));
	cudaMemcpy(d_param, params, sizeof(sim_param_t), hTd);

	device_info *h_info;
	h_info = (device_info*)malloc(sizeof(device_info));

	h_info->mass = s->mass;
	h_info->n = n;
	h_info->nBlocks = nBlocks;
	h_info->nThreads = nThreads;

	uint Tloop = n2 / nBlocks / nThreads + 1;

	for (uint nloop = 0; nloop < Tloop; ++nloop) {
		h_info->nloop = nloop;

		device_info* d_info;
		cudaMalloc((void**)&d_info, sizeof(device_info));
		cudaMemcpy(d_info, h_info, sizeof(device_info), hTd);

		d_compute_density <<< nBlocks, nThreads >>>(
			d_rho,
			d_x,
			d_param,
			d_info
			);

		cudaFree(d_info);
	}

	cudaMemcpy(h_rho, d_rho, sizeof(float)*n2, dTh);

	omp_set_num_threads(8);
	uint idx = 0;
#pragma omp parallel for private(idx)
	for (idx = 0; idx < n2; idx++) {
		uint i, j;

		i = (uint)(idx / n);
		j = (uint)(idx % n);
	
		if (i == j)
		{
			rho[i] += h_rho[idx];
		}
		else if (i < j)
		{
			rho[i] += h_rho[idx];
			rho[j] += h_rho[idx];
		}
	}

	delete[] h_info;
	delete[] h_rho;
	cudaFree(d_rho);
	cudaFree(d_x);
	cudaFree(d_param);
}


void compute_accel(
	sim_state_t* state,
	sim_param_t* params
	)
{
	// Unpack basic parameters
	const float h = params->h;
	const float rho0 = params->rho0;
	const float k = params->k;
	const float mu = params->mu;
	const float g = params->g;
	const float mass = state->mass;
	const float h2 = h*h;

	// Unpack system state
	const float* __restrict rho = state->rho;
	const float* __restrict x = state->x;
	const float* __restrict v = state->v;
	float* __restrict a = state->a;
	uint n = state->n;

	// Compute density and color
	par_compute_density(
		state,
		params
		);

#pragma omp parallel for
	for (uint i = 0; i < n; ++i) {
		a[DIM * i + 0] = 0.f;
		a[DIM * i + 1] = 0.f;
		a[DIM * i + 2] = -g;
	}

	// Constants for interaction term
	float C0 = mass / M_PI / (h2*h2);
	float Cp = 15 * k;
	float Cv = -40.f*mu;

	// Now compute interaction forces
//#pragma omp parallel for
	for (uint i = 0; i < n; ++i) {
		const float rhoi = rho[i];

		for (uint j = i + 1; j < n; ++j) {
			float dx = x[DIM * i + 0] - x[DIM * j + 0];
			float dy = x[DIM * i + 1] - x[DIM * j + 1];
			float dz = x[DIM * i + 2] - x[DIM * j + 2];
			float r2 = dx*dx + dy*dy + dz*dz;

			if (r2 < h2) {
				const float rhoj = rho[j];
				float q = sqrt(r2) / h;
				float u = 1 - q;
				float w0 = C0*u / rhoi / rhoj;
				float wp = w0*Cp*(rhoi + rhoj - 2 * rho0)*u / q;
				float wv = w0*Cv;
				float dvx = v[DIM * i + 0] - v[DIM * j + 0];
				float dvy = v[DIM * i + 1] - v[DIM * j + 1];
				float dvz = v[DIM * i + 2] - v[DIM * j + 2];

				a[DIM * i + 0] += (wp*dx + wv*dvx);
				a[DIM * i + 1] += (wp*dy + wv*dvy);
				a[DIM * i + 2] += (wp*dz + wv*dvz);
				a[DIM * j + 0] -= (wp*dx + wv*dvx);
				a[DIM * j + 1] -= (wp*dy + wv*dvy);
				a[DIM * j + 2] -= (wp*dz + wv*dvz);
			}
		}
	}
}


__global__
void d_compute_accel(
		float* rho,
		float* v,
		float* x,
		float* n2_a,
		sim_param_t* d_params,
		device_info* d_info
		)
{
	uint lo_indx = (uint)(blockDim.x * blockIdx.x + threadIdx.x);

	float mass = d_info->mass;
	float h = d_params->h;
	float h2 = h*h;
	float mu = d_params->mu;
	float k = d_params->k;
	float rho0 = d_params->rho0;

	uint n = d_info->n;
	uint n2 = n*n;
	uint nloop = d_info->nloop;
	uint nThreads = d_info->nThreads;
	uint nBlocks = d_info->nBlocks;
	uint Tthreads = nThreads*nBlocks;
	uint indx = lo_indx + nloop*Tthreads;


	if (indx < n2) {

		n2_a[DIM*indx] = 0.f;
		n2_a[DIM*indx + 1] = 0.f;
		n2_a[DIM*indx + 2] = 0.f;

		float C0 = mass / M_PI / (h2*h2);
		float Cp = 15 * k;
		float Cv = -40.f*0.f;

		uint i, j;

		i = (uint)(indx / n);
		j = (uint)(indx % n);

		if (i<j) {
			const float rhoi = rho[i];

			float dx = x[DIM * i + 0] - x[DIM * j + 0];
			float dy = x[DIM * i + 1] - x[DIM * j + 1];
			float dz = x[DIM * i + 2] - x[DIM * j + 2];
			float r2 = dx*dx + dy*dy + dz*dz;

			if (r2 < h2) {
				const float rhoj = rho[j];
				float q = sqrt(r2) / h;
				float u = 1 - q;
				float w0 = C0*u / rhoi / rhoj;
				float wp = w0*Cp*(rhoi + rhoj - 2 * rho0)*u / q;
				float wv = w0*Cv;
				float dvx = v[DIM * i + 0] - v[DIM * j + 0];
				float dvy = v[DIM * i + 1] - v[DIM * j + 1];
				float dvz = v[DIM * i + 2] - v[DIM * j + 2];
				
				n2_a[DIM*indx] = (wp*dx + wv*dvx);
				n2_a[DIM*indx + 1] = (wp*dy + wv*dvy);
				n2_a[DIM*indx + 2] = (wp*dz + wv*dvz);
				
			}
		}
	}
}



/*
void d_compute_accel(
		const float* __restrict rho,
		const float* __restrict v,
		const float* __restrict x,
		float* n2_a,
		sim_param_t* d_params,
		device_info* d_info,
		uint indx
		)
{
//	uint lo_indx = (uint)(blockDim.x * blockIdx.x + threadIdx.x);

	float mass = d_info->mass;
	float h = d_params->h;
	float h2 = h*h;
	float mu = d_params->mu;
	float k = d_params->k;
	float rho0 = d_params->rho0;

	uint n = d_info->n;
	uint n2 = n*n;
	uint nloop = d_info->nloop;
	uint nThreads = d_info->nThreads;
	uint nBlocks = d_info->nBlocks;
	uint Tthreads = nThreads*nBlocks;
//	uint indx = lo_indx + nloop*Tthreads;


	if (indx < n2) {

		n2_a[DIM*indx] = 0.f;
		n2_a[DIM*indx + 1] = 0.f;
		n2_a[DIM*indx + 2] = 0.f;

		float C0 = mass / M_PI / (h2*h2);
		float Cp = 15 * k;
		float Cv = -40.f*0.f;

		uint i, j;

		i = (uint)(indx / n);
		j = (uint)(indx % n);

		if (i<j) {
			const float rhoi = rho[i];

			float dx = x[DIM * i + 0] - x[DIM * j + 0];
			float dy = x[DIM * i + 1] - x[DIM * j + 1];
			float dz = x[DIM * i + 2] - x[DIM * j + 2];
			float r2 = dx*dx + dy*dy + dz*dz;

			if (r2 < h2) {
				const float rhoj = rho[j];
				float q = sqrt(r2) / h;
				float u = 1 - q;
				float w0 = C0*u / rhoi / rhoj;
				float wp = w0*Cp*(rhoi + rhoj - 2 * rho0)*u / q;
				float wv = w0*Cv;
				float dvx = v[DIM * i + 0] - v[DIM * j + 0];
				float dvy = v[DIM * i + 1] - v[DIM * j + 1];
				float dvz = v[DIM * i + 2] - v[DIM * j + 2];

				
				n2_a[DIM*indx] = (wp*dx + wv*dvx);
				n2_a[DIM*indx + 1] = (wp*dy + wv*dvy);
				n2_a[DIM*indx + 2] = (wp*dz + wv*dvz);
			}
		}
	}
}
*/


void par_compute_accel(
	sim_state_t* state,
	sim_param_t* params
	)
{
	uint nBlocks, nThreads;
	nBlocks = 35000;
	nThreads = 128;

	// Unpack basic parameters
	const float h = params->h;
	const float rho0 = params->rho0;
	const float k = params->k;
	const float mu = params->mu;
	const float g = params->g;
	const float mass = state->mass;
	const float h2 = h*h;

	// Unpack system state
	const float* __restrict rho = state->rho;
	const float* __restrict x = state->x;
	const float* __restrict v = state->v;
	float* __restrict a = state->a;
	uint n = state->n;
	uint n2 = n*n;

	// Compute density and color
	par_compute_density(
		state,
		params
		);

	for (uint i = 0; i < n; ++i) {
		a[DIM * i + 0] = 0.f;
		a[DIM * i + 1] = 0.f;
		a[DIM * i + 2] = -g;
	}

	float *d_rho, *d_v, *d_x, *d_n2_a;
	float *h_n2_a;

	h_n2_a = (float*)malloc(sizeof(float)*n2*DIM);

	cudaMalloc((void**)&d_rho, sizeof(float)*n);
	cudaMalloc((void**)&d_v, sizeof(float)*n*DIM);
	cudaMalloc((void**)&d_x, sizeof(float)*n*DIM);
	cudaMalloc((void**)&d_n2_a, sizeof(float)*n2*DIM);

	cudaMemcpy(d_rho, rho, sizeof(float)*n, hTd);
	cudaMemcpy(d_v, v, sizeof(float)*n*DIM, hTd);
	cudaMemcpy(d_x, x, sizeof(float)*n*DIM, hTd);

	sim_param_t* d_param;
	cudaMalloc((void**)&d_param, sizeof(sim_param_t));
	cudaMemcpy(d_param, params, sizeof(sim_param_t), hTd);

	device_info *h_info;
	h_info = (device_info*)malloc(sizeof(device_info));

	h_info->mass = state->mass;
	h_info->n = n;
	h_info->nBlocks = nBlocks;
	h_info->nThreads = nThreads;

	uint Tloop = n2 / nBlocks / nThreads + 1;

	for (uint nloop = 0; nloop < Tloop; ++nloop) {
		h_info->nloop = nloop;

		device_info* d_info;
		cudaMalloc((void**)&d_info, sizeof(device_info));
		cudaMemcpy(d_info, h_info, sizeof(device_info), hTd);

		d_compute_accel <<<nBlocks, nThreads >>>(
			d_rho,
			d_v,
			d_x,
			d_n2_a,
			d_param,
			d_info
			);

		cudaFree(d_info);
	}

	cudaMemcpy(h_n2_a, d_n2_a, sizeof(float)*n2*DIM, dTh);

	omp_set_num_threads(8);
	uint idx = 0;
#pragma omp parallel for private(idx)
	for (idx = 0; idx < n2; idx++) {
		uint i, j;

		i = (uint)(idx / n);
		j = (uint)(idx % n);

		if (i < j)
		{
			a[DIM * i + 0] += h_n2_a[DIM*idx];
			a[DIM * i + 1] += h_n2_a[DIM*idx + 1];
			a[DIM * i + 2] += h_n2_a[DIM*idx + 2];
			a[DIM * j + 0] -= h_n2_a[DIM*idx];
			a[DIM * j + 1] -= h_n2_a[DIM*idx + 1];
			a[DIM * j + 2] -= h_n2_a[DIM*idx + 2];
		}
	}

	cudaFree(d_rho);
	cudaFree(d_v);
	cudaFree(d_x);
	cudaFree(d_n2_a);

	delete[] h_n2_a;
}




/*
void par_compute_accel(
	sim_state_t* state,
	sim_param_t* params
	)
{
	uint nBlocks, nThreads;
	nBlocks = 100;
	nThreads = 64;

	// Unpack basic parameters
	const float h = params->h;
	const float rho0 = params->rho0;
	const float k = params->k;
	const float mu = params->mu;
	const float g = params->g;
	const float mass = state->mass;
	const float h2 = h*h;

	// Unpack system state
	const float* __restrict rho = state->rho;
	const float* __restrict x = state->x;
	const float* __restrict v = state->v;
	float* __restrict a = state->a;
	uint n = state->n;
	uint n2 = n*n;

	// Compute density and color
	par_compute_density(
		state,
		params
		);

#pragma omp parallel for
	for (uint i = 0; i < n; ++i) {
		a[DIM * i + 0] = 0.f;
		a[DIM * i + 1] = 0.f;
		a[DIM * i + 2] = -g;
	}

	float *d_rho, *d_v, *d_x, *d_n2_a;
	float *h_n2_a;

	h_n2_a = (float*)malloc(sizeof(float)*n2*DIM);

	cudaMalloc((void**)&d_rho, sizeof(float)*n);
	cudaMalloc((void**)&d_v, sizeof(float)*n*DIM);
	cudaMalloc((void**)&d_x, sizeof(float)*n*DIM);
	cudaMalloc((void**)&d_n2_a, sizeof(float)*n2*DIM);

	cudaMemcpy(d_rho, rho, sizeof(float)*n, hTd);
	cudaMemcpy(d_v, v, sizeof(float)*n*DIM, hTd);
	cudaMemcpy(d_x, x, sizeof(float)*n*DIM, hTd);

	sim_param_t* d_param;
	cudaMalloc((void**)&d_param, sizeof(sim_param_t));
	cudaMemcpy(d_param, params, sizeof(sim_param_t), hTd);

	device_info *h_info;
	h_info = (device_info*)malloc(sizeof(device_info));

	h_info->mass = state->mass;
	h_info->n = n;
	h_info->nBlocks = nBlocks;
	h_info->nThreads = nThreads;

	uint Tloop = n2 / nBlocks / nThreads + 1;

	for (uint nloop = 0; nloop < Tloop; ++nloop) {
		h_info->nloop = nloop;

		device_info* d_info;
		cudaMalloc((void**)&d_info, sizeof(device_info));
		cudaMemcpy(d_info, h_info, sizeof(device_info), hTd);

		d_compute_accel<< < nBlocks, nThreads >> >(
			d_rho,
			d_v,
			d_x,
			d_n2_a,
			d_param,
			d_info
			);

		cudaFree(d_info);
	}

	cudaMemcpy(h_n2_a, d_n2_a, sizeof(float)*n2*DIM, dTh);

	//omp_set_num_threads(8);
	//uint idx = 0;
//#pragma omp parallel for private(idx)
	for (uint idx = 0; idx < n2; idx++) {
		uint i, j;

		i = (uint)(idx / n);
		j = (uint)(idx % n);

		if (i < j)
		{
			printf("h_n2_a: %f\n", h_n2_a[idx]);

			a[DIM * i + 0] += h_n2_a[idx];
			a[DIM * i + 1] += h_n2_a[idx + 1];
			a[DIM * i + 2] += h_n2_a[idx + 2];
			a[DIM * j + 0] -= h_n2_a[idx];
			a[DIM * j + 1] -= h_n2_a[idx + 1];
			a[DIM * j + 2] -= h_n2_a[idx + 2];
		}
	}

	cudaFree(d_rho);
	cudaFree(d_v);
	cudaFree(d_x);
	cudaFree(d_n2_a);

	delete[] h_n2_a;
}
*/




static void damp_reflect(
	int which,
	float barrier,
	float* x,
	float* v,
	float* vh)
{
	// Coefficient of resitiution
	const float DAMP = .75;

	// Ignore degenerate cases
	if (v[which] == 0) return;

	// Scale back th distance traveled based on time from collision
	float tbounce = (x[which] - barrier) / v[which];
	x[0] -= v[0] * (1 - DAMP)*tbounce;
	x[1] -= v[1] * (1 - DAMP)*tbounce;
	x[2] -= v[2] * (1 - DAMP)*tbounce;

	// Reflect the position and velocity
	x[which] = 2 * barrier - x[which];
	v[which] = -v[which];
	vh[which] = -vh[which];

	// Damp the velocities
	v[0] *= DAMP;
	v[1] *= DAMP;
	v[2] *= DAMP;
	vh[0] *= DAMP;
	vh[1] *= DAMP;
	vh[2] *= DAMP;
}

static void reflect_bc(
	sim_state_t* s
	)
{
	// Boundaries of the computational domain
	const float XMIN = 0.;
	const float XMAX = 1.;
	const float YMIN = 0.;
	const float YMAX = 1.;
	const float ZMIN = 0.;
	const float ZMAX = 1.;

	float* __restrict vh = s->vh;
	float* __restrict v = s->v;
	float* __restrict x = s->x;
	uint n = s->n;

	for (uint i = 0; i < n; ++i, x += DIM, v += DIM, vh += DIM) {
		if (x[0] < XMIN) damp_reflect(0, XMIN, x, v, vh);
		if (x[0] > XMAX) damp_reflect(0, XMAX, x, v, vh);

		if (x[1] < YMIN) damp_reflect(1, YMIN, x, v, vh);
		if (x[1] > YMAX) damp_reflect(1, YMAX, x, v, vh);

		if (x[2] < ZMIN) damp_reflect(2, ZMIN, x, v, vh);
		if (x[2] > ZMAX) damp_reflect(2, ZMAX, x, v, vh);
	}

}

void leapfrog_step(
	sim_state_t* s,
	float dt)
{
	const float* __restrict a = s->a;
	float* __restrict vh = s->vh;
	float* __restrict v = s->v;
	float* __restrict x = s->x;
	uint n = s->n;
	for (uint i = 0; i < DIM * n; ++i) vh[i] += a[i] * dt;
	for (uint i = 0; i < DIM * n; ++i) v[i] = vh[i] + a[i] * dt / 2.f;
	for (uint i = 0; i < DIM * n; ++i) x[i] += vh[i] * dt;

	reflect_bc(s);
}

void leapfrog_start(
	sim_state_t* s,
	float dt)
{
	const float* __restrict a = s->a;
	float* __restrict vh = s->vh;
	float* __restrict v = s->v;
	float* __restrict x = s->x;
	uint n = s->n;

	for (uint i = 0; i < DIM * n; ++i) vh[i] = v[i] + a[i] * dt / 2.f;
	for (uint i = 0; i < DIM * n; ++i) v[i] += a[i] * dt;
	for (uint i = 0; i < DIM * n; ++i) x[i] += vh[i] * dt;

	reflect_bc(s);
}

typedef int(*domain_fun_t)(float, float, float);

int box_indicator(float x, float y, float z)
{
	return (x < .5) && (y < .5) && (z < .5);
}

int circ_indicator(float x, float y, float z)
{
	float dx = (x - .5);
	float dy = (y - .3);
	float r2 = dx*dx + dy*dy;
	return (r2 < .25*.25);
}

sim_state_t* place_particles(
	sim_param_t* param,
	domain_fun_t indicatef)
{
	float h = param->h;
	float hh = h / 1.3;

	// Count mesh points that fall in indicated region
	int count = 0;
	for (float x = 0; x < 1; x += hh) {
		for (float y = 0; y < 1; y += hh) {
			for (float z = 0; z < 1; z += hh) {
				count += indicatef(x, y, z);
			}
		}
	}

	// Populate the particle data structure
	sim_state_t* s = alloc_state(count);
	uint p = 0;
	for (float x = 0; x < 1; x += hh) {
		for (float y = 0; y < 1; y += hh) {
			for (float z = 0; z < 1; z += hh) {
				if (indicatef(x, y, z)) {
					s->x[DIM * p + 0] = x;
					s->x[DIM * p + 1] = y;
					s->x[DIM * p + 2] = z;
					s->v[DIM * p + 0] = 0;
					s->v[DIM * p + 1] = 0.;
					s->v[DIM * p + 2] = 0.;
					++p;
				}
			}
		}
	}

	return s;
}

void normalize_mass(
	sim_state_t* s,
	sim_param_t* param)
{
	s->mass = 1.;
	compute_density(s, param);
	float rho0 = param->rho0;
	float rho2s = 0;
	float rhos = 0;
	for (uint i = 0; i < s->n; ++i) {
		rho2s += (s->rho[i])*(s->rho[i]);
		rhos += s->rho[i];
	}

	s->mass *= (rho0*rhos / rho2s);
}

sim_state_t* init_particles(
	sim_param_t* param)
{
	sim_state_t* s = place_particles(param, box_indicator);
	normalize_mass(s, param);
	return s;
}

void check_state(
	sim_state_t* s)
{
#pragma omp parallel for
	for (uint i = 0; i < s->n; ++i){
		float xi = s->x[DIM * i + 0];
		float yi = s->x[DIM * i + 1];
		float zi = s->x[DIM * i + 2];
		assert(xi >= 0 || xi <= 1);
		assert(yi >= 0 || yi <= 1);
		assert(zi >= 0 || zi <= 1);
	}
}


void write_frame_data(
	FILE* fp,
	uint n,
	float* x,
	int* c)
{
	for (uint i = 0; i < n; ++i) {
		fprintf(fp, "%f, %f, %f\n", x[i * DIM], x[i * DIM + 1], x[i * DIM + 2]);
	}
}


///////////////////////////////////////////////////////////////////////

int main()
{
	sim_param_t params;

	//default_params

	params.fname = "run.out";
	params.nframes = 1;
	params.npframe = 2000;
	params.dt = 1.e-3;
	params.h = 5e-2;
	params.rho0 = 1000.f;
	params.k = 1000.f;
	params.mu = .0;
	params.g = 9.8;


	sim_state_t* state = init_particles(&params);

	int nframes = params.nframes;
	int npframe = params.npframe;
	float dt = params.dt;
	uint n = state->n;

	par_compute_accel(state, &params);
	leapfrog_start(state, dt);
	check_state(state);

	for (uint frame = 1; frame <= nframes; ++frame) {

		for (uint i = 1; i <= npframe; i++) {
			printf("i:%d\n", i);

			char fpname[] = ".\\data\\run_";
			char str[10];
			char ext[] = ".dat";
			itoa(i, str, 10);

			char* left = strcat(fpname, str);
			char* fp_name = strcat(left, ext);

			FILE* fp = fopen(fp_name, "w");
			par_compute_accel(state, &params);
			leapfrog_step(state, dt);
			check_state(state);

			write_frame_data(fp, n, state->x, NULL);

			fclose(fp);
		}
	}

	printf("Ran in seconds\n");

	return 0;
}