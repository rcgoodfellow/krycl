#ifndef KRYCL_H
#define KRYCL_H

#include </usr/local/include/CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define KRYCL_VERSION_MAJOR 0
#define KRYCL_VERSION_MINOR 1

#define KRY_SUCCESS 0
#define KRY_CL_PLATFORM_CREATE_ERROR -1000
#define KRY_NO_AVAILABLE_PLATFORMS -1001
#define KRY_CL_GPU_CREATE_ERROR -1002
#define KRY_NO_AVAILABLE_GPUS -1003
#define KRY_CL_CONTEXT_CREATE_ERROR -1004
#define KRY_CL_COMMAND_Q_CREATE_ERROR -1005
#define KRY_KALLOC_ERROR -1006
#define KRY_CL_SOURCE_NOT_FOUND -1007
#define KRY_INCOMPLETE_SOURCE_READ -1008
#define KRY_CREATE_PROGRAM_ERROR -1009
#define KRY_BUILD_PROGRAM_ERROR -1010
#define KRY_BUILD_LOG_ACCESS_ERROR -1011
#define KRY_CREATE_CL_KERNEL_ERROR -1012
#define KRY_SET_KERNEL_ARG_ERROR -1013
#define KRY_CL_DEVINFO_QUERY_ERROR -1014
#define KRY_EXEC_KERNEL_ERROR -1015
#define KRY_KERNEL_READBACK_ERROR -1016

extern cl_int clError;

typedef struct kryGPUInfo
{
  cl_platform_id pid;
  cl_device_id did;
  cl_context ctx;
  cl_command_queue q;
  size_t max_work_item_sizes[3];
  cl_uint max_compute_units;
  cl_program kry_core;
} 
kryGPUInfo;

typedef struct krySparseMatrix
{
  double *v;
  cl_uint *c, *r;
  cl_uint N, n;
} 
krySparseMatrix;

typedef struct kryExecInfo
{
  cl_kernel *kernels;
  kryGPUInfo *ginfo;
} 
kryExecInfo;

int kryGetAGPU(kryGPUInfo *ginfo); 
int kryArnoldi(kryGPUInfo *ginfo, kryExecInfo *xinfo, krySparseMatrix *A, double *b, double *x0, double *x);
int kryCLCSpew(kryExecInfo *xinfo, char **log);
int kryReadProgramSource (const char* fn, char **src, size_t *sz);
int kryLoadCore (kryGPUInfo *ginfo);

#endif
