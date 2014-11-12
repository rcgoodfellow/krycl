#ifndef KRYCL_H
#define KRYCL_H

#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>

#define KRYCL_VERSION_MAJOR 0
#define KRYCL_VERSION_MINOR 1

#define KRY_SUCCESS 0
#define KRY_CL_PLATFORM_CREATE_ERROR -1000
#define KRY_NO_AVAILABLE_PLATFORMS -1001
#define KRY_CL_GPU_CREATE_ERROR -1002
#define KRY_NO_AVAILABLE_GPUS -1003
#define KRY_CL_CONTEXT_CREATE_ERROR -1004
#define KRY_CL_COMMAND_Q_CREATE_ERROR -1005
#define KRY_BAD_SM_VALUE_PTR -1006
#define KRY_BAD_SM_COLUMN_PTR -1007
#define KRY_BAD_SM_ROW_PTR -1008
#define KRY_BAD_RHS_PTR -1009
#define KRY_BAD_X0_PTR -1010
#define KRY_BAD_X_PTR -1011
#define KRY_CL_SOURCE_NOT_FOUND -1012
#define KRY_INCOMPLETE_SOURCE_READ -1013
#define KRY_CREATE_PROGRAM_ERROR -1014
#define KRY_BUILD_PROGRAM_ERROR -1015
#define KRY_BUILD_LOG_ACCESS_ERROR -1016

extern cl_int clError;

typedef struct kryGPUInfo
{
  cl_platform_id pid;
  cl_device_id did;
  cl_context ctx;
  cl_command_queue q;
} kryGPUInfo;

typedef struct krySparseMatrix
{
  double *v;
  unsigned *c, *r;
  unsigned N, n;
} krySparseMatrix;

typedef struct kryExecInfo
{
  cl_program prog;
  kryGPUInfo *ginfo;
} kryExecInfo;

int kryGetAGPU(kryGPUInfo *ginfo); 

int kryArnoldi(kryGPUInfo *ginfo, 
    kryExecInfo *xinfo,
    krySparseMatrix *A, double *b, double *x0, double *x);

int kryCLCSpew(kryExecInfo *xinfo, char **log);

#endif
