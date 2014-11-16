/******************************************************************************
 *
 */
#include "krycl.h"

cl_int clError = 0;

int kryGetAGPU(kryGPUInfo *ginfo)
{
  cl_uint found = 0;
  clError = clGetPlatformIDs(1, &ginfo->pid, &found);
  if(clError) return KRY_CL_PLATFORM_CREATE_ERROR;
  if(!found) return KRY_NO_AVAILABLE_PLATFORMS;

  found = 0;
  clError = clGetDeviceIDs(ginfo->pid, CL_DEVICE_TYPE_GPU, 0, NULL, &found);
  if(clError) return KRY_CL_GPU_CREATE_ERROR;
  if(!found) return KRY_NO_AVAILABLE_GPUS;

  clError = clGetDeviceIDs(ginfo->pid, CL_DEVICE_TYPE_GPU, 1, &ginfo->did, &found);
  if(clError) return KRY_CL_GPU_CREATE_ERROR;

  ginfo->ctx = clCreateContext(NULL, 1, &ginfo->did, NULL, NULL, &clError);
  if(clError) return KRY_CL_CONTEXT_CREATE_ERROR;

  ginfo->q = clCreateCommandQueue(ginfo->ctx, ginfo->did, 0, &clError);
  if(clError) return KRY_CL_COMMAND_Q_CREATE_ERROR;

  clError = clGetDeviceInfo(ginfo->did, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
      sizeof(size_t)*3, ginfo->max_work_item_sizes, NULL);
  if(clError) return KRY_CL_DEVINFO_QUERY_ERROR;

  clError = clGetDeviceInfo(ginfo->did, CL_DEVICE_MAX_COMPUTE_UNITS,
      sizeof(cl_uint), &ginfo->max_compute_units, NULL);
  if(clError) return KRY_CL_DEVINFO_QUERY_ERROR;

  int err = kryLoadCore(ginfo);
  if(err) return err;

  return KRY_SUCCESS;
}

int kryReadProgramSource(const char* fn, char **src, size_t *sz)
{
  FILE *f = fopen(fn, "r");
  if(!f) return KRY_CL_SOURCE_NOT_FOUND;

  fseek(f, 0, SEEK_END);
  long _sz = ftell(f);
  rewind(f);
  *src = (char*)malloc(_sz);
  *sz = fread(*src, sizeof(char), _sz, f);
  if(*sz != (size_t)_sz) return KRY_INCOMPLETE_SOURCE_READ;
  fclose(f);

  return KRY_SUCCESS;
}

int kryLoadCore(kryGPUInfo *ginfo)
{
  const char *fn = "mulSmDv.cl";
  char *src = NULL;
  size_t sz; 
  int err = kryReadProgramSource(fn, &src, &sz);
  if(err) return err;

  ginfo->kry_core = clCreateProgramWithSource(ginfo->ctx, 1, (const char**)&src,
      &sz, &clError);
  if(clError) return KRY_CREATE_PROGRAM_ERROR;

  clError = clBuildProgram(ginfo->kry_core, 1, &ginfo->did, NULL, NULL, NULL);
  if(clError) return KRY_BUILD_PROGRAM_ERROR;

  free(src);
  return KRY_SUCCESS;
}

int kryCLCSpew(kryExecInfo *xinfo, char **log)
{
  size_t sz;
  clError = clGetProgramBuildInfo(xinfo->ginfo->kry_core, xinfo->ginfo->did,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
  if(clError) return KRY_BUILD_LOG_ACCESS_ERROR;

  *log = (char*)malloc(sz);

  clError = clGetProgramBuildInfo(xinfo->ginfo->kry_core, xinfo->ginfo->did,
      CL_PROGRAM_BUILD_LOG, sz, *log, NULL);
  if(clError) return KRY_BUILD_LOG_ACCESS_ERROR;

  return KRY_SUCCESS;
}
