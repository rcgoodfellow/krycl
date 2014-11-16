/******************************************************************************
 *
 */
#include "krycl.h"

cl_int clError = 0;
FILE *kryLog = NULL;

int kryInit()
{
  kryLog = fopen("kry.log", "a+");
  return EXIT_SUCCESS;
}

char tb[15];
void kryLogTime()
{
  time_t t;
  struct tm *tm_info;
  struct timeval tv;
  gettimeofday(&tv, NULL);

  time(&t);
  tm_info = localtime(&t);

  strftime(tb, 15, "%m/%d %H:%M:%S", tm_info);
  tb[14] = 0;
  fprintf(kryLog, "[%s.%u]", tb, (unsigned)((double)tv.tv_usec / 1000.0));
}

int kryGetAGPU(kryGPUInfo *ginfo)
{
  clError = clGetPlatformIDs(1, &ginfo->pid, NULL);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "%s - clError %d", 
        "Error getting OpenCL platform", clError);
    return KRY_CL_PLATFORM_CREATE_ERROR;
  }

  size_t sz;
  clError = clGetPlatformInfo(ginfo->pid, CL_PLATFORM_VERSION, 0, NULL, &sz);
  if(clError) 
  {
    KRYLOG_DEV(KRY_WARN, "%s - clError %d", 
        "Unable to query OpenCL platform info", clError);
  }
  else
  {
    char *pinfo = (char *)malloc(sizeof(char)*sz);
    clError = 
      clGetPlatformInfo(ginfo->pid, CL_PLATFORM_VERSION, sz, pinfo, NULL);
    if(clError) 
    {
      KRYLOG_DEV(KRY_WARN, "%s - clError %d", 
          "Unable to query OpenCL platform info", clError);
    }
    else 
    {
      KRYLOG(KRY_INFO, "OpenCL Platform Version: %s", pinfo);
    }
    free(pinfo);
  }

  clError = clGetDeviceIDs(ginfo->pid, CL_DEVICE_TYPE_GPU, 1, &ginfo->did, NULL);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "%s - clError %d", "Error getting GPU", clError);
    return KRY_CL_GPU_CREATE_ERROR;
  }

  ginfo->ctx = clCreateContext(NULL, 1, &ginfo->did, NULL, NULL, &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "%s - clError %d", 
        "Unable to create OpenCL Context", clError);
    return KRY_CL_CONTEXT_CREATE_ERROR;
  }

  ginfo->q = clCreateCommandQueue(ginfo->ctx, ginfo->did, 0, &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "%s - clError %d", 
        "Unable to create OpenCL Command Queue", clError);
    return KRY_CL_COMMAND_Q_CREATE_ERROR;
  }

  clError = clGetDeviceInfo(ginfo->did, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
      sizeof(size_t)*3, ginfo->max_work_item_sizes, NULL);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "%s - clError %d", 
        "Unable to determine OpenCL device maximum work item sizes", clError);
    return KRY_CL_DEVINFO_QUERY_ERROR;
  }

  clError = clGetDeviceInfo(ginfo->did, CL_DEVICE_MAX_COMPUTE_UNITS,
      sizeof(cl_uint), &ginfo->max_compute_units, NULL);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "%s - clError %d",
        "Unable to determine OpenCL device maximum compute units", clError);
    return KRY_CL_DEVINFO_QUERY_ERROR;
  }

  int err = kryLoadCore(ginfo);
  if(err) return err;

  return KRY_SUCCESS;
}

int kryReadProgramSource(const char* fn, char **src, size_t *sz)
{
  FILE *f = fopen(fn, "r");
  if(!f) 
  {
    KRYLOG(KRY_FAIL, "Source file not found: %s", fn);
    return KRY_CL_SOURCE_NOT_FOUND;
  }

  fseek(f, 0, SEEK_END);
  long _sz = ftell(f);
  rewind(f);
  *src = (char*)malloc(_sz);
  *sz = fread(*src, sizeof(char), _sz, f);
  if(*sz != (size_t)_sz) 
  {
    KRYLOG_DEV(KRY_FAIL, "Could only read %zu/%lu bytes of source file %s", *sz, _sz, fn);
    return KRY_INCOMPLETE_SOURCE_READ;
  }
  fclose(f);

  return KRY_SUCCESS;
}

int kryLoadCore(kryGPUInfo *ginfo)
{
  const char *fn = "krycore.cl";
  char *src = NULL;
  size_t sz; 
  int err = kryReadProgramSource(fn, &src, &sz);
  if(err) return err;

  ginfo->kry_core = clCreateProgramWithSource(ginfo->ctx, 1, (const char**)&src,
      &sz, &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "%s - clError %d", 
        "Could not create krycl core library", clError);
    return KRY_CREATE_PROGRAM_ERROR;
  }

  clError = clBuildProgram(ginfo->kry_core, 1, &ginfo->did, NULL, NULL, NULL);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "%s - clError %d",
        "Could not build krycl core library", clError);
    kryCLCLog(ginfo, ginfo->kry_core);
    return KRY_BUILD_PROGRAM_ERROR;
  }

  free(src);
  return KRY_SUCCESS;
}

int kryCLCLog(kryGPUInfo *ginfo, cl_program prog)
{
  size_t sz;
  clError = clGetProgramBuildInfo(prog, ginfo->did,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "Unable to access build log - clError %d", clError);
    return KRY_BUILD_LOG_ACCESS_ERROR;
  }

  char *log = (char*)malloc(sz);

  clError = clGetProgramBuildInfo(prog, ginfo->did,
      CL_PROGRAM_BUILD_LOG, sz, log, NULL);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, "Unable to access build log - clError %d", clError);
    return KRY_BUILD_LOG_ACCESS_ERROR;
  }

  KRYLOG(KRY_FAIL, "CLC Build Log\n %s", log);

  return KRY_SUCCESS;
}
