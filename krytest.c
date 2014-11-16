#include "krytest.h"

void krytest_Run(kryTestFn f)
{
  struct timespec t0, t1, res;

  clock_getres(CLOCK_MONOTONIC_RAW, &res);

  clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
  int result = f();
  clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

  size_t _seconds = t1.tv_sec - t0.tv_sec;
  double seconds = _seconds + (((double)(t1.tv_nsec - t0.tv_nsec)) / 1.0e9*res.tv_nsec);

  if(result != KRY_SUCCESS)
  {
  }
  else
  {
    printf("%s %f\n", KRY_GREEN "OK" KRY_RESET, seconds);
  }
}

