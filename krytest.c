#include "krytest.h"
#include "krycl.h"

void krytest_Run(kryTestFn f)
{
  int result = f();
  if(result != KRY_SUCCESS)
  {
  }
  else
  {
    printf("%s\n", KRY_GREEN "OK" KRY_RESET);
  }
}
