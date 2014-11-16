#ifndef KRYTEST_H
#define KRYTEST_H

#include "krycl.h"

#define KRY_RED     "\x1b[1;31m"
#define KRY_GREEN   "\x1b[1;32m"
#define KRY_CYAN    "\x1b[1;34m"
#define KRY_RESET   "\x1b[0;0m"

typedef int(*kryTestFn)(void);

void krytest_Run(kryTestFn f);

#define KRY_TEST(__TNAME__)                                                   \
  int __krytest__ ## __TNAME__()

#define KRY_RUNTEST(__TNAME__)                                                \
  printf(KRY_CYAN "%s     .........     " KRY_RESET, #__TNAME__);             \
  krytest_Run(& __krytest__ ## __TNAME__);

#define KRY_EXPECT_EQ(__EXPECTED__, __ACTUAL__)                               \
  if(__EXPECTED__ != __ACTUAL__)                                              \
  {                                                                           \
    printf("%s %s:%d\n", KRY_RED "FAIL" KRY_RESET, __FILE__, __LINE__);       \
    return EXIT_FAILURE;                                                      \
  }

#endif
