#ifndef KRYTEST_H
#define KRYTEST_H

#define KRY_RED     "\x1b[31m"
#define KRY_GREEN   "\x1b[32m"
#define KRY_RESET   "\x1b[0m"

typedef int(*kryTestFn)(void);

void krytest_Run(kryTestFn f);

#define KRY_TEST(__TNAME__)                                                   \
  int __krytest__ ## __TNAME__()

#define KRY_RUNTEST(__TNAME__)                                                \
  printf("%s     .........     ", #__TNAME__);                                \
  krytest_Run(& __krytest__ ## __TNAME__);

#define KRY_EXPECT_EQ(__EXPECTED__, __ACTUAL__)                               \
  if(__EXPECTED__ != __ACTUAL__)                                              \
  {                                                                           \
    printf("%s %s:%d\n", KRY_RED "FAIL" KRY_RESET, __FILE__, __LINE__);       \
    return EXIT_FAILURE;                                                      \
  }

#endif
