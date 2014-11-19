#Find MKL on host system

set(PARSTUDIO_ROOT $ENV{PARSTUDIO})
set(MKL_ROOT "${PARSTUDIO_ROOT}/mkl")

if(MKL_ROOT)
  set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
  set(PS_LIB_DIR "${PARSTUDIO_ROOT}/lib/intel64")
  set(MKL_LIBS 
    "${MKL_LIB_DIR}/libmkl_intel_lp64.so"
    "${MKL_LIB_DIR}/libmkl_core.so"
    "${MKL_LIB_DIR}/libmkl_intel_thread.so"
    "${PS_LIB_DIR}/libiomp5.so"
    "dl"
    "pthread"
    "m"
    )
  set(MKL_INCLUDE "${MKL_ROOT}/include")
  set(MKL_C_FLAGS  "-m64")
  set(MKL_LINK_FLAGS "-Wl,--no-as-needed")
endif(MKL_ROOT)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG 
  PARSTUDIO_ROOT MKL_ROOT MKL_LIBS MKL_INCLUDE MKL_C_FLAGS MKL_LINK_FLAGS)
