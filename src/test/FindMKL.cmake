#Find MKL on host system

set(MKL_ROOT $ENV{MKLROOT})

if(MKL_ROOT)
  set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
  set(MKL_LIBS "${MKL_ROOT}/libmkl_intel_ilp64")
  set(MKL_INCLUDE "${MKL_ROOT}/include")
endif(MKL_ROOT)

include(FindPackageHandleStandardArgsDLKFJ)
#find_package_handle_standard_args(MKL DEFAULT_MSG MKL_ROOT MKL_LIBS MKL_INCLUDE)
