# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(bench_hnsw_qinco EXCLUDE_FROM_ALL bench_hnsw_qinco.cpp)
if(FAISS_OPT_LEVEL STREQUAL "sve")
  target_link_libraries(bench_hnsw_qinco PRIVATE faiss_sve ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
  # Tune both QINCo and the native Faiss SQ8 baseline for the benchmark CPU.
  target_compile_options(faiss_sve PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=armv9-a+sve2+i8mm -mtune=neoverse-v2>)
  target_compile_options(bench_hnsw_qinco PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=armv9-a+sve2+i8mm -mtune=neoverse-v2>)
else()
  target_link_libraries(bench_hnsw_qinco PRIVATE faiss_avx2 ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
  target_compile_options(bench_hnsw_qinco PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
endif()

add_executable(bench_ivf_selector EXCLUDE_FROM_ALL bench_ivf_selector.cpp)
target_link_libraries(bench_ivf_selector PRIVATE faiss_avx512 ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(bench_ivf_selector PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)

add_executable(bench_result_handler_overhead EXCLUDE_FROM_ALL
  bench_result_handler_overhead.cpp)
target_link_libraries(bench_result_handler_overhead PRIVATE faiss_avx512 ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(bench_result_handler_overhead PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)
