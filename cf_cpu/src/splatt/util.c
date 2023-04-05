/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "util.h"

void par_memset(void* ptr, int c, size_t n)
{
#pragma omp parallel
  {
    int n_threads = omp_get_num_threads();
    int t_id = omp_get_thread_num();

    size_t n_per_thread = (n + n_threads - 1) / n_threads;
    size_t n_begin = SS_MIN(n_per_thread * t_id, n);
    size_t n_end = SS_MIN(n_begin + n_per_thread, n);

    memset((char*)ptr + n_begin, c, n_end - n_begin);
  }
}
