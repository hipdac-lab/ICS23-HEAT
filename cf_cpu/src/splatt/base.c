/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"

/* for `posix_memalign()` errors */
#include <errno.h>

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void * splatt_malloc(size_t const bytes)
{
  void * ptr;
  int const success = posix_memalign(&ptr, 64, bytes);

  if(success != 0) {
    switch(success) {
    case ENOMEM:
      fprintf(stderr, "SPLATT: posix_memalign() returned ENOMEM. "
                      "Insufficient memory.\n");
      break;
    case EINVAL:
      fprintf(stderr, "SPLATT: posix_memalign() returned EINVAL. "
                      "Alignment must be power of two.\n");
      break;
    default:
      fprintf(stderr, "SPLATT: posix_memalign() returned '%d'.\n", success);
      break;
    }

    ptr = NULL;
  }

  return ptr;
}

void splatt_free(void * ptr)
{
  free(ptr);
}
