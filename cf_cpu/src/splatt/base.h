#ifndef SPLATT_BASE_H
#define SPLATT_BASE_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <inttypes.h>
#include <float.h>

/******************************************************************************
 * DEFINES
 *****************************************************************************/

#define SS_MIN(x, y) ((x) < (y) ? (x) : (y))
#define SS_MAX(x, y) ((x) > (y) ? (x) : (y))


/******************************************************************************
 * TYPES
 *****************************************************************************/

/* USER: You may edit values to chance the size of integer and real types.
 * Accepted values are 32 and 64. Changing these values to 32 will decrease
 * memory consumption at the cost of precision and maximum supported tensor
 * size. */

#ifndef SPLATT_IDX_TYPEWIDTH
  #define SPLATT_IDX_TYPEWIDTH 64
#endif

#ifndef SPLATT_VAL_TYPEWIDTH
  #define SPLATT_VAL_TYPEWIDTH 64
#endif


/* Set type constants based on width. */
#if   SPLATT_IDX_TYPEWIDTH == 32
  typedef uint32_t idx_t;
  #define SPLATT_IDX_MAX UINT32_MAX
  #define SPLATT_PF_IDX PRIu32
  #define SPLATT_MPI_IDX MPI_UINT32_T

#elif SPLATT_IDX_TYPEWIDTH == 64
  typedef uint64_t idx_t;
  #define SPLATT_IDX_MAX UINT64_MAX
  #define SPLATT_PF_IDX PRIu64
  #define SPLATT_MPI_IDX MPI_UINT64_T
#else
  #error *** Incorrect user-supplied value of SPLATT_IDX_TYPEWIDTH ***
#endif


#if   SPLATT_VAL_TYPEWIDTH == 32
  typedef float val_t;
  #define SPLATT_VAL_MIN FLT_MIN
  #define SPLATT_VAL_MAX FLT_MAX
  #define SPLATT_PF_VAL "f"
  #define SPLATT_MPI_VAL MPI_FLOAT

#elif SPLATT_VAL_TYPEWIDTH == 64
  typedef double val_t;
  #define SPLATT_VAL_MIN DBL_MIN
  #define SPLATT_VAL_MAX DBL_MAX
  #define SPLATT_PF_VAL "f"
  #define SPLATT_MPI_VAL MPI_DOUBLE

#else
  #error *** Incorrect user-supplied value of SPLATT_VAL_TYPEWIDTH ***
#endif


#ifdef __cplusplus
extern "C"
{
#endif

    /******************************************************************************
     * MEMORY ALLOCATION
     *****************************************************************************/

    /**
     * @brief Allocate 'bytes' memory, 64-bit aligned. Returns a pointer to memory.
     *
     * @param bytes The number of bytes to allocate.
     *
     * @return The allocated memory.
     */
    void* splatt_malloc(size_t const bytes);

    /**
     * @brief Free memory allocated by splatt_malloc().
     *
     * @param ptr The pointer to free.
     */
    void splatt_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif
