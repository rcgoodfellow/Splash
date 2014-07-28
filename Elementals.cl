/******************************************************************************
 *  The Splash Project
 *
 *  elemental functions for vectors - 27 July '14
 *  ~ ry
 *
 *  Required Compilation Input:
 *    - @REAL macro as must be defined either double or float for compilation
 *
 */

/*= ELEMENTAL_UFUNC============================================================
 *  preprocessor macro to generate unary elemental kernels based on a unary 
 *  functions
 *
 *  Macro Parameters:
 *    - @NAME - The name of the elemental kernel to be generated.
 *    - @FUNC - The name of the elemental unary function to be called.
 *
 *  Generated Kernel Parameters:
 *    - @x - input vector
 *    - @N - size of the input
 *    - @ipt - the number of input elements handled per thread
 *    - @result - memory where the result of the computation is placed
 */
#define ELEMENTAL_UFUNC(NAME, FUNC) \
__kernel \
void \
NAME( \
    __global REAL *x, unsigned long N, unsigned long ipt, \
    __global REAL *result \
    ) { \
\
  size_t tid = get_global_id(0); \
  size_t begin = tid*ipt, end = begin + ipt; \
  for(size_t i=begin; i<end; ++i) { result[i] = FUNC(x[i]); } \
}

ELEMENTAL_UFUNC(ksqrt, sqrt)

/*= ELEMENTAL_BINOP_VS=========================================================
 *  preprocessor macro to generate binary vector scalar elemental kernels
 *  based on binary operators
 *
 *  Macro Parameters:
 *    - @NAME - The name of the elemental kernel to be generated
 *    - @OP - The binary operation to be used
 *
 *  Generated Kernel Parameters:
 *    - @x - input vector
 *    - @s - input scalar
 *    - @N - size of the input vector
 *    - @ipt - the number of input elements handled per thread
 *    - @result - memory where the result of the computation is placed
 */
#define ELEMENTAL_BINOP_VS(NAME, OP) \
__kernel \
void \
NAME( \
    __global REAL *x, REAL s, unsigned long N, unsigned long ipt, \
    __global REAL *result \
    ) { \
\
  size_t tid = get_global_id(0); \
  size_t begin = tid*ipt, end = begin + ipt; \
  for(size_t i=begin; i<end; ++i) { result[i] = OP(x[i], s); } \
}

REAL div(REAL a, REAL b) { return a / b; }
REAL mul(REAL a, REAL b) { return a * b; }
REAL add(REAL a, REAL b) { return a + b; }
REAL sub(REAL a, REAL b) { return a - b; }

ELEMENTAL_BINOP_VS(kdiv, div)
ELEMENTAL_BINOP_VS(kmul, mul)
ELEMENTAL_BINOP_VS(kadd, add)
ELEMENTAL_BINOP_VS(ksub, sub)
