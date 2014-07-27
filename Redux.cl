/******************************************************************************
 *  The Splash Project
 *
 *  reduction kernels - 25 July '14
 *  ~ ry
 *
 *  Required Compilation Input:
 *    - @REAL macro as must be defined either double or float for compilation
 *
 */

/*= redux_add =================================================================
 * reduces the vector @x using addition
 *
 *  Parameters:
 *    - @x - the vector to be reduced
 *    - @N - the size of the input
 *    - @ipt - the number of input items handled per thread
 *    - @lrspace - the local reduction space e.g., a local memory space for
 *                 the compute units to use for reduction processing
 *    - @grspace - the global reduction space e.g., a global memory space for
 *                 reduction processing
 *  
 */
__kernel
void
redux_add(
    __global REAL *x, unsigned long N, unsigned long ipt,
    __local double *lrspace,
    __global double *grspace
    ) {

  //global and local (workgroup) thread dimensions
  size_t 
    G0 = get_global_size(0),
    G1 = get_global_size(1),
    L0 = get_local_size(0),
    L1 = get_local_size(1);

  //global and local thread indices
  size_t tg0 = get_global_id(0),
         tg1 = get_global_id(1),
         tg = G0*tg0 + tg1,
         tl0 = get_local_id(0),
         tl1 = get_local_id(1),
         tl = L0*tl0 + tl1;

  __private REAL acc; //private variable used for accumulations
  acc = 0;

  //compact the input space into local the reduced space
  for(size_t i=tg*ipt; i<(tg*ipt + ipt) && i < N; ++i) {
      acc += x[i];
  }
  lrspace[tl] = acc;
  acc = 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  //compact the reduced 2 dimensional space into a 1 dimensional space
  if(tl1 == 0) { 
    for(size_t i=0; i<L1; ++i) { acc += lrspace[L0*tl0 + i]; } 
    lrspace[L0*tl0] = acc;
    acc = 0;
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  //compact the reduced 1 dimensional space onto a point

  if(tl1 == 0 && tl0 == 0) {
    for(size_t i=0; i<L0; ++i) { acc += lrspace[L0 * i]; }

    //put the point in the global reduction store so it can be accumulated
    size_t gid = get_num_groups(0)*get_group_id(0) + get_group_id(1);
    grspace[gid] = acc;
  }

}
