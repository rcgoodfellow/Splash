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
 *    - @r - the sum of @x
 *    - @lrspace - the local reduction space e.g., a local memory space for
 *                 the compute units to use for reduction processing
 *    - @Nl - the size of the local reduction space
 *    - @grspace - the global reduction space e.g., a global memory space for
 *                 reduction processing
 *    - @Ng - the size of the global reduction space
 *  
 */
__kernel
void
redux_add(
    __global REAL *x, unsigned long N, unsigned long ipt,
    __global REAL *r,
    __local double *lrspace,
    __global double *grspace
    ) {

  size_t 
    G0 = get_global_size(0),
    G1 = get_global_size(1),
    L0 = get_local_size(0),
    L1 = get_local_size(1);

  //thread indicies
  size_t tg0 = get_global_id(0),
         tg1 = get_global_id(1),
         tg = G0*tg0 + tg1,
         tl0 = get_local_id(0),
         tl1 = get_local_id(1),
         tl = L0*tl0 + tl1;

  lrspace[tl] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  //if(tg >= N) { return; }

  __private REAL acc; //private variable used for accumulations
  acc = 0;

  //compact the input space into local the reduced space
  for(size_t i=tg*ipt; i<(tg*ipt + ipt) && i < N; ++i) {
      acc += x[i];
  }
  lrspace[tl] = acc;
  //acc = 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  //compact the reduced 2 dimensional space into a 1 dimensional space
  if(tl1 == 0) { 
    acc = 0;
    for(size_t i=0; i<L1; ++i) { acc += lrspace[L0*tl0 + i]; } 
    lrspace[L0*tl0] = acc;
    acc = 0;
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  //compact the reduced 1 dimensional space onto a point

  if(tl1 == 0 && tl0 == 0) {
    acc = 0;
    for(size_t i=0; i<L0; ++i) { acc += lrspace[L0 * i]; }

    //size_t gid = (size_t)(tg/(float)ipt) / (float)(L0 * L1);
    //size_t gid = 0;
    //put the point in the global reduction store so it can be accumulated
    size_t gid = get_num_groups(0)*get_group_id(0) + get_group_id(1);
    grspace[gid] = acc;
    acc = 0;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  //Accumulate the results of the work groups
  if(tg == 0) {
    acc = 0;
    //size_t lim =  ceil(((N/(float)(ipt))/(float)(L0*L1)));
    for(size_t i=0; i<(get_num_groups(0)*get_num_groups(1)); ++i) 
        { acc += grspace[i]; }
    *r = acc;
    //*r = get_local_size(0);
    //*r = lrspace[16*3];
  }

}
