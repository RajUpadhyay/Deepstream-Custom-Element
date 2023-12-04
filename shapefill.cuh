#ifndef _DISABLEFILL_CUH_
#define _DISABLEFILL_CUH_

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * cuda-process API to draw black filled rectangles
 *
 * @param d_data : uchar* data of GpuMat image
 * @param step : step value of gpumat
 * @param rows : rows/height of the gpumat
 * @param cols : cols/width of the gpumat
 * @param x1 : x1/left value of rectangle
 * @param y1 : y1/top value of rectangle
 * @param x2 : x2/right value of rectangle
 * @param y2 : y2/bottom value of rectangle
 */
extern "C" void drawShapeFill(uchar4* d_data,
                                    int step,
                                    uint rows,
                                    uint cols,
                                    uint x1,
                                    uint y1,
                                    uint x2,
                                    uint y2);

#endif
