#include "shapefill.cuh"

__global__ void drawRectFilled(uchar4* data, int step, int x1, int y1, int x2, int y2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= 1920 || y >= 960 )
        return;

    step = step/4;

    if(y >= y1 && y < y2 && x >= x1 && x <= x2)
    {
      data[y * step + x].x = 0;
      data[y * step + x].y = 0;
      data[y * step + x].z = 0;
      data[y * step + x].w = 0;
    }
}


void drawShapeFill(uchar4* d_data,
                   int step,
                   uint rows,
                   uint cols,
                   uint x1,
                   uint y1,
                   uint x2,
                   uint y2)
{
  dim3 block_size(32, 32);
  dim3 grid_size((cols + block_size.x - 1) / block_size.x, (rows + block_size.y - 1) / block_size.y);
  drawRectFilled<<<grid_size, block_size>>>(d_data, step, x1, y1, x2, y2);
  CUresult status = cuCtxSynchronize();
}
