# Custom Gstreamer Element (GPU Usage)
Deepstream SDK custom gst-element.
(NVMM memory --> GpuMat)
Sample Repo to get buffer from nvmm memory and map it to gpumat without going to/from cpu.
It uses cuda programming to draw a filled rectangle. (basic preprocessing)

## System Environment
- Hardware : Jetson Orin AGX
- OS : Ubuntu 20.04 
- JetPack - 5.1.1
  - TensorRT - 8.5.2.2
  - Deepstream - 6.2
  - OpenCV - 4.6.0 + CUDA


## Installation
```
git clone https://github.com/RajUpadhyay/Deepstream-Custom-Element
cd Deepstream-Custom-Element
```

Set your cuda version in the Makefile like this
`CUDA_VER?=11.4`

Now run make and sudo make install
```
make
sudo make install
```


## Please set up nvinfer by yourself or refer [this](https://github.com/RajUpadhyay/Detectron2-Deepstream) repo
```
gst-launch-1.0 uridecodebin3 uri=file:///video.mp4 ! nvvidconv ! m.sink_0 nvstreammux name=m width=1920 height=960 batch-size=1 ! dscustomplugin x1=100 y1=100 x2=400 y2=300 ! nvinfer config-file-path=config_infer.txt ! nvdsosd ! nvvidconv ! fpsdisplaysink sync=0
```
