################################################################################
# Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#################################################################################
#enable this flag to use optimized dsexample plugin
#it can also be exported from command line

WITH_OPENCV?=1

CUDA_VER?=11.4
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CXX:= g++
NVCC:= nvcc

SRCS:= $(wildcard *.cpp)
CUSRCS:=$(wildcard *.cu)
INCS:= $(wildcard *.h)
LIB:=libnvdsgst_dscustomplugin.so

NVDS_VERSION:=6.2

CFLAGS+= -fPIC -DDS_VERSION=\"6.2.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes \
	 -I/usr/src/jetson_multimedia_api/include \
	 -I/usr/include/aarch64-linux-gnu \
	 -I/usr/include/gstreamer-1.0/gst \
	 -I/opt/nvidia/deepstream/deepstream-6.2/sources/includes

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	-L/usr/lib/aarch64-linux-gnu/ -lnvinfer_plugin \
	-L/usr/lib/aarch64-linux-gnu/tegra/ -lEGL -lGLESv2 \
	-L/usr/lib/aarch64-linux-gnu/tegra/ -lcuda -lnvbuf_utils \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-L/usr/lib/aarch64-linux-gnu/ -lopencv_cudaimgproc \
	-L/opt/nvidia/deepstream/deepstream-6.2/includes \
	-L/opt/nvidia/deepstream/deepstream-6.2/lib/ -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform -lnvinfer -lnvparsers -lnvonnxparser
OBJS:= $(SRCS:.cpp=.o) $(CUSRCS:.cu=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

ifeq ($(WITH_OPENCV),1)
CFLAGS+= -I /usr/local/include/opencv4
PKGS+= opencv4
endif

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cu Makefile
	$(NVCC) -c -o $@ -arch compute_87 --compiler-options '-fPIC' $<

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

$(DEP): $(DEP_FILES)
	$(MAKE) -C dsexample_lib/

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf *.so *.o
