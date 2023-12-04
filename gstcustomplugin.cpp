/**
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include "gstdscustomplugin.h"
#include <sys/time.h>

#include "shapefill.cuh"

GST_DEBUG_CATEGORY_STATIC (gst_customplugin_debug);
#define GST_CAT_DEFAULT gst_customplugin_debug
#define USE_EGLIMAGE 1
static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_HEIGHT,
  PROP_WIDTH,
  PROP_X1,
  PROP_X2,
  PROP_Y1,
  PROP_Y2,
  PROP_GPU_DEVICE_ID,
};

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)  \
  ({ int _errtype=0;\
   do {  \
    if ((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) && \
        (surface->gpuId != object->gpu_id))  { \
    GST_ELEMENT_ERROR (object, RESOURCE, FAILED, \
        ("Input surface gpu-id doesnt match with configured gpu-id for element," \
         " please allocate input using unified memory, or use same gpu-ids"),\
        ("surface-gpu-id=%d,%s-gpu-id=%d",surface->gpuId,GST_ELEMENT_NAME(object),\
         object->gpu_id)); \
    _errtype = 1;\
    } \
    } while(0); \
    _errtype; \
  })

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_HEIGHT 960
#define DEFAULT_WIDTH 1920
#define DEFAULT_X1 0
#define DEFAULT_X2 0
#define DEFAULT_Y1 0
#define DEFAULT_Y2 0
#define DEFAULT_GPU_ID 0

#define CHECK_NPP_STATUS(npp_status,error_str) do { \
  if ((npp_status) != NPP_SUCCESS) { \
    g_print ("Error: %s in %s at line %d: NPP Error %d\n", \
        error_str, __FILE__, __LINE__, npp_status); \
    goto error; \
  } \
} while (0)

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dscustomplugin_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dscustomplugin_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dscustomplugin_parent_class parent_class
G_DEFINE_TYPE (GstDsCustomPlugin, gst_dscustomplugin, GST_TYPE_BASE_TRANSFORM);

static void gst_dscustomplugin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dscustomplugin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_dscustomplugin_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dscustomplugin_start (GstBaseTransform * btrans);
static gboolean gst_dscustomplugin_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_dscustomplugin_transform_ip (GstBaseTransform *
    btrans, GstBuffer * inbuf);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dscustomplugin_class_init (GstDsCustomPluginClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  /* Indicates we want to use DS buf api */
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dscustomplugin_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dscustomplugin_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dscustomplugin_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dscustomplugin_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dscustomplugin_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_dscustomplugin_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_HEIGHT,
      g_param_spec_uint ("height",
          "image height",
          "set the image height, default: 960", 
          0, 960, DEFAULT_HEIGHT, GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_WIDTH,
      g_param_spec_uint ("width",
          "image width",
          "set the image width, default: 1920", 
          0, 1920, DEFAULT_WIDTH, GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_X1,
      g_param_spec_uint ("x1",
          "x1/left",
          "set the x1 value, default: 0", 
          0, 1920, DEFAULT_X1, GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_Y1,
      g_param_spec_uint ("y1",
          "y1/top",
          "set the y1 value, default: 0", 
          0, 960, DEFAULT_Y1, GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_X2,
      g_param_spec_uint ("x2",
          "x2/right",
          "set the x2 value, default: 0", 
          0, 1920, DEFAULT_X2, GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_Y2,
      g_param_spec_uint ("y2",
          "y2/bottom",
          "set the y2 value, default: 0", 
          0, 960, DEFAULT_Y2, GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));


  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dscustomplugin_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dscustomplugin_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsCustomPlugin plugin",
      "DsCustomPlugin Plugin",
      "Process a 3rdparty customplugin algorithm",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_dscustomplugin_init (GstDsCustomPlugin * dscustomplugin)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (dscustomplugin);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  dscustomplugin->unique_id = DEFAULT_UNIQUE_ID;
  dscustomplugin->height = DEFAULT_HEIGHT;
  dscustomplugin->width = DEFAULT_WIDTH;
  dscustomplugin->x1 = DEFAULT_X1;
  dscustomplugin->x2 = DEFAULT_X2;
  dscustomplugin->y1 = DEFAULT_Y1;
  dscustomplugin->y2 = DEFAULT_Y2;
  dscustomplugin->gpu_id = DEFAULT_GPU_ID;

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_dscustomplugin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsCustomPlugin *dscustomplugin = GST_DSCUSTOMPLUGIN (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      dscustomplugin->unique_id = g_value_get_uint (value);
      break;
    case PROP_HEIGHT:
      dscustomplugin->height = g_value_get_uint (value);
      break;
    case PROP_WIDTH:
      dscustomplugin->width = g_value_get_uint (value);
      break;
    case PROP_X1:
      dscustomplugin->x1 = g_value_get_uint (value);
      break;
    case PROP_X2:
      dscustomplugin->x2 = g_value_get_uint (value);
      break;
    case PROP_Y1:
      dscustomplugin->y1 = g_value_get_uint (value);
      break;
    case PROP_Y2:
      dscustomplugin->y2 = g_value_get_uint (value);
      break;
    case PROP_GPU_DEVICE_ID:
      dscustomplugin->gpu_id = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_dscustomplugin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsCustomPlugin *dscustomplugin = GST_DSCUSTOMPLUGIN (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, dscustomplugin->unique_id);
      break;
    case PROP_HEIGHT:
      g_value_set_uint (value, dscustomplugin->height);
      break;
    case PROP_WIDTH:
      g_value_set_uint (value, dscustomplugin->width);
      break;
    case PROP_X1:
      g_value_set_uint (value, dscustomplugin->x1);
      break;
    case PROP_X2:
      g_value_set_uint (value, dscustomplugin->y1);
      break;
    case PROP_Y1:
      g_value_set_uint (value, dscustomplugin->x2);
      break;
    case PROP_Y2:
      g_value_set_uint (value, dscustomplugin->y2);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, dscustomplugin->gpu_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_dscustomplugin_start (GstBaseTransform * btrans)
{
  GstDsCustomPlugin *dscustomplugin = GST_DSCUSTOMPLUGIN (btrans);

  GstQuery *queryparams = NULL;
  guint batch_size = 1;

  CHECK_CUDA_STATUS (cudaSetDevice (dscustomplugin->gpu_id),
      "Unable to set cuda device");

  dscustomplugin->batch_size = 1;
  queryparams = gst_nvquery_batch_size_new ();
  if (gst_pad_peer_query (GST_BASE_TRANSFORM_SINK_PAD (btrans), queryparams)
      || gst_pad_peer_query (GST_BASE_TRANSFORM_SRC_PAD (btrans), queryparams)) {
    if (gst_nvquery_batch_size_parse (queryparams, &batch_size)) {
      dscustomplugin->batch_size = batch_size;
    }
  }
  GST_DEBUG_OBJECT (dscustomplugin, "Setting batch-size %d \n",
      dscustomplugin->batch_size);
  gst_query_unref (queryparams);

  return TRUE;

error:
  return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_dscustomplugin_stop (GstBaseTransform * btrans)
{
  GstDsCustomPlugin *dscustomplugin = GST_DSCUSTOMPLUGIN (btrans);
  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_dscustomplugin_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsCustomPlugin *dscustomplugin = GST_DSCUSTOMPLUGIN (btrans);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&dscustomplugin->video_info, incaps);
  return TRUE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dscustomplugin_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstDsCustomPlugin *dscustomplugin = GST_DSCUSTOMPLUGIN (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_ERROR;

  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList * l_frame = NULL;

  dscustomplugin->frame_num++;
  CHECK_CUDA_STATUS (cudaSetDevice (dscustomplugin->gpu_id),
      "Unable to set cuda device");

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_print ("Error: Failed to map gst buffer\n");
    goto error;
  }

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (dscustomplugin));
  surface = (NvBufSurface *) in_map_info.data;
  GST_DEBUG_OBJECT (dscustomplugin,
      "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
      dscustomplugin->frame_num, surface);

  if (CHECK_NVDS_MEMORY_AND_GPUID (dscustomplugin, surface))
    goto error;

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dscustomplugin, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  // std::cout << dscustomplugin->x1 << " " << dscustomplugin->y1 << " " << dscustomplugin->x2 << " " << dscustomplugin->y2 << std::endl;

  {
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
      frame_meta = (NvDsFrameMeta *) (l_frame->data);
      if (USE_EGLIMAGE) {
        if (NvBufSurfaceMapEglImage (surface, 0) !=0 ) {
            g_printerr("NvBufSurfaceMapEglImage error\n");
            return GST_FLOW_ERROR;
        }

        static bool create_filter = true;
        static cv::Ptr< cv::cuda::Filter > filter;
        CUresult status;
        CUeglFrame eglFrame;
        CUgraphicsResource pResource = NULL;
        cudaFree(0);
        status = cuGraphicsEGLRegisterImage(&pResource,
                                            surface->surfaceList[0].mappedAddr.eglImage,
                                            CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
        status = cuCtxSynchronize();

        cv::cuda::GpuMat d_mat(960, 1920, CV_8UC4, eglFrame.frame.pPitch[0]);

        // Ensure x1 <= x2 and y1 <= y2
        uint x1 = std::min(dscustomplugin->x1, dscustomplugin->x2);
        uint x2 = std::max(dscustomplugin->x1, dscustomplugin->x2);
        uint y1 = std::min(dscustomplugin->y1, dscustomplugin->y2);
        uint y2 = std::max(dscustomplugin->y1, dscustomplugin->y2);

        uchar4 *d_data = d_mat.ptr<uchar4>();
        cudaMemcpy(d_data, d_mat.data, sizeof(uchar4) * d_mat.cols * d_mat.rows, cudaMemcpyDeviceToDevice);

        drawShapeFill(d_data, d_mat.step, d_mat.rows, d_mat.cols, x1, y1, x2 ,y2);

        cudaMemcpy(d_mat.data, d_data, sizeof(uchar4) * d_mat.cols * d_mat.rows, cudaMemcpyDeviceToDevice);
        cudaFree(d_data);
        // make pixels between x1 x2 y1 y2 black
        // Create a kernel to set pixels in the specified region to black

        status = cuCtxSynchronize();
        status = cuGraphicsUnregisterResource(pResource);
        // NvBufSurfTransform (dsexample->inter_buf, &ip_surf, &transform_params);
        // Destroy the EGLImage
        NvBufSurfaceUnMapEglImage (surface, 0);

      }
    }
  }
  flow_ret = GST_FLOW_OK;

error:
  nvds_set_output_system_timestamp (inbuf, GST_ELEMENT_NAME (dscustomplugin));
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
dscustomplugin_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_customplugin_debug, "dscustomplugin", 0,
      "dscustomplugin plugin");

  return gst_element_register (plugin, "dscustomplugin", GST_RANK_PRIMARY,
      GST_TYPE_DSCUSTOMPLUGIN);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dscustomplugin,
    DESCRIPTION, dscustomplugin_plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE, URL)
