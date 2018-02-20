#include <iostream>
#include <vector>
#include <algorithm>

#include "clew/clew.h"
#include "Timer.h"
#include "vsgl3/clHelper.h"

#include "bitonic_sort_cpu.h"
#include "bitonic_sort_gpu.h"

int main(int argc, const char** argv)
{
  clewInit(L"opencl.dll");

  std::vector<int2> data(1024*1024*16); // 1024*1024*16

  std::cout << "initialising data ... " << std::endl;

  for (int i = 0; i < data.size(); i++)
  {
    data[i].x = (rand() + 100*rand());
    data[i].y = i;
  }

  std::cout << "making copyes ... " << std::endl;

  std::vector<int2> data2 = data;
  std::vector<int2> data3 = data;

  std::cout << "run std::sort ... " << std::endl;

  // CPU sort
  //
  Timer myTimer(true);

  std::sort(data.begin(), data.end(), MyCompare());

  float time1 = myTimer.getElapsed()*1000.0f;
  myTimer.start();

  std::cout << "run GPU sort ... " << std::endl;

  // GPU sort
  //
  try
  {
    std::vector<PlatformDevPair> devList = listAllOpenCLDevices();

    auto device   = devList[0].dev;
    auto platform = devList[0].platform;

    cl_int ciErr1 = CL_SUCCESS;
    auto ctx = clCreateContext(0, 1, &device, NULL, NULL, &ciErr1);

    if (ciErr1 != CL_SUCCESS)
      RUN_TIME_ERROR("Error in clCreateContext");

    auto cmdQueue = clCreateCommandQueue(ctx, device, 0, &ciErr1);

    if (ciErr1 != CL_SUCCESS)
      RUN_TIME_ERROR("Error in clCreateCommandQueue");

    std::string options     = "";
    std::string sshaderpath = "sort.cl";

    CLProgram bitonicProgs = CLProgram(device, ctx, sshaderpath.c_str(), options.c_str(), "");

    cl_kernel bitonicPassK = bitonicProgs.kernel("bitonic_pass_kernel");
    cl_kernel bitonicOpt   = bitonicProgs.kernel("bitonic_512");
    cl_kernel bitonicOpt2  = bitonicProgs.kernel("bitonic_1024");
    cl_kernel bitonicOpt3  = bitonicProgs.kernel("bitonic_2048");

    auto gpuData = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int2)*data3.size(), &data3[0], &ciErr1);

    //
    //
    CHECK_CL(clEnqueueWriteBuffer(cmdQueue, gpuData, CL_TRUE, 0, sizeof(int2)*data2.size(), &data2[0], 0, NULL, NULL));

    clFinish(cmdQueue);

    myTimer.start();

    {
      BitonicCLArgs args;
      args.dev          = device;
      args.cmdQueue     = cmdQueue;
      args.bitonicPassK = bitonicPassK;
      args.bitonic512   = nullptr;  // bitonic_sort_gpu_simple don't use shmem kernel
      args.bitonic1024  = nullptr;  // bitonic_sort_gpu_simple don't use shmem kernel
      args.bitonic2048  = nullptr;
      bitonic_sort_gpu_simple(gpuData, int(data2.size()), args);
    }

    clFinish(cmdQueue);

    float time2 = myTimer.getElapsed()*1000.0f;

    CHECK_CL(clEnqueueReadBuffer(cmdQueue, gpuData, CL_TRUE, 0, sizeof(int2)*data2.size(), &data2[0], 0, NULL, NULL));

    //
    //
    bool passed = true;
    int faileId = -1;
    for (int i = 0; i < data.size(); i++)
    {
      if (data[i].x != data2[i].x)
      {
        faileId = i;
        passed = false;
        break;
      }
    }

    if (passed)
      std::cout << "gpu test sort simple\tPASSED!" << std::endl;
    else
      std::cout << "gpu test sort simple\tFAILED! (" << faileId << ")" << std::endl;

    // 
    //
    CHECK_CL(clEnqueueWriteBuffer(cmdQueue, gpuData, CL_TRUE, 0, sizeof(int2)*data3.size(), &data3[0], 0, NULL, NULL));

    clFinish(cmdQueue);

    myTimer.start();

    {
      BitonicCLArgs args;
      args.dev          = device;
      args.cmdQueue     = cmdQueue;
      args.bitonicPassK = bitonicPassK;
      args.bitonic512   = bitonicOpt;
      args.bitonic1024  = bitonicOpt2;
      args.bitonic2048  = bitonicOpt3;

      bitonic_sort_gpu(gpuData, int(data3.size()), args);
    }

    clFinish(cmdQueue);

    float time3 = myTimer.getElapsed()*1000.0f;

    CHECK_CL(clEnqueueReadBuffer(cmdQueue, gpuData, CL_TRUE, 0, sizeof(int2)*data3.size(), &data3[0], 0, NULL, NULL));

    //
    //
    bool passed2 = true;
    faileId = -1;
    for (int i = 0; i < data.size(); i++)
    {
      if (data[i].x != data3[i].x)
      {
        faileId = i;
        passed2 = false;
        break;
      }
    }

    if (passed2)
      std::cout << "gpu test sort opt\tPASSED!" << std::endl;
    else
      std::cout << "gpu test sort opt\tFAILED! (" << faileId << ")" << std::endl;

    std::cout << std::endl;
    std::cout << "[CPU]: std::sort time      = " << time1 << " ms" << std::endl;
    std::cout << "[GPU]: bitonic simple time = " << time2 << " ms" << std::endl;
    std::cout << "[GPU]: bitonic opt time    = " << time3 << " ms" << std::endl;

    clReleaseMemObject(gpuData);

    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(ctx);
  }
  catch (std::runtime_error e)
  {
    std::cout << e.what() << std::endl;
  }
  catch (...)
  {
    std::cout << "unknown error" << std::endl;
  }


  return 0;
}

