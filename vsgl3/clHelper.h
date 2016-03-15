//////////////////////////////////////////////////////////////////
// clHelper.h Author: Vladimir Frolov, 2014, Graphics & Media Lab.
//////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <string>
#include <vector>

#include "../clew/clew.h"

static std::string ToString(int i)
{
  std::stringstream out;
  out << i;
  return out.str();
}

static std::string ToString(float i)
{
  std::stringstream out;
  out << i;
  return out.str();
}

static std::string ToString(unsigned int i)
{
  std::stringstream out;
  out << i;
  return out.str();
}


static void RunTimeError(const char* file, int line, std::string msg)
{
  throw std::runtime_error(std::string("Run time error at ") + file + std::string(", line ") + ToString(line) + ": " + msg);
}

#undef  RUN_TIME_ERROR
#define RUN_TIME_ERROR(e) (RunTimeError(__FILE__,__LINE__,(e)))

#undef  RUN_TIME_ERROR_AT
#define RUN_TIME_ERROR_AT(e, file, line) (RunTimeError((file),(line),(e)))

#undef  ASSERT
#ifdef  NDEBUG
#define ASSERT(_expression) ((void)0)
#else
#define ASSERT(_expression) if(!(_expression)) RUN_TIME_ERROR_AT("Assertion Failed", __FILE__, __LINE__)
#endif


#if defined (__APPLE__) || defined(MACOSX)
static const char* CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

int IsExtensionSupported(const char* support_str, const char* ext_string, size_t ext_buffer_size);
const char * getOpenCLErrorString(cl_int err);

void checkCLFun(cl_int cErr, char* file, int line);
int clglSharingIsSupported(cl_device_id device_id);

#define CHECK_CL(call) checkCLFun((call), __FILE__, __LINE__);

size_t roundWorkGroupSize(size_t a_size, size_t a_blockSize);


struct CLProgram
{

  CLProgram();
  CLProgram(cl_device_id a_devId, cl_context a_ctx, const std::string& cs_path, const std::string& options, 
            const std::string& includeFolderPath = "", const std::string& encryptedBufferPath = "", const std::string& binPath = "");

  virtual ~CLProgram();
  CLProgram& operator=(const CLProgram& a_prog);

  cl_kernel kernel(const std::string& name) const;

  cl_program program;

  void saveBinary(const std::string& a_fileName);

protected:

  bool Link();

  cl_context   m_ctx;
  cl_device_id m_dev;
  cl_int       m_lastErr;

  mutable size_t m_programLength;
  mutable int    m_refCounter;
  mutable std::map<std::string, cl_kernel> kernels;

};



struct PlatformDevPair
{
  PlatformDevPair(cl_device_id a_dev, cl_platform_id a_platform) : dev(a_dev), platform(a_platform) {}

  cl_device_id   dev;
  cl_platform_id platform;
};

std::vector<PlatformDevPair> listAllOpenCLDevices();

