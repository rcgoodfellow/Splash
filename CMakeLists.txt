CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)

SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_C_COMPILER clang)
SET(CMAKE_CXX_FLAGS "-g -std=c++11 -Wall -Wextra -fcolor-diagnostics -fpic -O0")
SET(CMAKE_C_FLAGS "-std=c11 -Wall -Wextra -fcolor-diagnostics -g")

SET(CMAKE_EXE_LINKER_FLAGS "-std=c++11")
SET(CMAKE_MODULE_LINKER_FLAGS "-std=c++11")
SET(CMAKE_SHARED_LINKER_FLAGS "-std=c++11")

ADD_LIBRARY(SplashCore SHARED 
  SparseMatrix.c SparseMatrix.h
  Vector.c Vector.h
  )
ADD_LIBRARY(SplashRuntime SHARED 
  Runtime.cxx Runtime.hxx 
  SparseMatrix.cxx SparseMatrix.hxx
  Utility.cxx Utility.hxx
  )

ADD_EXECUTABLE(LinearAlgebraBasicTests LinearAlgebraBasicTests.cxx)
TARGET_LINK_LIBRARIES(LinearAlgebraBasicTests SplashCore SplashRuntime OpenCL)

ADD_EXECUTABLE(DataTransferBench DataTransferBench.cxx)
TARGET_LINK_LIBRARIES(DataTransferBench SplashCore SplashRuntime OpenCL)

ADD_EXECUTABLE(VectorNormTests VectorNormTests.cxx)
TARGET_LINK_LIBRARIES(VectorNormTests SplashCore SplashRuntime OpenCL)

