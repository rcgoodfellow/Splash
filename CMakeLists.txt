CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)

SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_C_COMPILER clang)
SET(CMAKE_CXX_FLAGS "-g -std=c++11 -Wall -Wextra -fcolor-diagnostics")
SET(CMAKE_C_FLAGS "-std=c99 -Wall -Wextra -fcolor-diagnostics -g")

ADD_LIBRARY(SplashCore SparseMatrix.c SparseMatrix.h Utility.c Utility.h)

ADD_EXECUTABLE(LinearAlgebraBasicTests LinearAlgebraBasicTests.c)
TARGET_LINK_LIBRARIES(LinearAlgebraBasicTests SplashCore OpenCL)
