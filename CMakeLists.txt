cmake_minimum_required(VERSION 3.0)
project(mgg)
find_package(OpenCV REQUIRED)

set(SOURCES 
	main.cpp 
	utils.cpp
	)

set(HEADERS
	utils.h
	)

if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O2 -march=corei7-avx")
endif()

add_executable(mgg main.cpp ${SOURCES} ${HEADERS})
target_link_libraries(mgg ${OpenCV_LIBS})