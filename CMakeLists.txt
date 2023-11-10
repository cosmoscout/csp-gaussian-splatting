# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_GAUSSIAN_SPLATTING "Enable compilation of this plugin" ON)

if (NOT CSP_GAUSSIAN_SPLATTING)
  return()
endif()

# add cuda support ---------------------------------------------------------------------------------

find_package(CUDA 11.0 REQUIRED)

# compile the rasterizer lib -----------------------------------------------------------------------

add_subdirectory(externals/diff-gaussian-rasterization)

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp src/renderer/GaussianData.cpp src/renderer/SurfaceRenderer.cpp)

# Resoucre files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOUCRE_FILES textures/*)

add_library(csp-gaussian-splatting SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOUCRE_FILES}
)

target_link_libraries(csp-gaussian-splatting
  PUBLIC
    cs-core
    CudaRasterizer
    ${CUDA_LIBRARIES}
)

target_include_directories(csp-gaussian-splatting PUBLIC ${CUDA_INCLUDE_DIRS})

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-gaussian-splatting PROPERTY FOLDER "plugins")

# We mark all resource files as "header" in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOUCRE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES 
  ${SOURCE_FILES} ${HEADER_FILES} ${RESOUCRE_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS   csp-gaussian-splatting DESTINATION "share/plugins")
install(DIRECTORY "shaders"              DESTINATION "share/resources")