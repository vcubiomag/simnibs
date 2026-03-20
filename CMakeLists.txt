cmake_minimum_required(VERSION 3.25)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

find_package(CGAL REQUIRED COMPONENTS Core)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)
include(CGAL_TBB_support)
include(CGAL_Eigen3_support)

function(add_cython_target name pyx_source install_dest)
  cmake_parse_arguments(ARG "CPLUS" "" "LINK_LIBRARIES;INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS" ${ARGN})

  if(ARG_CPLUS)
    get_filename_component(output_file "${name}.cpp" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(cython_flags --cplus)
  else()
    get_filename_component(output_file "${name}.c" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(cython_flags "")
  endif()

  add_custom_command(
    OUTPUT "${output_file}"
    COMMAND Python::Interpreter -m cython ${cython_flags}
            "${CMAKE_CURRENT_SOURCE_DIR}/${pyx_source}" --output-file "${output_file}"
    DEPENDS "${pyx_source}"
    VERBATIM
  )

  python_add_library(${name} MODULE "${output_file}" WITH_SOABI)
  target_link_libraries(${name} PRIVATE Python::NumPy ${ARG_LINK_LIBRARIES})

  if(ARG_INCLUDE_DIRECTORIES)
    target_include_directories(${name} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
  endif()

  if(ARG_COMPILE_DEFINITIONS)
    target_compile_definitions(${name} PRIVATE ${ARG_COMPILE_DEFINITIONS})
  endif()

  install(TARGETS ${name} DESTINATION "${install_dest}")
endfunction()

# Pure Cython modules
add_cython_target(cython_msh simnibs/mesh_tools/cython_msh.pyx simnibs/mesh_tools/)
add_cython_target(_marching_cubes_lewiner_cy simnibs/segmentation/_marching_cubes_lewiner_cy.pyx simnibs/segmentation/)
add_cython_target(_thickness simnibs/segmentation/_thickness.pyx simnibs/segmentation/)

add_cython_target(_sanlm simnibs/segmentation/_sanlm.pyx simnibs/segmentation/
  INCLUDE_DIRECTORIES simnibs/segmentation
)

# CGAL Cython modules
set(CGAL_LIBS CGAL::CGAL CGAL::CGAL_Core CGAL::Eigen3_support CGAL::TBB_support)

foreach(name IN ITEMS create_mesh_surf create_mesh_vol cgal_misc)
  add_cython_target(${name} simnibs/mesh_tools/cgal/${name}.pyx simnibs/mesh_tools/cgal/
    CPLUS
    LINK_LIBRARIES ${CGAL_LIBS}
    INCLUDE_DIRECTORIES simnibs/mesh_tools/cgal
    COMPILE_DEFINITIONS CGAL_CONCURRENT_MESH_3=ON
  )
endforeach()

# Copy OS-specific external files
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(OS_FOLDER "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(OS_FOLDER "osx")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(OS_FOLDER "win")
else()
  message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

add_custom_target(copy_os_specific_externals ALL)

foreach(SUB_DIR IN ITEMS bin include lib)
  set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/external/${SUB_DIR}/${OS_FOLDER}")
  set(DEST_DIR "${SKBUILD_PLATLIB_DIR}/simnibs/external/${SUB_DIR}/${OS_FOLDER}")
  if(IS_DIRECTORY "${SOURCE_DIR}")
    add_custom_command(
      TARGET copy_os_specific_externals
      COMMAND ${CMAKE_COMMAND} -E make_directory "${DEST_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${SOURCE_DIR}/" "${DEST_DIR}"
      COMMENT "Copying ${OS_FOLDER} externals: ${SUB_DIR}"
      VERBATIM
    )
  endif()
endforeach()
