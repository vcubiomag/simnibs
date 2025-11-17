cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(CGAL REQUIRED COMPONENTS Core)

include(CGAL_Eigen3_support)
include(CGAL_TBB_support)

add_custom_command(
    OUTPUT cython_msh.c
    COMMAND Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/mesh_tools/cython_msh.pyx" --output-file cython_msh.c
    DEPENDS simnibs/mesh_tools/cython_msh.pyx
    VERBATIM
)

add_custom_command(
    OUTPUT _marching_cubes_lewiner_cy.c
    COMMAND Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/segmentation/_marching_cubes_lewiner_cy.pyx" --output-file _marching_cubes_lewiner_cy.c
    DEPENDS simnibs/segmentation/_marching_cubes_lewiner_cy.pyx
    VERBATIM
)

add_custom_command(
    OUTPUT _thickness.c
    COMMAND Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/segmentation/_thickness.pyx" --output-file _thickness.c
    DEPENDS simnibs/segmentation/_thickness.pyx
    VERBATIM
)

add_custom_command(
    OUTPUT _cat_c_utils.c
    COMMAND Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/segmentation/_cat_c_utils.pyx" --output-file _cat_c_utils.c
    DEPENDS simnibs/segmentation/_cat_c_utils.pyx
    VERBATIM
)

add_custom_command(
    OUTPUT create_mesh_surf.cpp
    COMMAND Python::Interpreter -m cython --cplus "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/mesh_tools/cgal/create_mesh_surf.pyx" --output-file create_mesh_surf.cpp
    DEPENDS simnibs/mesh_tools/cgal/create_mesh_surf.pyx
    VERBATIM
)

add_custom_command(
    OUTPUT create_mesh_vol.cpp
    COMMAND Python::Interpreter -m cython --cplus "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/mesh_tools/cgal/create_mesh_vol.pyx" --output-file create_mesh_vol.cpp
    DEPENDS simnibs/mesh_tools/cgal/create_mesh_vol.pyx
    VERBATIM
)

add_custom_command(
    OUTPUT cgal_misc.cpp
    COMMAND Python::Interpreter -m cython --cplus "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/mesh_tools/cgal/cgal_misc.pyx" --output-file cgal_misc.cpp
    DEPENDS simnibs/mesh_tools/cgal/cgal_misc.pyx
    VERBATIM
)

add_custom_command(
    OUTPUT polygon_mesh_processing.cpp
    COMMAND Python::Interpreter -m cython --cplus "${CMAKE_CURRENT_SOURCE_DIR}/simnibs/mesh_tools/cgal/polygon_mesh_processing.pyx" --output-file polygon_mesh_processing.cpp
    DEPENDS simnibs/mesh_tools/cgal/polygon_mesh_processing.pyx
    VERBATIM
)

python_add_library(cython_msh MODULE cython_msh.c WITH_SOABI)
target_link_libraries(cython_msh PRIVATE Python::NumPy)

python_add_library(_marching_cubes_lewiner_cy MODULE _marching_cubes_lewiner_cy.c WITH_SOABI)
target_link_libraries(_marching_cubes_lewiner_cy PRIVATE Python::NumPy)

python_add_library(_thickness MODULE _thickness.c WITH_SOABI)
target_link_libraries(_thickness PRIVATE Python::NumPy)

python_add_library(_cat_c_utils MODULE _cat_c_utils.c simnibs/segmentation/cat_c_utils/genus0.c WITH_SOABI)
target_link_libraries(_cat_c_utils PRIVATE Python::NumPy)
target_include_directories(_cat_c_utils PRIVATE simnibs/segmentation/cat_c_utils)

python_add_library(create_mesh_surf MODULE create_mesh_surf.cpp WITH_SOABI)
target_link_libraries(create_mesh_surf PRIVATE Python::NumPy CGAL::CGAL CGAL::CGAL_Core CGAL::Eigen3_support CGAL::TBB_support)
target_include_directories(create_mesh_surf PRIVATE simnibs/mesh_tools/cgal)
target_compile_definitions(create_mesh_surf PRIVATE CGAL_CONCURRENT_MESH_3=ON)

python_add_library(create_mesh_vol MODULE create_mesh_vol.cpp WITH_SOABI)
target_link_libraries(create_mesh_vol PRIVATE Python::NumPy CGAL::CGAL CGAL::CGAL_Core CGAL::Eigen3_support CGAL::TBB_support)
target_include_directories(create_mesh_vol PRIVATE simnibs/mesh_tools/cgal)
target_compile_definitions(create_mesh_vol PRIVATE CGAL_CONCURRENT_MESH_3=ON)

python_add_library(cgal_misc MODULE cgal_misc.cpp WITH_SOABI)
target_link_libraries(cgal_misc PRIVATE Python::NumPy CGAL::CGAL CGAL::CGAL_Core CGAL::Eigen3_support CGAL::TBB_support)
target_include_directories(cgal_misc PRIVATE simnibs/mesh_tools/cgal)
target_compile_definitions(cgal_misc PRIVATE CGAL_CONCURRENT_MESH_3=ON)

python_add_library(polygon_mesh_processing MODULE polygon_mesh_processing.cpp WITH_SOABI)
target_link_libraries(polygon_mesh_processing PRIVATE Python::NumPy CGAL::CGAL CGAL::CGAL_Core CGAL::Eigen3_support CGAL::TBB_support)
target_include_directories(polygon_mesh_processing PRIVATE simnibs/mesh_tools/cgal)
target_compile_definitions(polygon_mesh_processing PRIVATE CGAL_CONCURRENT_MESH_3=ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(OS_FOLDER "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(OS_FOLDER "osx")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(OS_FOLDER "win")
else()
    message(ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}. No external files will be copied.")
    set(OS_FOLDER "")
endif()

add_custom_target(
    copy_os_specific_externals
    ALL
)

if(OS_FOLDER)
    set(SUB_DIRS "bin" "include" "lib")
    foreach(SUB_DIR IN LISTS SUB_DIRS)
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
endif()

install(TARGETS cython_msh DESTINATION simnibs/mesh_tools/)
install(TARGETS _marching_cubes_lewiner_cy DESTINATION simnibs/segmentation/)
install(TARGETS _thickness DESTINATION simnibs/segmentation/)
install(TARGETS _cat_c_utils DESTINATION simnibs/segmentation/)
install(TARGETS create_mesh_surf DESTINATION simnibs/mesh_tools/cgal/)
install(TARGETS create_mesh_vol DESTINATION simnibs/mesh_tools/cgal/)
install(TARGETS cgal_misc DESTINATION simnibs/mesh_tools/cgal/)
install(TARGETS polygon_mesh_processing DESTINATION simnibs/mesh_tools/cgal/)
