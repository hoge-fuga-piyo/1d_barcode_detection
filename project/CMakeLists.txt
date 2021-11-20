###############################
# Project
###############################

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB subdirs RELATIVE ${dir} ${dir}/*)

foreach(subdir ${subdirs})
  if(IS_DIRECTORY ${dir}/${subdir})
    message("subdir: ${subdir}")
    add_subdirectory(${subdir})
  endif()
endforeach()
