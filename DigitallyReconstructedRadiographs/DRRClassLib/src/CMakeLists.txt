project(DRRGenerate)

add_library(DRRGenerate DRRGenerateClass.cpp DRRGenerateClass.h)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

target_link_libraries(DRRGenerate ${ITK_LIBRARIES})


set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
set(BUILD_SHARED_LIBS TRUE)


