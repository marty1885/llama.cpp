# dependencies

find_package(Threads REQUIRED)

# third-party

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(interp)
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(vdot)
    endif()
endif()
