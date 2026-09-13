# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Run after dependency bundling, which may copy libraries with build-tree rpaths.
file(GLOB_RECURSE _lfs_images
    "${CMAKE_INSTALL_PREFIX}/lib/*.dylib"
    "${CMAKE_INSTALL_PREFIX}/lib/*.so")
list(APPEND _lfs_images "${CMAKE_INSTALL_PREFIX}/bin/LichtFeld-Studio"
    "${CMAKE_INSTALL_PREFIX}/bin/python3")
foreach(_image IN LISTS _lfs_images)
    if(IS_SYMLINK "${_image}")
        continue()
    endif()
    execute_process(COMMAND /usr/bin/otool -l "${_image}"
        OUTPUT_VARIABLE _commands COMMAND_ERROR_IS_FATAL ANY)
    string(REGEX MATCHALL "cmd LC_RPATH\n[^\n]*\n[^\n]*" _rpaths "${_commands}")
    set(_retained_rpaths)
    foreach(_rpath IN LISTS _rpaths)
        string(REGEX REPLACE ".*\n[ ]*path ([^\n]*) \\(offset.*" "\\1" _rpath "${_rpath}")
        if(IS_ABSOLUTE "${_rpath}")
            execute_process(COMMAND /usr/bin/install_name_tool -delete_rpath "${_rpath}" "${_image}"
                COMMAND_ERROR_IS_FATAL ANY)
        else()
            list(APPEND _retained_rpaths "${_rpath}")
        endif()
    endforeach()
    get_filename_component(_image_dir "${_image}" DIRECTORY)
    file(RELATIVE_PATH _lib_path "${_image_dir}" "${CMAKE_INSTALL_PREFIX}/lib")
    set(_relative_rpath "@loader_path/${_lib_path}")
    if(NOT _relative_rpath IN_LIST _retained_rpaths)
        execute_process(COMMAND /usr/bin/install_name_tool -add_rpath "${_relative_rpath}" "${_image}"
            COMMAND_ERROR_IS_FATAL ANY)
    endif()
    get_filename_component(_name "${_image}" NAME)
    if(_image MATCHES "\\.dylib$")
        execute_process(COMMAND /usr/bin/install_name_tool -id "@rpath/${_name}" "${_image}"
            COMMAND_ERROR_IS_FATAL ANY)
    endif()
    execute_process(COMMAND /usr/bin/codesign --force --sign - "${_image}"
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
