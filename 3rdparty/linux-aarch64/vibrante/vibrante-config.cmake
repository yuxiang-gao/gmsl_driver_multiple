## CMake configuration file
#
# INPUT:
#  VIBRANTE_PDK:STRING - contains path to the vibrante PDK
#
# OUTPUT:
#  vibrante_VERSION
#  vibrante_VERSION_MAJOR
#  vibrante_VERSION_MINOR
#  vibrante_VERSION_PATCH
#  vibrante_VERSION_BUILD
#

set(LIBNAME vibrante)

# VIBRANTE_PDK variable must be set with proper PDK path
if (NOT DEFINED VIBRANTE_PDK)
    message(FATAL_ERROR "Vibrante-package need to have VIBRANTE_PDK variable specified")
endif()

if(EXISTS "${VIBRANTE_PDK}/lib-target/version-nv-pdk.txt")
    set(VIBRANTE_PDK_FILE "${VIBRANTE_PDK}/lib-target/version-nv-pdk.txt")
elseif(EXISTS "${VIBRANTE_PDK}/lib-target/version-nv-sdk.txt")
    set(VIBRANTE_PDK_FILE "${VIBRANTE_PDK}/lib-target/version-nv-sdk.txt")
endif()

if(VIBRANTE_PDK_FILE)
   file(READ ${VIBRANTE_PDK_FILE} version-nv-pdk)
   if(${version-nv-pdk} MATCHES "^(.+)-[0123456789]+")
       set(VIBRANTE_PDK_BRANCH ${CMAKE_MATCH_1})
       message(STATUS "VIBRANTE_PDK_BRANCH = ${VIBRANTE_PDK_BRANCH}")
   else()
       message(FATAL_ERROR "Can't determine PDK branch for PDK ${VIBRANTE_PDK}")
   endif()
else()
   message(FATAL_ERROR "Can't open ${VIBRANTE_PDK}/lib-target/version-nv-(pdk/sdk).txt for PDK branch detection")
endif()

# library version information
string(REPLACE "." ";" PDK_VERSION_LIST ${VIBRANTE_PDK_BRANCH})

# Some PDK's have less than three version numbers. Pad the list so we always
# have at least three numbers, allowing pre-existing logic depending on major,
# minor, patch versioning to work without modifications
list(LENGTH PDK_VERSION_LIST _PDK_VERSION_LIST_LENGTH)
while(_PDK_VERSION_LIST_LENGTH LESS 3)
  list(APPEND PDK_VERSION_LIST 0)
  math(EXPR _PDK_VERSION_LIST_LENGTH "${_PDK_VERSION_LIST_LENGTH} + 1")
endwhile()

set(_VERSION_PATCH 0)
set(_VERSION_BUILD 0)

list(LENGTH PDK_VERSION_LIST PDK_VERSION_LIST_LENGTH)
list(GET PDK_VERSION_LIST 0 _VERSION_MAJOR)
list(GET PDK_VERSION_LIST 1 _VERSION_MINOR)
if (PDK_VERSION_LIST_LENGTH GREATER 2)
    list(GET PDK_VERSION_LIST 2 _VERSION_PATCH)
endif()

if (PDK_VERSION_LIST_LENGTH GREATER 3)
    list(GET PDK_VERSION_LIST 3 _VERSION_BUILD)
endif()


string(REGEX MATCH "[0-9]+" ${LIBNAME}_VERSION_MAJOR ${_VERSION_MAJOR})
string(REGEX MATCH "[0-9]+" ${LIBNAME}_VERSION_MINOR ${_VERSION_MINOR})
string(REGEX MATCH "[0-9]+" ${LIBNAME}_VERSION_PATCH ${_VERSION_PATCH})
string(REGEX MATCH "[0-9]+" ${LIBNAME}_VERSION_BUILD ${_VERSION_BUILD})

set(${LIBNAME}_VERSION_STRING ${VIBRANTE_PDK_BRANCH})

# installation prefix
get_filename_component (CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component (_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/" ABSOLUTE)

# include directory
#
# Newer versions of CMake set the INTERFACE_INCLUDE_DIRECTORIES property
# of the imported targets. It is hence not necessary to add this path
# manually to the include search path for targets.
set (${LIBNAME}_LIBRARY_DIR "${_INSTALL_PREFIX}/lib")
set (${LIBNAME}_INCLUDE_DIR "${_INSTALL_PREFIX}/include")
set (${LIBNAME}_LIBRARY_VERSION_DIR "${_INSTALL_PREFIX}/lib/${${LIBNAME}_VERSION_STRING}")
set (${LIBNAME}_INCLUDE_VERSION_DIR "${_INSTALL_PREFIX}/include/${${LIBNAME}_VERSION_STRING}")

# alias for default import target to be compatible with older CMake package configurations
set (${LIBNAME}_LIBRARIES "${LIBNAME}")

# --------
# modifications based on PDK changes
set(_VIBRANTE_LIBRARIES X11 Xrandr Xinerama Xi Xcursor GLESv2 udev usb-1.0)

# new versioning scheme
if (${LIBNAME}_VERSION_MAJOR EQUAL 4 AND ${LIBNAME}_VERSION_MINOR GREATER 0 AND
    (NOT(    ${LIBNAME}_VERSION_MAJOR LESS 4
         OR (${LIBNAME}_VERSION_MAJOR EQUAL 4 AND ${LIBNAME}_VERSION_MINOR LESS 1)
         OR (${LIBNAME}_VERSION_MAJOR EQUAL 4 AND ${LIBNAME}_VERSION_MINOR EQUAL 1 AND ${LIBNAME}_VERSION_PATCH LESS 1))))
  list(APPEND _VIBRANTE_LIBRARIES nv_extimgdev)
  list(APPEND _VIBRANTE_LIBRARIES nv_embstatsplugin)
  message(STATUS "Vibrante >= 4.1.1.0 - append nv_extimgdev library")
  message(STATUS "Vibrante >= 4.1.1.0 - append nv_embstatsplugin library")
endif()

# >= 25.05.01 nv_extimgdev
if (NOT(    ${LIBNAME}_VERSION_MAJOR LESS 25
        OR (${LIBNAME}_VERSION_MAJOR EQUAL 25 AND ${LIBNAME}_VERSION_MINOR LESS 5)
        OR (${LIBNAME}_VERSION_MAJOR EQUAL 25 AND ${LIBNAME}_VERSION_MINOR EQUAL 5 AND ${LIBNAME}_VERSION_PATCH LESS 1)))
  list(APPEND _VIBRANTE_LIBRARIES nv_extimgdev)
  message(STATUS "Vibrante >= 25.05.01 - append nv_extimgdev library")
endif()

# >= 25.10.03 nv_embstatsplugin
if (NOT(    ${LIBNAME}_VERSION_MAJOR LESS 25
        OR (${LIBNAME}_VERSION_MAJOR EQUAL 25 AND ${LIBNAME}_VERSION_MINOR LESS 10)
        OR (${LIBNAME}_VERSION_MAJOR EQUAL 25 AND ${LIBNAME}_VERSION_MINOR EQUAL 10 AND ${LIBNAME}_VERSION_PATCH LESS 3)))
  list(APPEND _VIBRANTE_LIBRARIES nv_embstatsplugin)
  message(STATUS "Vibrante >= 25.10.01 - append nv_embstatsplugin library")
else()

endif()

list(APPEND _VIBRANTE_LIBRARIES rt dl)

# import targets
if(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 3.1.3)
    # Support for old cmake that did not support INTERFACE libraries
    set (${LIBNAME}_LIBRARIES ${_VIBRANTE_LIBRARIES})
    include_directories(${${LIBNAME}_INCLUDE_DIR})
    include_directories(${${LIBNAME}_INCLUDE_VERSION_DIR})
else()
    add_library(${LIBNAME} INTERFACE)

    # include directories
    target_include_directories(${LIBNAME} INTERFACE ${${LIBNAME}_INCLUDE_DIR})
    target_include_directories(${LIBNAME} INTERFACE ${${LIBNAME}_INCLUDE_VERSION_DIR})

    # add all libraries with absolute paths
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    foreach(LIB ${_VIBRANTE_LIBRARIES})
        set(FOUND_LIB "FOUND_LIB-NOTFOUND")
        find_library(FOUND_LIB ${LIB}
                    HINTS ${${LIBNAME}_LIBRARY_VERSION_DIR} ${${LIBNAME}_LIBRARY_DIR}
                    NO_DEFAULT_PATH)
        if(FOUND_LIB)
            target_link_libraries(${LIBNAME} INTERFACE ${FOUND_LIB})
            message(STATUS "Found vibrante lib: ${FOUND_LIB}")
        else()
            target_link_libraries(${LIBNAME} INTERFACE ${LIB})
        endif()
    endforeach()
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

    set (${LIBNAME}_LIBRARIES ${LIBNAME})
endif()

# unset private variables
unset (_INSTALL_PREFIX)
