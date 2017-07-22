## gflags CMake configuration version file

# -----------------------------------------------------------------------------
# library version
set (PACKAGE_VERSION "3.1.1")

# -----------------------------------------------------------------------------
# check compatibility

# Perform compatibility check here using the input CMake variables.
# See example in http://www.cmake.org/Wiki/CMake_2.6_Notes.

set (PACKAGE_VERSION_COMPATIBLE TRUE)
set (PACKAGE_VERSION_UNSUITABLE FALSE)

if ("${PACKAGE_FIND_VERSION_MAJOR}" EQUAL "3" AND
    "${PACKAGE_FIND_VERSION_MINOR}" EQUAL "1")
  set (PACKAGE_VERSION_EXACT TRUE)
else ()
  set (PACKAGE_VERSION_EXACT FALSE)
endif ()
