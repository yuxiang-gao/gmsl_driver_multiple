#This config is a shortcut to try different paths where the real config might be
include(${CMAKE_CURRENT_LIST_DIR}/CMake/glfw3Config.cmake OPTIONAL RESULT_VARIABLE res1)
include(${CMAKE_CURRENT_LIST_DIR}/lib/cmake/glfw/glfw3Config.cmake OPTIONAL RESULT_VARIABLE res2)

if((NOT res1) AND (NOT res2))
	message(SEND_ERROR "Could not find real module config file.")
endif()