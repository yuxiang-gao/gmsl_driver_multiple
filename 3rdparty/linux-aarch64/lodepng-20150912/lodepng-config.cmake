#This config is a shortcut to try different paths where the real config might be
get_filename_component(this_file ${CMAKE_CURRENT_LIST_FILE} NAME)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/${this_file} OPTIONAL RESULT_VARIABLE res1)

if(NOT res1)
	message(SEND_ERROR "Could not find real module config file.")
endif()