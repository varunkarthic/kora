execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "VK_ICD_FILENAMES=kora-no-such-vulkan-icd.json"
    "${KORA_INFO_EXECUTABLE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)

if(result EQUAL 0)
  message(FATAL_ERROR "kora-info succeeded even though no Vulkan ICD is available")
endif()

if(NOT "${output}${error}" MATCHES "Vulkan")
  message(FATAL_ERROR "kora-info failed without a useful Vulkan error message: ${output}${error}")
endif()
