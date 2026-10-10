kora_add_test(kora_integration_tests integration
  USES_VULKAN
  SOURCES
    test_physical_device.cpp
    test_vulkan_instance.cpp
    test_vulkan_skip.cpp
)
target_link_libraries(kora_integration_tests PRIVATE ${KORA_VULKAN_HEADERS})

if(TARGET kora-info)
  add_test(NAME KoraInfo.ListsDevices COMMAND $<TARGET_FILE:kora-info>)
  set_tests_properties(KoraInfo.ListsDevices PROPERTIES
    LABELS integration
    PASS_REGULAR_EXPRESSION "Physical devices: [0-9]+")

  add_test(NAME KoraInfo.MissingVulkan
    COMMAND ${CMAKE_COMMAND}
      "-DKORA_INFO_EXECUTABLE=$<TARGET_FILE:kora-info>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_kora_info_failure.cmake")
  set_tests_properties(KoraInfo.MissingVulkan PROPERTIES LABELS integration)
endif()
