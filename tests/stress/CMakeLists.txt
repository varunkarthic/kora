kora_add_test(kora_memory_stress_tests stress
  USES_VULKAN
  TIMEOUT 29
  SOURCES test_memory_stress.cpp
)
target_link_libraries(kora_memory_stress_tests PRIVATE ${KORA_VULKAN_HEADERS})
