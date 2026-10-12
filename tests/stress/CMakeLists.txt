# Race instrumentation makes this allocation-heavy stress test substantially slower. Keep a
# bounded but larger timeout for TSan, without weakening the normal Debug/Release test limit.
if(KORA_SANITIZE_THREAD)
  set(KORA_MEMORY_STRESS_TIMEOUT 120)
else()
  set(KORA_MEMORY_STRESS_TIMEOUT 29)
endif()

kora_add_test(kora_memory_stress_tests stress
  USES_VULKAN
  TIMEOUT ${KORA_MEMORY_STRESS_TIMEOUT}
  SOURCES test_memory_stress.cpp
)
target_link_libraries(kora_memory_stress_tests PRIVATE ${KORA_VULKAN_HEADERS})
