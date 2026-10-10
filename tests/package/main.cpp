#include <iostream>
#include <kora/version.hpp>

int main() {
    std::cout << kora::version_string() << '\n';
    return 0;
}
