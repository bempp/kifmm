#include "kifmm.h"
#include <iostream>
#include <cstdarg>


int main(void) {

    std::cout << "hello " << std::endl;

    hello_world();

    uintptr_t one = 1;

    std::cout << "answer " << add_from_rust(one, one) << std::endl;

    return 0;
}