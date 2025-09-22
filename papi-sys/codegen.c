#include <papi.h>
#include <stdio.h>

#define CODEGEN_DEFINE(var) codegen_define(#var, var)

void codegen_define(const char *name, const int val) {
    printf("pub const %s: ::std::os::raw::c_int = %d;\n", name, val);
}

void codegen() {
    CODEGEN_DEFINE(PAPI_VER_CURRENT);
    CODEGEN_DEFINE(PAPI_NATIVE_MASK);
}

int main(void) {
    codegen();
    return 0;
}