gcc -I${PAPI_PREFIX}/include -L${PAPI_PREFIX}/lib -o codegen codegen.c
./codegen
rm -f codegen