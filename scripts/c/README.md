# CRAY Profile Analysis

Make sure perftools module is loaded

```bash
module load perftools
```

#### Build

Using example of `weak_scaling/` script

```bash
cd weak_scaling
```

```bash
cmake -S . -B build -DENABLE_CRAYPAT=ON
cmake --build build
```

Run `pat_build` for instrumentation

```bash
pat_build ./build/weak_scaling
```

Should return a binary `weak_scaling+pat`, with instrumentation enabled.
