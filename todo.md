1. Go through todos and understand what's missing.
2. source to target setup for fft-m2l, no need to parallelise
3. helmholtz afterwards should be a copy and paste job

- software engineering (code-reorg, format imports, documentation, clippy, remove dep on Float trait)
- add tests for serialisation code
- add test for first_child_at_level
- add test for helmholtz upward pass and FMM
- add experimental config code
