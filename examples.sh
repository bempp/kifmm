echo "
===============================
Running mpi_test_ghost_exchange
===============================
" && \
cargo run --example mpi_test_ghost_exchange --release && \
echo "
=======================
Running mpi_test_layout
=======================
" && \
cargo run --example mpi_test_layout --release && \
echo "
=========================================
Running mpi_test_attach_charges_unordered
=========================================
" && \
cargo run --example mpi_test_attach_charges_unordered --release && \
echo "
====================
Running mpi_test_fmm
====================
" && \
cargo run --example mpi_test_fmm --release && \
echo "
===============
Running laplace
===============
" && \
cargo run --example laplace --release && \
echo "
====================================
Running mpi_test_sort on 2 processes
====================================
" && \
cargo mpirun --example mpi_test_sort --release -n 2 --features "mpi" && \
echo "
====================================
Running mpi_test_sort on 4 processes
====================================
" && \
cargo mpirun --example mpi_test_sort --release -n 4 --features "mpi" && \
echo "
=======================================
Running mpi_test_attach_charges_ordered
=======================================
" && \
cargo run --example mpi_test_attach_charges_ordered --release && \
echo "
====================
Running mpi_test_p2m
====================
" && \
cargo run --example mpi_test_p2m --release && \
echo "
================
Running mpi_tree
================
" && \
cargo run --example mpi_tree --release && \
echo "
==================================
Running laplace_variable_expansion
==================================
" && \
cargo run --example laplace_variable_expansion --release && \
echo "
=================
Running helmholtz
=================
" && \
cargo run --example helmholtz --release && \
echo "
============================
Running mpi_test_upward_pass
============================
" && \
cargo run --example mpi_test_upward_pass --release && \
echo "
======================
Running mpi_test_trees
======================
" && \
cargo run --example mpi_test_trees --release