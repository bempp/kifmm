echo "
===========================================
Running mpi_test_all_to_allv on 2 processes
===========================================
" && \
cargo mpirun --example mpi_test_all_to_allv --release -n 2 --features "mpi" && \
echo "
===========================================
Running mpi_test_all_to_allv on 4 processes
===========================================
" && \
cargo mpirun --example mpi_test_all_to_allv --release -n 4 --features "mpi" && \
echo "
===========================
Running single_node_laplace
===========================
" && \
cargo run --example single_node_laplace --release && \
echo "
========================================
Running mpi_test_metadata on 2 processes
========================================
" && \
cargo mpirun --example mpi_test_metadata --release -n 2 --features "mpi" && \
echo "
========================================
Running mpi_test_metadata on 4 processes
========================================
" && \
cargo mpirun --example mpi_test_metadata --release -n 4 --features "mpi" && \
echo "
====================
Running mpi_test_fmm
====================
" && \
cargo run --example mpi_test_fmm --release && \
echo "
==============================================
Running mpi_test_hyksort_unique on 2 processes
==============================================
" && \
cargo mpirun --example mpi_test_hyksort_unique --release -n 2 --features "mpi" && \
echo "
==============================================
Running mpi_test_hyksort_unique on 4 processes
==============================================
" && \
cargo mpirun --example mpi_test_hyksort_unique --release -n 4 --features "mpi" && \
echo "
==============================================
Running single_node_laplace_variable_expansion
==============================================
" && \
cargo run --example single_node_laplace_variable_expansion --release && \
echo "
=============================
Running single_node_helmholtz
=============================
" && \
cargo run --example single_node_helmholtz --release && \
echo "
================================================
Running mpi_test_tree_sample_sort on 2 processes
================================================
" && \
cargo mpirun --example mpi_test_tree_sample_sort --release -n 2 --features "mpi" && \
echo "
================================================
Running mpi_test_tree_sample_sort on 4 processes
================================================
" && \
cargo mpirun --example mpi_test_tree_sample_sort --release -n 4 --features "mpi" && \
echo "
=================================================
Running mpi_test_hyksort_redundant on 2 processes
=================================================
" && \
cargo mpirun --example mpi_test_hyksort_redundant --release -n 2 --features "mpi" && \
echo "
=================================================
Running mpi_test_hyksort_redundant on 4 processes
=================================================
" && \
cargo mpirun --example mpi_test_hyksort_redundant --release -n 4 --features "mpi" && \
echo "
============================
Running mpi_test_upward_pass
============================
" && \
cargo run --example mpi_test_upward_pass --release