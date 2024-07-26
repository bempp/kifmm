#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Static FMM type
 */
typedef enum FmmCType {
  FmmCType_Laplace32,
  FmmCType_Laplace64,
  FmmCType_Helmholtz32,
  FmmCType_Helmholtz64,
} FmmCType;

/**
 * M2L field translation mode
 */
typedef enum FmmTranslationCType {
  FmmTranslationCType_Blas,
  FmmTranslationCType_Fft,
} FmmTranslationCType;

/**
 * Scalar type
 */
typedef enum ScalarType {
  /**
   * Float
   */
  ScalarType_F32,
  /**
   * Double
   */
  ScalarType_F64,
  /**
   * Complex FLoat
   */
  ScalarType_C32,
  /**
   * Complex Double
   */
  ScalarType_C64,
} ScalarType;

/**
 * Runtime FMM type constructed from C
 */
typedef struct FmmEvaluator {
  enum FmmCType ctype;
  enum FmmTranslationCType ctranslation_type;
  void *data;
} FmmEvaluator;

/**
 * Potential data
 */
typedef struct Potential {
  /**
   * Length of underlying buffer, of length n_eval_mode*n_coordinates
   * currently only support n_eval_mode=1, i.e. potentials
   */
  uintptr_t len;
  /**
   * Pointer to underlying buffer
   */
  const void *data;
  /**
   * Associated scalar type
   */
  enum ScalarType scalar;
} Potential;

/**
 * Morton keys, used to describe octree boxes, each represented as unique u64.
 */
typedef struct MortonKeys {
  /**
   * Number of morton keys
   */
  uintptr_t len;
  /**
   * Pointer to underlying buffer
   */
  const uint64_t *data;
} MortonKeys;

/**
 * Implicit map between input coordinate index and global index
 * after sorting during octree construction
 */
typedef struct GlobalIndices {
  /**
   * Number of global indices
   */
  uintptr_t len;
  /**
   * Pointer to underlying buffer
   */
  const void *data;
} GlobalIndices;

/**
 * Container for multiple Potentials. Used when FMM run over multiple
 * charge vectors.
 */
typedef struct Potentials {
  /**
   * Number of charge vectors associated with FMM call ()
   */
  uintptr_t n;
  /**
   * Pointer to underlying buffer
   */
  struct Potential *data;
  /**
   * Associated scalar type
   */
  enum ScalarType scalar;
} Potentials;

/**
 * Coordinates
 */
typedef struct Coordinates {
  /**
   * Length of coordinates buffer of length 3*n_coordinates
   */
  uintptr_t len;
  /**
   * Pointer to underlying buffer
   */
  const void *data;
  /**
   * Associated scalar type
   */
  enum ScalarType scalar;
} Coordinates;

/**
 * Free the FmmEvaluator object
 */
void free_fmm_evaluator(struct FmmEvaluator *fmm_p);

/**
 * Constructor for F32 Laplace FMM with BLAS based M2L translations compressed
 * with deterministic SVD.
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 */
struct FmmEvaluator *laplace_blas_svd_f32_alloc(const uintptr_t *expansion_order,
                                                uintptr_t nexpansion_order,
                                                const void *sources,
                                                uintptr_t nsources,
                                                const void *targets,
                                                uintptr_t ntargets,
                                                const void *charges,
                                                uintptr_t ncharges,
                                                bool prune_empty,
                                                uint64_t n_crit,
                                                uint64_t depth,
                                                float singular_value_threshold);

/**
 * Constructor for F64 Laplace FMM with BLAS based M2L translations compressed
 * with deterministic SVD.
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 */
struct FmmEvaluator *laplace_blas_svd_f64_alloc(const uintptr_t *expansion_order,
                                                uintptr_t nexpansion_order,
                                                const void *sources,
                                                uintptr_t nsources,
                                                const void *targets,
                                                uintptr_t ntargets,
                                                const void *charges,
                                                uintptr_t ncharges,
                                                bool prune_empty,
                                                uint64_t n_crit,
                                                uint64_t depth,
                                                double singular_value_threshold);

/**
 * Constructor for F32 Laplace FMM with BLAS based M2L translations compressed
 * with randomised SVD.
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 */
struct FmmEvaluator *laplace_blas_rsvd_f32_alloc(const uintptr_t *expansion_order,
                                                 uintptr_t nexpansion_order,
                                                 const void *sources,
                                                 uintptr_t nsources,
                                                 const void *targets,
                                                 uintptr_t ntargets,
                                                 const void *charges,
                                                 uintptr_t ncharges,
                                                 bool prune_empty,
                                                 uint64_t n_crit,
                                                 uint64_t depth,
                                                 float singular_value_threshold,
                                                 uintptr_t n_components,
                                                 uintptr_t n_oversamples);

/**
 * Constructor for F64 Laplace FMM with BLAS based M2L translations compressed
 * with randomised SVD.
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 */
struct FmmEvaluator *laplace_blas_rsvd_f64_alloc(const uintptr_t *expansion_order,
                                                 uintptr_t nexpansion_order,
                                                 const void *sources,
                                                 uintptr_t nsources,
                                                 const void *targets,
                                                 uintptr_t ntargets,
                                                 const void *charges,
                                                 uintptr_t ncharges,
                                                 bool prune_empty,
                                                 uint64_t n_crit,
                                                 uint64_t depth,
                                                 double singular_value_threshold,
                                                 uintptr_t n_components,
                                                 uintptr_t n_oversamples);

/**
 * Constructor for F32 Laplace FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 */
struct FmmEvaluator *laplace_fft_f32_alloc(const uintptr_t *expansion_order,
                                           uintptr_t nexpansion_order,
                                           const void *sources,
                                           uintptr_t nsources,
                                           const void *targets,
                                           uintptr_t ntargets,
                                           const void *charges,
                                           uintptr_t ncharges,
                                           bool prune_empty,
                                           uint64_t n_crit,
                                           uint64_t depth,
                                           uintptr_t block_size);

/**
 * Constructor for F64 Laplace FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 */
struct FmmEvaluator *laplace_fft_f64_alloc(const uintptr_t *expansion_order,
                                           uintptr_t nexpansion_order,
                                           const void *sources,
                                           uintptr_t nsources,
                                           const void *targets,
                                           uintptr_t ntargets,
                                           const void *charges,
                                           uintptr_t ncharges,
                                           bool prune_empty,
                                           uint64_t n_crit,
                                           uint64_t depth,
                                           uintptr_t block_size);

/**
 * Constructor for F32 Helmholtz FMM with BLAS based M2L translations compressed
 * with deterministic SVD.
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 */
struct FmmEvaluator *helmholtz_blas_svd_f32_alloc(const uintptr_t *expansion_order,
                                                  uintptr_t nexpansion_order,
                                                  float wavenumber,
                                                  const void *sources,
                                                  uintptr_t nsources,
                                                  const void *targets,
                                                  uintptr_t ntargets,
                                                  const void *charges,
                                                  uintptr_t ncharges,
                                                  bool prune_empty,
                                                  uint64_t n_crit,
                                                  uint64_t depth,
                                                  float singular_value_threshold);

/**
 * Constructor for F64 Helmholtz FMM with BLAS based M2L translations compressed
 * with deterministic SVD.
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 */
struct FmmEvaluator *helmholtz_blas_svd_f64_alloc(const uintptr_t *expansion_order,
                                                  uintptr_t nexpansion_order,
                                                  double wavenumber,
                                                  const void *sources,
                                                  uintptr_t nsources,
                                                  const void *targets,
                                                  uintptr_t ntargets,
                                                  const void *charges,
                                                  uintptr_t ncharges,
                                                  bool prune_empty,
                                                  uint64_t n_crit,
                                                  uint64_t depth,
                                                  double singular_value_threshold);

/**
 * Constructor for F32 Helmholtz FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 */
struct FmmEvaluator *helmholtz_fft_f32_alloc(const uintptr_t *expansion_order,
                                             uintptr_t nexpansion_order,
                                             float wavenumber,
                                             const void *sources,
                                             uintptr_t nsources,
                                             const void *targets,
                                             uintptr_t ntargets,
                                             const void *charges,
                                             uintptr_t ncharges,
                                             bool prune_empty,
                                             uint64_t n_crit,
                                             uint64_t depth,
                                             uintptr_t block_size);

/**
 * Constructor for F64 Helmholtz FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 *
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `nexpansion_order`: The number of expansion orders.
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `nsources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `ntargets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `ncharges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 * reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 */
struct FmmEvaluator *helmholtz_fft_f64_alloc(const uintptr_t *expansion_order,
                                             uintptr_t nexpansion_order,
                                             double wavenumber,
                                             const void *sources,
                                             uintptr_t nsources,
                                             const void *targets,
                                             uintptr_t ntargets,
                                             const void *charges,
                                             uintptr_t ncharges,
                                             bool prune_empty,
                                             uint64_t n_crit,
                                             uint64_t depth,
                                             uintptr_t block_size);

/**
 * Evaluate
 */
void evaluate(struct FmmEvaluator *fmm, bool timed);

/**
 * Clear charges, and attach new charges
 */
void clear(struct FmmEvaluator *fmm, const void *charges, uintptr_t ncharges);

/**
 * Query for all potentials
 */
struct Potential *all_potentials(struct FmmEvaluator *fmm);

/**
 * Free Morton keys
 */
void free_morton_keys(struct MortonKeys *keys_p);

/**
 * Query for global indices of target points
 */
struct GlobalIndices *global_indices_target_tree(struct FmmEvaluator *fmm);

struct Potentials *leaf_potentials(struct FmmEvaluator *fmm, uint64_t leaf);

/**
 * Query source tree for coordinates contained in a leaf box
 */
struct Coordinates *coordinates_source_tree(struct FmmEvaluator *fmm, uint64_t leaf);

struct MortonKeys *leaves_target_tree(struct FmmEvaluator *fmm);
