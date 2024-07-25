#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Maximum block size to use to process leaf boxes during P2M kernel.
 */
#define P2M_MAX_BLOCK_SIZE 32

/**
 * Maximum block size to use to process boxes by level during M2M kernel.
 */
#define M2M_MAX_BLOCK_SIZE 16

/**
 * Maximum block size to use to process boxes by level during L2L kernel.
 */
#define L2L_MAX_BLOCK_SIZE 16

/**
 * Default value chosen for maximum number of particles per leaf.
 */
#define DEFAULT_NCRIT 150

/**
 * Default maximum block size to use to process multiple child clusters during FFT M2L
 */
#define DEFAULT_M2L_FFT_BLOCK_SIZE 128

/**
 * Maximum possible level of octree recursion, by definition.
 */
#define DEEPEST_LEVEL 16

/**
 * The 'size' of each level in terms of octants along each axis, at the maximum depth of recursion.
 */
#define LEVEL_SIZE 65536

/**
 * Number of bits used for Level information.
 */
#define LEVEL_DISPLACEMENT 15

/**
 * Mask for the last 15 bits.
 */
#define LEVEL_MASK 32767

/**
 * Mask for lowest order byte.
 */
#define BYTE_MASK 255

/**
 * Number of bits in a byte.
 */
#define BYTE_DISPLACEMENT 8

/**
 * Mask encapsulating a bit.
 */
#define NINE_BIT_MASK 511

/**
 * Number of siblings for each node in octree
 */
#define NSIBLINGS 8

/**
 * Number of siblings squared for each node in octree.
 */
#define NSIBLINGS_SQUARED 64

/**
 * Number of corners for each box.
 */
#define NCORNERS 8

/**
 * Number of unique transfer vectors for homogenous, scale invariant, kernels.
 */
#define NTRANSFER_VECTORS_KIFMM 316

/**
 * Maximum number of boxes in a 1 box deep halo around a given box in 3D.
 */
#define NHALO 26

/**
 * Fmm Type
 */
typedef enum FmmCType {
  FmmCType_Laplace32,
  FmmCType_Laplace64,
  FmmCType_Helmholtz32,
  FmmCType_Helmholtz64,
} FmmCType;

/**
 * Translation type
 */
typedef enum FmmTranslationCType {
  FmmTranslationCType_Blas,
  FmmTranslationCType_Fft,
} FmmTranslationCType;

/**
 * Scalar type
 */
typedef enum ScalarType {
  ScalarType_F32,
  ScalarType_F64,
  ScalarType_C32,
  ScalarType_C64,
} ScalarType;

/**
 * Runtime FMM object
 */
typedef struct FmmEvaluator {
  enum FmmCType ctype;
  enum FmmTranslationCType ctranslation_type;
  void *data;
} FmmEvaluator;

/**
 * Potentials
 */
typedef struct Potentials {
  uintptr_t len;
  const void *data;
  enum ScalarType scalar;
} Potentials;

/**
 * Global indices
 */
typedef struct GlobalIndices {
  uintptr_t len;
  const void *data;
} GlobalIndices;

void free_fmm_evaluator(struct FmmEvaluator *fmm_p);

/**
 * Constructor
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
 * Constructor
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
 * Constructor
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
 * Constructor
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
 * Constructor
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
 * Constructor
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
 * Constructor
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
 * Constructor
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
 * Constructor
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
 * Constructor
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
 * Evaluate FMM
 */
void evaluate(struct FmmEvaluator *fmm, bool timed);

/**
 * Clear charges, and attach new charges
 */
void clear(struct FmmEvaluator *fmm, const void *charges, uintptr_t ncharges);

/**
 * Query for all potentials
 */
struct Potentials potentials(struct FmmEvaluator *fmm);

/**
 * Query for global indices of target points
 */
struct GlobalIndices global_indices_target_tree(struct FmmEvaluator *fmm);
