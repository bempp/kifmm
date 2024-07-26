#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

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
 * potentials
 */
typedef struct Potential {
  uintptr_t len;
  const void *data;
  enum ScalarType scalar;
} Potential;

/**
 * potentials
 */
typedef struct Potentials {
  uintptr_t n;
  const struct Potential *data;
  enum ScalarType scalar;
} Potentials;

/**
 * Global indices
 */
typedef struct GlobalIndices {
  uintptr_t len;
  const void *data;
} GlobalIndices;

/**
 * Potentials
 */
typedef struct Coordinates {
  uintptr_t len;
  const void *data;
  enum ScalarType scalar;
} Coordinates;

/**
 * Morton keys
 */
typedef struct MortonKeys {
  uintptr_t len;
  const uint64_t *data;
} MortonKeys;

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
struct Potential *all_potentials(struct FmmEvaluator *fmm);

/**
 * Free potential
 */
void free_potential(struct Potential *potential_p);

/**
 * Free potentials
 */
void free_potentials(struct Potentials *potentials_p);

void free_global_indices(struct GlobalIndices *global_indices_p);

/**
 * Query for global indices of target points
 */
struct GlobalIndices *global_indices_target_tree(struct FmmEvaluator *fmm);

/**
 * Query source tree for coordinates contained in a leaf box
 */
struct Coordinates *coordinates_source_tree(struct FmmEvaluator *fmm, uint64_t leaf);

struct MortonKeys *leaves_target_tree(struct FmmEvaluator *fmm);
