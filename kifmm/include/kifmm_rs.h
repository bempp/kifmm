#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Number of bytes used to encode vector lengths in serialisation
 */
#define LEN_BYTES 8

/**
 * Enumeration of communication types for timing
 */
typedef enum CommunicationType {
  /**
   * Tree construction
   */
  CommunicationType_SourceTree,
  /**
   * Tree construction
   */
  CommunicationType_TargetTree,
  /**
   * Domain exchange
   */
  CommunicationType_TargetDomain,
  /**
   * Domain exchange
   */
  CommunicationType_SourceDomain,
  /**
   * Layout
   */
  CommunicationType_Layout,
  /**
   * V list ghost exchange
   */
  CommunicationType_GhostExchangeV,
  /**
   * V list ghost exchange at runtime
   */
  CommunicationType_GhostExchangeVRuntime,
  /**
   * U list ghost exchange
   */
  CommunicationType_GhostExchangeU,
  /**
   * Gather global FMM
   */
  CommunicationType_GatherGlobalFmm,
  /**
   * Scatter global FMM
   */
  CommunicationType_ScatterGlobalFmm,
} CommunicationType;

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
 * Enumeration of metadata construction for timing
 */
typedef enum MetadataType {
  /**
   * Field translation data
   */
  MetadataType_SourceToTargetData,
  /**
   * Source tree translations
   */
  MetadataType_SourceData,
  /**
   * Target tree translations
   */
  MetadataType_TargetData,
  /**
   * Global FMM
   */
  MetadataType_GlobalFmm,
  /**
   * Ghost FMM V
   */
  MetadataType_GhostFmmV,
  /**
   * Ghost FMM U
   */
  MetadataType_GhostFmmU,
  /**
   * Pointer and Buffer Creationmp
   */
  MetadataType_MetadataCreation,
  /**
   * Displacement Map Creation
   */
  MetadataType_DisplacementMap,
} MetadataType;

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
 * Runtime FMM type constructed from C
 */
typedef struct FmmEvaluatorMPI {
  enum FmmCType ctype;
  enum FmmTranslationCType ctranslation_type;
  void *data;
} FmmEvaluatorMPI;

/**
 * C compatible struct for timing
 */
typedef struct OperatorTime {
  /**
   * Time in milliseconds
   */
  uint64_t time;
} OperatorTime;

typedef struct CommunicationEntry {
  enum CommunicationType comm_type;
  struct OperatorTime time;
} CommunicationEntry;

typedef struct CommunicationTimes {
  struct CommunicationEntry *times;
  uintptr_t length;
} CommunicationTimes;

typedef struct MetadataEntry {
  enum MetadataType metadata_type;
  struct OperatorTime time;
} MetadataEntry;

typedef struct MetadataTimes {
  struct MetadataEntry *times;
  uintptr_t length;
} MetadataTimes;

/**
 * Enumeration of operator types for timing
 */
typedef enum FmmOperatorType_Tag {
  /**
   * particle to multipole
   */
  FmmOperatorType_P2M,
  /**
   * multipole to multipole (level)
   */
  FmmOperatorType_M2M,
  /**
   * multipole to local (level)
   */
  FmmOperatorType_M2L,
  /**
   * local to local (level)
   */
  FmmOperatorType_L2L,
  /**
   * local to particle
   */
  FmmOperatorType_L2P,
  /**
   * particle to particle
   */
  FmmOperatorType_P2P,
} FmmOperatorType_Tag;

typedef struct FmmOperatorType {
  FmmOperatorType_Tag tag;
  union {
    struct {
      uint64_t m2m;
    };
    struct {
      uint64_t m2l;
    };
    struct {
      uint64_t l2l;
    };
  };
} FmmOperatorType;

typedef struct FmmOperatorEntry {
  struct FmmOperatorType op_type;
  struct OperatorTime time;
} FmmOperatorEntry;

typedef struct FmmOperatorTimes {
  struct FmmOperatorEntry *times;
  uintptr_t length;
} FmmOperatorTimes;

/**
 * Potential data
 */
typedef struct Potential {
  /**
   * Length of underlying buffer, of length n_eval_mode*n_coordinates*n_evals
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
 * Expansion data
 */
typedef struct Expansion {
  /**
   * Length of underlying buffer, of length n_eval_mode*n_evals*n_coeffs
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
} Expansion;

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
 * Free the FmmEvaluator object
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - `fmm_p` is a valid pointer to a properly initialized `FmmEvaluator` instance.
 * - The `fmm_p` pointer remains valid for the duration of the function call.
 */
void free_fmm_evaluator(struct FmmEvaluator *fmm_p);

/**
 * Free the FmmEvaluatorMPI object
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - `fmm_p` is a valid pointer to a properly initialized `FmmEvaluator` instance.
 * - The `fmm_p` pointer remains valid for the duration of the function call.
 */
void free_fmm_evaluator_mpi(struct FmmEvaluatorMPI *fmm_p);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *laplace_blas_svd_f32_alloc(bool timed,
                                                const uintptr_t *expansion_order,
                                                uintptr_t n_expansion_order,
                                                bool eval_type,
                                                const void *sources,
                                                uintptr_t n_sources,
                                                const void *targets,
                                                uintptr_t n_targets,
                                                const void *charges,
                                                uintptr_t n_charges,
                                                bool prune_empty,
                                                uint64_t n_crit,
                                                uint64_t depth,
                                                float singular_value_threshold,
                                                uintptr_t surface_diff);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *laplace_blas_svd_f64_alloc(bool timed,
                                                const uintptr_t *expansion_order,
                                                uintptr_t n_expansion_order,
                                                bool eval_type,
                                                const void *sources,
                                                uintptr_t n_sources,
                                                const void *targets,
                                                uintptr_t n_targets,
                                                const void *charges,
                                                uintptr_t n_charges,
                                                bool prune_empty,
                                                uint64_t n_crit,
                                                uint64_t depth,
                                                double singular_value_threshold,
                                                uintptr_t surface_diff);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *laplace_blas_rsvd_f32_alloc(bool timed,
                                                 const uintptr_t *expansion_order,
                                                 uintptr_t n_expansion_order,
                                                 bool eval_type,
                                                 const void *sources,
                                                 uintptr_t n_sources,
                                                 const void *targets,
                                                 uintptr_t n_targets,
                                                 const void *charges,
                                                 uintptr_t n_charges,
                                                 bool prune_empty,
                                                 uint64_t n_crit,
                                                 uint64_t depth,
                                                 float singular_value_threshold,
                                                 uintptr_t surface_diff,
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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *laplace_blas_rsvd_f64_alloc(bool timed,
                                                 const uintptr_t *expansion_order,
                                                 uintptr_t n_expansion_order,
                                                 bool eval_type,
                                                 const void *sources,
                                                 uintptr_t n_sources,
                                                 const void *targets,
                                                 uintptr_t n_targets,
                                                 const void *charges,
                                                 uintptr_t n_charges,
                                                 bool prune_empty,
                                                 uint64_t n_crit,
                                                 uint64_t depth,
                                                 double singular_value_threshold,
                                                 uintptr_t surface_diff,
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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *laplace_fft_f32_alloc(bool timed,
                                           const uintptr_t *expansion_order,
                                           uintptr_t n_expansion_order,
                                           bool eval_type,
                                           const void *sources,
                                           uintptr_t n_sources,
                                           const void *targets,
                                           uintptr_t n_targets,
                                           const void *charges,
                                           uintptr_t n_charges,
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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *laplace_fft_f64_alloc(bool timed,
                                           const uintptr_t *expansion_order,
                                           uintptr_t n_expansion_order,
                                           bool eval_type,
                                           const void *sources,
                                           uintptr_t n_sources,
                                           const void *targets,
                                           uintptr_t n_targets,
                                           const void *charges,
                                           uintptr_t n_charges,
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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *helmholtz_blas_rsvd_f32_alloc(bool timed,
                                                   const uintptr_t *expansion_order,
                                                   uintptr_t n_expansion_order,
                                                   bool eval_type,
                                                   float wavenumber,
                                                   const void *sources,
                                                   uintptr_t n_sources,
                                                   const void *targets,
                                                   uintptr_t n_targets,
                                                   const void *charges,
                                                   uintptr_t n_charges,
                                                   bool prune_empty,
                                                   uint64_t n_crit,
                                                   uint64_t depth,
                                                   float singular_value_threshold,
                                                   uintptr_t surface_diff,
                                                   uintptr_t n_components,
                                                   uintptr_t n_oversamples);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *helmholtz_blas_rsvd_f64_alloc(bool timed,
                                                   const uintptr_t *expansion_order,
                                                   uintptr_t n_expansion_order,
                                                   bool eval_type,
                                                   double wavenumber,
                                                   const void *sources,
                                                   uintptr_t n_sources,
                                                   const void *targets,
                                                   uintptr_t n_targets,
                                                   const void *charges,
                                                   uintptr_t n_charges,
                                                   bool prune_empty,
                                                   uint64_t n_crit,
                                                   uint64_t depth,
                                                   double singular_value_threshold,
                                                   uintptr_t surface_diff,
                                                   uintptr_t n_components,
                                                   uintptr_t n_oversamples);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *helmholtz_blas_svd_f32_alloc(bool timed,
                                                  const uintptr_t *expansion_order,
                                                  uintptr_t n_expansion_order,
                                                  bool eval_type,
                                                  float wavenumber,
                                                  const void *sources,
                                                  uintptr_t n_sources,
                                                  const void *targets,
                                                  uintptr_t n_targets,
                                                  const void *charges,
                                                  uintptr_t n_charges,
                                                  bool prune_empty,
                                                  uint64_t n_crit,
                                                  uint64_t depth,
                                                  float singular_value_threshold,
                                                  uintptr_t surface_diff);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *helmholtz_blas_svd_f64_alloc(bool timed,
                                                  const uintptr_t *expansion_order,
                                                  uintptr_t n_expansion_order,
                                                  bool eval_type,
                                                  double wavenumber,
                                                  const void *sources,
                                                  uintptr_t n_sources,
                                                  const void *targets,
                                                  uintptr_t n_targets,
                                                  const void *charges,
                                                  uintptr_t n_charges,
                                                  bool prune_empty,
                                                  uint64_t n_crit,
                                                  uint64_t depth,
                                                  double singular_value_threshold,
                                                  uintptr_t surface_diff);

/**
 * Constructor for F32 Helmholtz FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *helmholtz_fft_f32_alloc(bool timed,
                                             const uintptr_t *expansion_order,
                                             uintptr_t n_expansion_order,
                                             bool eval_type,
                                             float wavenumber,
                                             const void *sources,
                                             uintptr_t n_sources,
                                             const void *targets,
                                             uintptr_t n_targets,
                                             const void *charges,
                                             uintptr_t n_charges,
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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluator *helmholtz_fft_f64_alloc(bool timed,
                                             const uintptr_t *expansion_order,
                                             uintptr_t n_expansion_order,
                                             bool eval_type,
                                             double wavenumber,
                                             const void *sources,
                                             uintptr_t n_sources,
                                             const void *targets,
                                             uintptr_t n_targets,
                                             const void *charges,
                                             uintptr_t n_charges,
                                             bool prune_empty,
                                             uint64_t n_crit,
                                             uint64_t depth,
                                             uintptr_t block_size);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *laplace_blas_svd_f32_mpi_alloc(bool timed,
                                                       const uintptr_t *expansion_order,
                                                       uintptr_t n_expansion_order,
                                                       bool eval_type,
                                                       const void *sources,
                                                       uintptr_t n_sources,
                                                       const void *targets,
                                                       uintptr_t n_targets,
                                                       const void *charges,
                                                       uintptr_t n_charges,
                                                       bool prune_empty,
                                                       uint64_t local_depth,
                                                       uint64_t global_depth,
                                                       float singular_value_threshold,
                                                       uintptr_t surface_diff,
                                                       uint64_t sort_kind,
                                                       uintptr_t n_samples,
                                                       void *communicator);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *laplace_blas_svd_f64_mpi_alloc(bool timed,
                                                       const uintptr_t *expansion_order,
                                                       uintptr_t n_expansion_order,
                                                       bool eval_type,
                                                       const void *sources,
                                                       uintptr_t n_sources,
                                                       const void *targets,
                                                       uintptr_t n_targets,
                                                       const void *charges,
                                                       uintptr_t n_charges,
                                                       bool prune_empty,
                                                       uint64_t local_depth,
                                                       uint64_t global_depth,
                                                       double singular_value_threshold,
                                                       uintptr_t surface_diff,
                                                       uint64_t sort_kind,
                                                       uintptr_t n_samples,
                                                       void *communicator);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *laplace_blas_rsvd_f32_mpi_alloc(bool timed,
                                                        const uintptr_t *expansion_order,
                                                        uintptr_t n_expansion_order,
                                                        bool eval_type,
                                                        const void *sources,
                                                        uintptr_t n_sources,
                                                        const void *targets,
                                                        uintptr_t n_targets,
                                                        const void *charges,
                                                        uintptr_t n_charges,
                                                        bool prune_empty,
                                                        uint64_t local_depth,
                                                        uint64_t global_depth,
                                                        float singular_value_threshold,
                                                        uintptr_t surface_diff,
                                                        uintptr_t n_components,
                                                        uintptr_t n_oversamples,
                                                        uint64_t sort_kind,
                                                        uintptr_t n_samples,
                                                        void *communicator);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *laplace_blas_rsvd_f64_mpi_alloc(bool timed,
                                                        const uintptr_t *expansion_order,
                                                        uintptr_t n_expansion_order,
                                                        bool eval_type,
                                                        const void *sources,
                                                        uintptr_t n_sources,
                                                        const void *targets,
                                                        uintptr_t n_targets,
                                                        const void *charges,
                                                        uintptr_t n_charges,
                                                        bool prune_empty,
                                                        uint64_t local_depth,
                                                        uint64_t global_depth,
                                                        double singular_value_threshold,
                                                        uintptr_t surface_diff,
                                                        uintptr_t n_components,
                                                        uintptr_t n_oversamples,
                                                        uint64_t sort_kind,
                                                        uintptr_t n_samples,
                                                        void *communicator);

/**
 * Constructor for F32 Laplace FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *laplace_fft_f32_mpi_alloc(bool timed,
                                                  const uintptr_t *expansion_order,
                                                  uintptr_t n_expansion_order,
                                                  bool eval_type,
                                                  const void *sources,
                                                  uintptr_t n_sources,
                                                  const void *targets,
                                                  uintptr_t n_targets,
                                                  const void *charges,
                                                  uintptr_t n_charges,
                                                  bool prune_empty,
                                                  uint64_t local_depth,
                                                  uint64_t global_depth,
                                                  uintptr_t block_size,
                                                  uint64_t sort_kind,
                                                  uintptr_t n_samples,
                                                  void *communicator);

/**
 * Constructor for F64 Laplace FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *laplace_fft_f64_mpi_alloc(bool timed,
                                                  const uintptr_t *expansion_order,
                                                  uintptr_t n_expansion_order,
                                                  bool eval_type,
                                                  const void *sources,
                                                  uintptr_t n_sources,
                                                  const void *targets,
                                                  uintptr_t n_targets,
                                                  const void *charges,
                                                  uintptr_t n_charges,
                                                  bool prune_empty,
                                                  uint64_t local_depth,
                                                  uint64_t global_depth,
                                                  uintptr_t block_size,
                                                  uint64_t sort_kind,
                                                  uintptr_t n_samples,
                                                  void *communicator);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *helmholtz_blas_rsvd_f32_mpi_alloc(bool timed,
                                                          const uintptr_t *expansion_order,
                                                          uintptr_t n_expansion_order,
                                                          bool eval_type,
                                                          float wavenumber,
                                                          const void *sources,
                                                          uintptr_t n_sources,
                                                          const void *targets,
                                                          uintptr_t n_targets,
                                                          const void *charges,
                                                          uintptr_t n_charges,
                                                          bool prune_empty,
                                                          uint64_t local_depth,
                                                          uint64_t global_depth,
                                                          float singular_value_threshold,
                                                          uintptr_t surface_diff,
                                                          uintptr_t n_components,
                                                          uintptr_t n_oversamples,
                                                          uint64_t sort_kind,
                                                          uintptr_t n_samples,
                                                          void *communicator);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 * - `n_components`: If known, can specify the rank of the M2L matrix for randomised range finding, otherwise set to 0.
 * - `n_oversamples`: Optionally choose the number of oversamples for randomised range finding, otherwise set to 10.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *helmholtz_blas_rsvd_f64_mpi_alloc(bool timed,
                                                          const uintptr_t *expansion_order,
                                                          uintptr_t n_expansion_order,
                                                          bool eval_type,
                                                          double wavenumber,
                                                          const void *sources,
                                                          uintptr_t n_sources,
                                                          const void *targets,
                                                          uintptr_t n_targets,
                                                          const void *charges,
                                                          uintptr_t n_charges,
                                                          bool prune_empty,
                                                          uint64_t local_depth,
                                                          uint64_t global_depth,
                                                          double singular_value_threshold,
                                                          uintptr_t surface_diff,
                                                          uintptr_t n_components,
                                                          uintptr_t n_oversamples,
                                                          uint64_t sort_kind,
                                                          uintptr_t n_samples,
                                                          void *communicator);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *helmholtz_blas_svd_f32_mpi_alloc(bool timed,
                                                         const uintptr_t *expansion_order,
                                                         uintptr_t n_expansion_order,
                                                         bool eval_type,
                                                         float wavenumber,
                                                         const void *sources,
                                                         uintptr_t n_sources,
                                                         const void *targets,
                                                         uintptr_t n_targets,
                                                         const void *charges,
                                                         uintptr_t n_charges,
                                                         bool prune_empty,
                                                         uint64_t local_depth,
                                                         uint64_t global_depth,
                                                         float singular_value_threshold,
                                                         uintptr_t surface_diff,
                                                         uint64_t sort_kind,
                                                         uintptr_t n_samples,
                                                         void *communicator);

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
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `singular_value_threshold`: Threshold for singular values used in compressing M2L matrices.
 * - `surface_diff`: Set to 0 to disable, otherwise uses surface_diff+equivalent_surface_expansion_order = check_surface_expansion_order
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *helmholtz_blas_svd_f64_mpi_alloc(bool timed,
                                                         const uintptr_t *expansion_order,
                                                         uintptr_t n_expansion_order,
                                                         bool eval_type,
                                                         double wavenumber,
                                                         const void *sources,
                                                         uintptr_t n_sources,
                                                         const void *targets,
                                                         uintptr_t n_targets,
                                                         const void *charges,
                                                         uintptr_t n_charges,
                                                         bool prune_empty,
                                                         uint64_t local_depth,
                                                         uint64_t global_depth,
                                                         double singular_value_threshold,
                                                         uintptr_t surface_diff,
                                                         uint64_t sort_kind,
                                                         uintptr_t n_samples,
                                                         void *communicator);

/**
 * Constructor for F32 Helmholtz FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *helmholtz_fft_f32_mpi_alloc(bool timed,
                                                    const uintptr_t *expansion_order,
                                                    uintptr_t n_expansion_order,
                                                    bool eval_type,
                                                    float wavenumber,
                                                    const void *sources,
                                                    uintptr_t n_sources,
                                                    const void *targets,
                                                    uintptr_t n_targets,
                                                    const void *charges,
                                                    uintptr_t n_charges,
                                                    bool prune_empty,
                                                    uint64_t local_depth,
                                                    uint64_t global_depth,
                                                    uintptr_t block_size,
                                                    uint64_t sort_kind,
                                                    uintptr_t n_samples,
                                                    void *communicator);

/**
 * Constructor for F64 Helmholtz FMM with FFT based M2L translations
 *
 * Note that either `n_crit` or `depth` must be specified. If `n_crit` is specified, depth
 * must be set to 0 and vice versa. If `depth` is specified, `expansion_order` must be specified
 * at each level, and stored as a buffer of length `depth` + 1.
 *
 *
 * # Parameters
 * - `timed`: Modulates whether operators and metadata are timed.
 * - `expansion_order`: A pointer to an array of expansion orders.
 * - `n_expansion_order`: The number of expansion orders.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `wavenumber`: The wavenumber.
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `prune_empty`: A boolean flag indicating whether to prune empty leaf nodes, and their ancestors.
 * - `n_crit`: Threshold for tree refinement, if set to 0 ignored. Otherwise will refine until threshold is
 *  reached based on a uniform particle distribution.
 * - `depth`: The maximum depth of the tree, max supported depth is 16.
 * - `block_size`: Parameter size controls cache utilisation in field translation, set to 0 to use default.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmEvaluatorMPI *helmholtz_fft_f64_mpi_alloc(bool timed,
                                                    const uintptr_t *expansion_order,
                                                    uintptr_t n_expansion_order,
                                                    bool eval_type,
                                                    double wavenumber,
                                                    const void *sources,
                                                    uintptr_t n_sources,
                                                    const void *targets,
                                                    uintptr_t n_targets,
                                                    const void *charges,
                                                    uintptr_t n_charges,
                                                    bool prune_empty,
                                                    uint64_t local_depth,
                                                    uint64_t global_depth,
                                                    uintptr_t block_size,
                                                    uint64_t sort_kind,
                                                    uintptr_t n_samples,
                                                    void *communicator);

/**
 * Get the communication runtimes
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct CommunicationTimes *communication_times_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Get the metadata runtimes
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MetadataTimes *metadata_times_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Get the operator runtimes
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmOperatorTimes *operator_times_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Evaluate the Fast Multipole Method (FMM).
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `timed`: Boolean flag to time each operator.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmOperatorTimes *evaluate_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Clear charges and attach new charges.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void clear_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Attach new charges, in final Morton ordering
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `charges`: A pointer to the new charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void attach_charges_ordered_mpi(struct FmmEvaluatorMPI *fmm,
                                const void *charges,
                                uintptr_t n_charges);

/**
 * Attach new charges, in initial input ordering before global Morton sort
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `charges`: A pointer to the new charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void attach_charges_unordered_mpi(struct FmmEvaluatorMPI *fmm,
                                  const void *charges,
                                  uintptr_t n_charges);

/**
 * Query for all evaluated potentials, returned in order of global index.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Potential *all_potentials_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query for global indices of target points, where each index position corresponds to input
 * coordinate data index, and the elements correspond to the index as stored in the target tree
 * and therefore in the evaluated potentials.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct GlobalIndices *global_indices_target_tree_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query for locals at a specific key.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `key`: The identifier of a node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Expansion *local_mpi(struct FmmEvaluatorMPI *fmm, uint64_t key);

/**
 * Query for multipoles at a specific key.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `key`: The identifier of a node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Expansion *multipole_mpi(struct FmmEvaluatorMPI *fmm, uint64_t key);

/**
 * Query for potentials at a specific leaf.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of a leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Potentials *leaf_potentials_mpi(struct FmmEvaluatorMPI *fmm, uint64_t leaf);

/**
 * Construct a surface for a given key
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Coordinates *surface_mpi(struct FmmEvaluatorMPI *fmm,
                                double alpha,
                                uint64_t expansion_order,
                                uint64_t key);

/**
 * Query target tree for local depth
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
uint64_t target_tree_local_depth_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query target tree for global depth
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
uint64_t target_tree_global_depth_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query source tree for local depth
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
uint64_t source_tree_local_depth_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query target tree for global depth
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
uint64_t source_tree_global_depth_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query source tree for coordinates contained in a leaf box.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Coordinates *coordinates_source_tree_mpi(struct FmmEvaluatorMPI *fmm, uint64_t leaf);

/**
 * Query target tree for coordinates contained in a leaf box.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Coordinates *coordinates_target_tree_mpi(struct FmmEvaluatorMPI *fmm, uint64_t leaf);

/**
 * Query source tree for all keys
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MortonKeys *keys_source_tree_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query target tree for all keys
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MortonKeys *keys_target_tree_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query target tree for leaves.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MortonKeys *leaves_target_tree_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Query source tree for coordinates contained in a leaf box.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MortonKeys *leaves_source_tree_mpi(struct FmmEvaluatorMPI *fmm);

/**
 * Evaluate the kernel in single threaded mode
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `result`: A pointer to the results associated with the target points.
 * - `n_charges`: The length of the charges buffer.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void evaluate_kernel_st_mpi(struct FmmEvaluatorMPI *fmm,
                            bool eval_type,
                            const void *sources,
                            uintptr_t n_sources,
                            const void *targets,
                            uintptr_t n_targets,
                            const void *charges,
                            uintptr_t n_charges,
                            void *result,
                            uintptr_t nresult);

/**
 * Evaluate the kernel in multithreaded mode
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `result`: A pointer to the results associated with the target points.
 * - `n_charges`: The length of the charges buffer.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void evaluate_kernel_mt_mpi(struct FmmEvaluatorMPI *fmm,
                            bool eval_type,
                            const void *sources,
                            uintptr_t n_sources,
                            const void *targets,
                            uintptr_t n_targets,
                            const void *charges,
                            uintptr_t n_charges,
                            void *result,
                            uintptr_t nresult);

/**
 * Get the communication runtimes
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct CommunicationTimes *communication_times(struct FmmEvaluator *fmm);

/**
 * Get the metadata runtimes
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MetadataTimes *metadata_times(struct FmmEvaluator *fmm);

/**
 * Get the operator runtimes
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmOperatorTimes *operator_times(struct FmmEvaluator *fmm);

/**
 * Evaluate the Fast Multipole Method (FMM).
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `timed`: Boolean flag to time each operator.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct FmmOperatorTimes *evaluate(struct FmmEvaluator *fmm);

/**
 * Clear charges and attach new charges.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void clear(struct FmmEvaluator *fmm);

/**
 * Attach new charges, in final Morton ordering
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `charges`: A pointer to the new charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void attach_charges_ordered(struct FmmEvaluator *fmm, const void *charges, uintptr_t n_charges);

/**
 * Attach new charges, in initial input ordering before global Morton sort
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `charges`: A pointer to the new charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void attach_charges_unordered(struct FmmEvaluator *fmm, const void *charges, uintptr_t n_charges);

/**
 * Query for all evaluated potentials, returned in order of global index.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Potential *all_potentials(struct FmmEvaluator *fmm);

/**
 * Free Morton keys
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void free_morton_keys(struct MortonKeys *keys_p);

/**
 * Query for global indices of target points, where each index position corresponds to input
 * coordinate data index, and the elements correspond to the index as stored in the target tree
 * and therefore in the evaluated potentials.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct GlobalIndices *global_indices_target_tree(struct FmmEvaluator *fmm);

/**
 * Query for locals at a specific key.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `key`: The identifier of a node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Expansion *local(struct FmmEvaluator *fmm, uint64_t key);

/**
 * Query for multipoles at a specific key.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `key`: The identifier of a node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Expansion *multipole(struct FmmEvaluator *fmm, uint64_t key);

/**
 * Query for potentials at a specific leaf.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of a leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Potentials *leaf_potentials(struct FmmEvaluator *fmm, uint64_t leaf);

/**
 * Query key for level
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `key`: The identifier of the key.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
uint64_t level(uint64_t key);

/**
 * Construct a surface for a given key
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Coordinates *surface(struct FmmEvaluator *fmm,
                            double alpha,
                            uint64_t expansion_order,
                            uint64_t key);

/**
 * Query target tree for depth
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
uint64_t target_tree_depth(struct FmmEvaluator *fmm);

/**
 * Query source tree for depth
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
uint64_t source_tree_depth(struct FmmEvaluator *fmm);

/**
 * Query source tree for coordinates contained in a leaf box.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Coordinates *coordinates_source_tree(struct FmmEvaluator *fmm, uint64_t leaf);

/**
 * Query target tree for coordinates contained in a leaf box.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct Coordinates *coordinates_target_tree(struct FmmEvaluator *fmm, uint64_t leaf);

/**
 * Query source tree for all keys
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MortonKeys *keys_source_tree(struct FmmEvaluator *fmm);

/**
 * Query target tree for all keys
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MortonKeys *keys_target_tree(struct FmmEvaluator *fmm);

/**
 * Query target tree for leaves.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MortonKeys *leaves_target_tree(struct FmmEvaluator *fmm);

/**
 * Query source tree for coordinates contained in a leaf box.
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `leaf`: The identifier of the leaf node.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
struct MortonKeys *leaves_source_tree(struct FmmEvaluator *fmm);

/**
 * Evaluate the kernel in single threaded mode
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `result`: A pointer to the results associated with the target points.
 * - `n_charges`: The length of the charges buffer.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void evaluate_kernel_st(struct FmmEvaluator *fmm,
                        bool eval_type,
                        const void *sources,
                        uintptr_t n_sources,
                        const void *targets,
                        uintptr_t n_targets,
                        const void *charges,
                        uintptr_t n_charges,
                        void *result,
                        uintptr_t nresult);

/**
 * Evaluate the kernel in multithreaded mode
 *
 * # Parameters
 *
 * - `fmm`: Pointer to an `FmmEvaluator` instance.
 * - `eval_type`: true corresponds to evaluating potentials, false corresponds to evaluating potentials and potential derivatives
 * - `sources`: A pointer to the source points.
 * - `n_sources`: The length of the source points buffer
 * - `targets`: A pointer to the target points.
 * - `n_targets`: The length of the target points buffer.
 * - `charges`: A pointer to the charges associated with the source points.
 * - `n_charges`: The length of the charges buffer.
 * - `result`: A pointer to the results associated with the target points.
 * - `n_charges`: The length of the charges buffer.
 *
 * # Safety
 * This function is intended to be called from C. The caller must ensure that:
 * - Input data corresponds to valid pointers
 * - That they remain valid for the duration of the function call
 */
void evaluate_kernel_mt(struct FmmEvaluator *fmm,
                        bool eval_type,
                        const void *sources,
                        uintptr_t n_sources,
                        const void *targets,
                        uintptr_t n_targets,
                        const void *charges,
                        uintptr_t n_charges,
                        void *result,
                        uintptr_t nresult);
