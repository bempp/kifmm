#include <memory>

// Constructor implementation
template <typename T>
KiFmm<T>::KiFmm(const std::vector<T> &source, const std::vector<T> &target,
                const std::vector<T> &charges,
                const FieldTranslation<T> &fieldTranslation)
    : sourceCoordinates(source), targetCoordinates(target),
      sourceCharges(charges), fieldTranslation(fieldTranslation) {

  T *sources_ptr = sourceCoordinates.data();
  size_t num_sources = sourceCoordinates.size();
  size_t num_targets = targetCoordinates.size();
  size_t num_charges = charges.size();

  T *targets_ptr = targetCoordinates.data();
  T *charges_ptr = sourceCharges.data();

  if constexpr (std::is_same_v<T, float>) {
    LaplaceBlas32 *fmm =
        laplace_blas_f32(sources_ptr, num_sources, targets_ptr, num_targets,
                         charges_ptr, num_charges);
    fmmInstance.set(fmm);
  } else if constexpr (std::is_same_v<T, double>) {
    LaplaceBlas64 *fmm =
        laplace_blas_f64(sources_ptr, num_sources, targets_ptr, num_targets,
                         charges_ptr, num_charges);
    fmmInstance.set(fmm);
  }
}

template <typename T> void KiFmm<T>::evaluate(bool timed) {

  if constexpr (std::is_same_v<T, float>) {
    LaplaceBlas32 *fmm = static_cast<LaplaceBlas32 *>(this->fmmInstance.get());
    evaluate_laplace_blas_f32(fmm, timed);
    std::cout << "Running FMM single precision " << std::endl;
  } else if constexpr (std::is_same_v<T, double>) {
    LaplaceBlas64 *fmm = static_cast<LaplaceBlas64 *>(this->fmmInstance.get());
    evaluate_laplace_blas_f64(fmm, timed);
    std::cout << "Running FMM double precision" << std::endl;
  }
}

FmmPointer::FmmPointer() : ptr(nullptr), type(Type::None) {}

FmmPointer::~FmmPointer() { clear(); }

void FmmPointer::set(LaplaceBlas32 *ptr) {
  clear();
  this->ptr = ptr;
  type = Type::Blas32;
}

void FmmPointer::set(LaplaceBlas64 *ptr) {
  clear();
  this->ptr = ptr;
  type = Type::Blas64;
}

void *FmmPointer::get() const { return ptr; }

FmmPointer::Type FmmPointer::getType() const { return type; }

void FmmPointer::clear() {
  if (type == Type::Blas32) {
    // destroyLaplaceBlas32(static_cast<LaplaceBlas32*>(ptr));
  } else if (type == Type::Blas64) {
    // destroyLaplaceBlas64(static_cast<LaplaceBlas64*>(ptr));
  }
  ptr = nullptr;
  type = Type::None;
}

// Constructor for FieldTranslation
template <typename T>
FieldTranslation<T>::FieldTranslation(FieldTranslationType type,
                                      size_t target_rank,
                                      T singular_value_threshold)
    : type(type) {
  if (type == FieldTranslationType::Blas) {
    new (&data.blas) BlasFieldTranslation<T>(
        target_rank,
        singular_value_threshold); // Placement new with initializer list
  } else {
    throw std::invalid_argument("Invalid type for Blas constructor");
  }
}

template <typename T>
FieldTranslation<T>::FieldTranslation(FieldTranslationType type,
                                      size_t block_size)
    : type(type) {
  if (type == FieldTranslationType::Fft) {
    new (&data.fft)
        FftFieldTranslation(block_size); // Placement new with initializer list
  } else {
    throw std::invalid_argument("Invalid type for Fft constructor");
  }
}

// Destructor
template <typename T> FieldTranslation<T>::~FieldTranslation() {
  switch (type) {
  case FieldTranslationType::Blas:
    data.blas.~BlasFieldTranslation<T>();
    break;
  case FieldTranslationType::Fft:
    data.fft.~FftFieldTranslation();
    break;
  }
}