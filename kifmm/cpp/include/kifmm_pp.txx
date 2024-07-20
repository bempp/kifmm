
// Constructor implementation
template <typename T>
KiFmm<T>::KiFmm(const std::vector<size_t> &expansionOrder,
                const std::vector<T> &sources, const std::vector<T> &targets,
                const std::vector<T> &charges,
                const FieldTranslation<T> &fieldTranslation, bool pruneEmpty,
                std::optional<uint64_t> nCrit, std::optional<uint64_t> depth)
    : expansionOrder(expansionOrder), sourceCoordinates(sources),
      targetCoordinates(targets), sourceCharges(charges),
      fieldTranslation(fieldTranslation), pruneEmpty(pruneEmpty),
      nCrit(nCrit.value_or(0)), depth(depth.value_or(0)) {

  T *sourcesPtr = sourceCoordinates.data();
  T *targetsPtr = targetCoordinates.data();
  T *chargesPtr = sourceCharges.data();
  size_t nSources = sourceCoordinates.size();
  size_t nTargets = targetCoordinates.size();
  size_t nCharges = charges.size();
  size_t *expansionOrderPtr = this->expansionOrder.data();

  if (depth.has_value()) {
    if (expansionOrder.size() != this->depth + 1) {
      throw std::invalid_argument("expansionOrder must have exactly depth + 1 "
                                  "elements when depth is set.");
    }
  } else {
    if (this->nCrit == 0) {
      throw std::invalid_argument("nCrit must be set if depth is not set.");
    }
    if (expansionOrder.size() != 1) {
      throw std::invalid_argument(
          "expansionOrder must have exactly one element when used with nCrit.");
    }
  }

  switch (this->fieldTranslation.type) {
  case FieldTranslationType::Blas:

    if constexpr (std::is_same_v<T, float>) {
      LaplaceBlas32 *fmm = laplace_blas_f32(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth,
          this->fieldTranslation.singularValueThreshold);
      fmmInstance.set(fmm);
    } else if constexpr (std::is_same_v<T, double>) {
      LaplaceBlas64 *fmm = laplace_blas_f64(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth,
          this->fieldTranslation.singularValueThreshold);
      fmmInstance.set(fmm);
    }
    break;
  case FieldTranslationType::Fft:
    if constexpr (std::is_same_v<T, float>) {
      LaplaceFft32 *fmm = laplace_fft_f32(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth, this->fieldTranslation.blockSize);
      fmmInstance.set(fmm);
    } else if constexpr (std::is_same_v<T, double>) {
      LaplaceFft64 *fmm = laplace_fft_f64(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth, this->fieldTranslation.blockSize);
      fmmInstance.set(fmm);
    }

    std::cout << "FFT " << std::endl;
    break;
  default:
    break;
  }
}

template <typename T> void KiFmm<T>::evaluate(bool timed) {

  switch (this->fieldTranslation.type) {
  case FieldTranslationType::Blas:
    if constexpr (std::is_same_v<T, float>) {
      LaplaceBlas32 *fmm =
          static_cast<LaplaceBlas32 *>(this->fmmInstance.get());
      evaluate_laplace_blas_f32(fmm, timed);
      std::cout << "Running FMM single precision " << std::endl;
    } else if constexpr (std::is_same_v<T, double>) {
      LaplaceBlas64 *fmm =
          static_cast<LaplaceBlas64 *>(this->fmmInstance.get());
      evaluate_laplace_blas_f64(fmm, timed);
      std::cout << "Running FMM double precision" << std::endl;
    }
    break;
  case FieldTranslationType::Fft:
    if constexpr (std::is_same_v<T, float>) {
      LaplaceFft32 *fmm = static_cast<LaplaceFft32 *>(this->fmmInstance.get());
      evaluate_laplace_fft_f32(fmm, timed);
      std::cout << "Running FMM single precision " << std::endl;
    } else if constexpr (std::is_same_v<T, double>) {
      LaplaceFft64 *fmm = static_cast<LaplaceFft64 *>(this->fmmInstance.get());
      evaluate_laplace_fft_f64(fmm, timed);
      std::cout << "Running FMM double precision" << std::endl;
    }
    break;
  }
}

FmmPointer::FmmPointer() : ptr(nullptr) {}

FmmPointer::~FmmPointer() { clear(); }

void FmmPointer::set(LaplaceBlas32 *ptr) {
  // clear();
  this->ptr = ptr;
}

void FmmPointer::set(LaplaceBlas64 *ptr) {
  // clear();
  this->ptr = ptr;
}

void FmmPointer::set(LaplaceFft32 *ptr) {
  // clear();
  this->ptr = ptr;
}

void FmmPointer::set(LaplaceFft64 *ptr) {
  // clear();
  this->ptr = ptr;
}

void *FmmPointer::get() const { return ptr; }

// FmmPointer::Type FmmPointer::getType() const { return type; }

void FmmPointer::clear() {
  //   if (type == Type::Blas32) {
  //     // destroyLaplaceBlas32(static_cast<LaplaceBlas32*>(ptr));
  //   } else if (type == Type::Blas64) {
  //     // destroyLaplaceBlas64(static_cast<LaplaceBlas64*>(ptr));
  //   }
  ptr = nullptr;
  // type = Type::None;
}

// Constructor for FieldTranslation
template <typename T>
FieldTranslation<T>::FieldTranslation(FieldTranslationType type,
                                      size_t targetRank,
                                      T singularValueThreshold)
    : type(type), singularValueThreshold(singularValueThreshold) {
  if (type == FieldTranslationType::Blas) {
    new (&data.blas) BlasFieldTranslation<T>(
        targetRank,
        singularValueThreshold); // Placement new with initializer list
  } else {
    throw std::invalid_argument("Invalid type for Blas constructor");
  }
}

template <typename T>
FieldTranslation<T>::FieldTranslation(FieldTranslationType type,
                                      size_t blockSize)
    : type(type), blockSize(blockSize) {
  if (type == FieldTranslationType::Fft) {
    new (&data.fft)
        FftFieldTranslation(blockSize); // Placement new with initializer list
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