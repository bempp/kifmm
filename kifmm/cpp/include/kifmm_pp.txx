
// Constructor implementation
template <typename T, typename V>
KiFmm<T, V>::KiFmm(const std::vector<size_t> &expansionOrder,
                const std::vector<T>& sources, const std::vector<T>& targets,
                std::vector<V>& charges, const Laplace<T> &kernel,
                const FieldTranslation<T> &fieldTranslation, bool pruneEmpty,
                std::optional<uint64_t> nCrit, std::optional<uint64_t> depth)
    : expansionOrder(expansionOrder), sourceCoordinates(sources),
      targetCoordinates(targets), sourceCharges(charges),
      fieldTranslation(fieldTranslation), pruneEmpty(pruneEmpty),
      nCrit(nCrit.value_or(0)), depth(depth.value_or(0)), kernel(kernel) {

  const T *sourcesPtr = sourceCoordinates.data();
  const T *targetsPtr = targetCoordinates.data();
  // const T *chargesPtr = sourceCharges.data();
  T *chargesPtr = static_cast<T *>(sourceCharges.data());
  const size_t *expansionOrderPtr = expansionOrder.data();
  size_t nSources = sourceCoordinates.size();
  size_t nTargets = targetCoordinates.size();
  size_t nCharges = charges.size();

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

  switch (this->fieldTranslation.mode) {
  case FieldTranslation<T>::Mode::Blas: {

    // Check if SVD is being done in Random mode
    bool rSvd = (this->fieldTranslation.blas.svdMode.mode == SvdMode<T>::Mode::Random);

    size_t nComponents = 0;
    size_t nOversamples = 0;
    size_t randomState = 0;
    if (rSvd) {
      nComponents = this->fieldTranslation.blas.svdMode.randomParams.nComponents;
      nOversamples = this->fieldTranslation.blas.svdMode.randomParams.nOversamples;
      randomState = this->fieldTranslation.blas.svdMode.randomParams.randomState;
    };

    if constexpr (std::is_same_v<T, float>) {
      LaplaceBlas32 *fmm = laplace_blas_f32(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth,
          this->fieldTranslation.blas.svdMode.singularValueThreshold,
          rSvd, nComponents, nOversamples, randomState);
      fmmInstance.set(fmm);
    } else if constexpr (std::is_same_v<T, double>) {

      LaplaceBlas64 *fmm = laplace_blas_f64(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth,
          this->fieldTranslation.blas.svdMode.singularValueThreshold,
          rSvd, nComponents, nOversamples, randomState);
      fmmInstance.set(fmm);
    }
    break;
  }

  case FieldTranslation<T>::Mode::Fft:
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

template <typename T, typename V>
KiFmm<T, V>::KiFmm(const std::vector<size_t> &expansionOrder,
                const std::vector<T>& sources, const std::vector<T>& targets,
                std::vector<V>& charges, const Helmholtz<T> &kernel,
                const FieldTranslation<T> &fieldTranslation, bool pruneEmpty,
                std::optional<uint64_t> nCrit, std::optional<uint64_t> depth)
    : expansionOrder(expansionOrder), sourceCoordinates(sources),
      targetCoordinates(targets), sourceCharges(charges),
      fieldTranslation(fieldTranslation), pruneEmpty(pruneEmpty),
      nCrit(nCrit.value_or(0)), depth(depth.value_or(0)), kernel(kernel) {

  const T *sourcesPtr = sourceCoordinates.data();
  const T *targetsPtr = targetCoordinates.data();
  T *chargesPtr = reinterpret_cast<T *>(sourceCharges.data());
  const size_t *expansionOrderPtr = expansionOrder.data();
  size_t nSources = sourceCoordinates.size();
  size_t nTargets = targetCoordinates.size();
  size_t nCharges = charges.size();

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

  switch (this->fieldTranslation.mode) {
  case FieldTranslation<T>::Mode::Blas: {

    // Check if SVD is being done in Random mode
    bool rSvd = (this->fieldTranslation.blas.svdMode.mode == SvdMode<T>::Mode::Random);

    size_t nComponents = 0;
    size_t nOversamples = 0;
    size_t randomState = 0;
    if (rSvd) {
      nComponents = this->fieldTranslation.blas.svdMode.randomParams.nComponents;
      nOversamples = this->fieldTranslation.blas.svdMode.randomParams.nOversamples;
      randomState = this->fieldTranslation.blas.svdMode.randomParams.randomState;
    };

    if constexpr (std::is_same_v<T, float>) {
      HelmholtzBlas32 *fmm = helmholtz_blas_f32(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth,
          this->fieldTranslation.blas.svdMode.singularValueThreshold,
          rSvd, nComponents, nOversamples, randomState);
      fmmInstance.set(fmm);
    } else if constexpr (std::is_same_v<T, double>) {

      HelmholtzBlas64 *fmm = helmholtz_blas_f64(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth,
          this->fieldTranslation.blas.svdMode.singularValueThreshold,
          rSvd, nComponents, nOversamples, randomState);
      fmmInstance.set(fmm);
    }
    break;
  }

  case FieldTranslation<T>::Mode::Fft:
    if constexpr (std::is_same_v<T, float>) {
      HelmholtzFft32 *fmm = helmholtz_fft_f32(
          expansionOrderPtr, this->expansionOrder.size(), sourcesPtr, nSources,
          targetsPtr, nTargets, chargesPtr, nCharges, this->pruneEmpty,
          this->nCrit, this->depth, this->fieldTranslation.blockSize);
      fmmInstance.set(fmm);
    } else if constexpr (std::is_same_v<T, double>) {
      HelmholtzFft64 *fmm = helmholtz_fft_f64(
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

template <typename T, typename V> void KiFmm<T, V>::clear(const std::vector<V> &charges) {}

template <typename T, typename V> void KiFmm<T, V>::evaluate(bool timed) {

  switch (this->fieldTranslation.mode) {
  case FieldTranslation<T>::Mode::Blas:
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
  case FieldTranslation<T>::Mode::Fft:
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

void FmmPointer::set(HelmholtzBlas32 *ptr) {
  // clear();
  this->ptr = ptr;
}

void FmmPointer::set(HelmholtzBlas64 *ptr) {
  // clear();
  this->ptr = ptr;
}

void FmmPointer::set(HelmholtzFft32 *ptr) {
  // clear();
  this->ptr = ptr;
}

void FmmPointer::set(HelmholtzFft64 *ptr) {
  // clear();
  this->ptr = ptr;
}

void *FmmPointer::get() const { return ptr; }

// FmmPointer::Type FmmPointer::getType() const { return type; }

void FmmPointer::clear() {
  //   if (mode == Type::Blas32) {
  //     // destroyLaplaceBlas32(static_cast<LaplaceBlas32*>(ptr));
  //   } else if (mode == Type::Blas64) {
  //     // destroyLaplaceBlas64(static_cast<LaplaceBlas64*>(ptr));
  //   }
  ptr = nullptr;
  // type = Type::None;
}

// Destructor
template <typename T> FieldTranslation<T>::~FieldTranslation() {}

// Default constructor sets to Deterministic mode
template <typename T>
SvdMode<T>::SvdMode(T singularValueThreshold) : singularValueThreshold(singularValueThreshold), mode(Mode::Deterministic) {}

// Constructor for Random mode
template <typename T>
SvdMode<T>::SvdMode(T singularValueThreshold, RandomParams params) : singularValueThreshold(singularValueThreshold), mode(Mode::Random), randomParams(params) {}

// Copy constructor
template <typename T>
SvdMode<T>::SvdMode(const SvdMode<T>& other) : mode(other.mode) {
    if (mode == Mode::Random) {
        new(&randomParams) RandomParams(other.randomParams);
    }
}

template <typename T>
FieldTranslation<T>::FieldTranslation(Mode mode, SvdMode<T> fmmSvdMode)
    : mode(mode), fmmSvdMode(fmmSvdMode) {
        new (&blas) BlasFieldTranslation(fmmSvdMode);
}

// Constructor for FieldTranslation with blockSize
template <typename T>
FieldTranslation<T>::FieldTranslation(Mode mode, size_t blockSize)
    : mode(mode), blockSize(blockSize) {
        new (&fft) FftFieldTranslation(blockSize);
}

template <typename T>
FieldTranslation<T>::FieldTranslation(const FieldTranslation& other) {
  new(&blas) BlasFieldTranslation(other.blas);
  new (&fft) FftFieldTranslation(other.fft);
}


// Destructor
template <typename T>
SvdMode<T>::~SvdMode() {
    if (mode == Mode::Random) {
        randomParams.~RandomParams();
    }
}

template <typename T>
SvdMode<T>::RandomParams::RandomParams(size_t nComponents,
                                       std::optional<size_t> nOversamples,
                                       std::optional<size_t> randomState)
    : nComponents(nComponents),
      nOversamples(nOversamples.value_or(10)),
      randomState(randomState.value_or(1)) {}