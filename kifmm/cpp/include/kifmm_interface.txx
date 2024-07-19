// Constructor implementation
template <typename T>
KiFmm<T>::KiFmm(const std::vector<T> &source, const std::vector<T> &target,
                const std::vector<T> &charges)
    : sourceCoordinates(source), targetCoordinates(target),
      sourceCharges(charges) {

  T *sources_ptr = sourceCoordinates.data();
  size_t num_sources = sourceCoordinates.size() / 3;
  size_t num_targets  = targetCoordinates.size() / 3;
  size_t num_charges = charges.size();

  T *targets_ptr = targetCoordinates.data();
  T *charges_ptr = sourceCharges.data();

  LaplaceBlas32 *fmm = laplace_blas_f32(sources_ptr, num_sources, targets_ptr,
                                        num_targets, charges_ptr, num_charges);
  fmmInstance.set(fmm);
}

// Member function implementation (example)
template <typename T> void KiFmm<T>::printCoordinates() const {
  std::cout << "Source Coordinates: ";
  for (const auto &coord : sourceCoordinates) {
    std::cout << coord << " ";
  }
  std::cout << std::endl;

  std::cout << "Target Coordinates: ";
  for (const auto &coord : targetCoordinates) {
    std::cout << coord << " ";
  }
  std::cout << std::endl;
}

FmmPointer::FmmPointer() : ptr(nullptr), type(Type::None) {}

FmmPointer::~FmmPointer() { clear(); }

void FmmPointer::set(LaplaceBlas32 *ptr) {
  clear();
  this->ptr = ptr;
  type = Type::Blas32;
}

// void FmmPointer::set(LaplaceBlas32* ptr) {
//     clear();
//     this->ptr = ptr;
//     type = Type::Blas64;
// }

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