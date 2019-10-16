#include <cuda_runtime_api.h>

namespace detectron2 {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace detectron2
