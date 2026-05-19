#include <iostream>

#include "observation_codec.hpp"

int main() {
  hybrid_arena::EncodedObservation observation;
  observation.action_mask.fill(1.0F);
  const auto flat = hybrid_arena::FlattenObservation(observation);
  std::cout << "encoded_observation_size=" << flat.size() << "\n";
  std::cout << "onnxruntime_demo_requires_ONNXRUNTIME_ROOT\n";
  return 0;
}
