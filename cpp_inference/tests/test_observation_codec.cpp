#include <cassert>

#include "observation_codec.hpp"
#include "ring_buffer.hpp"

int main() {
  hybrid_arena::EncodedObservation observation;
  observation.action_mask.fill(1.0F);
  const auto flat = hybrid_arena::FlattenObservation(observation);
  assert(flat.size() == hybrid_arena::kLocalMapSize + hybrid_arena::kSelfStateSize +
                            hybrid_arena::kTeammateStateSize + hybrid_arena::kGlobalInfoSize +
                            hybrid_arena::kActionMaskSize);
  const auto action = hybrid_arena::DecodeAction(323);
  assert(action[0] == 8 && action[1] == 3 && action[2] == 8);

  hybrid_arena::RingBuffer<int, 2> buffer;
  assert(buffer.push(1));
  assert(buffer.push(2));
  assert(!buffer.push(3));
  assert(buffer.pop().value() == 1);
  return 0;
}
