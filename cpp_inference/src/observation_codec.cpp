#include "observation_codec.hpp"

#include <stdexcept>

namespace hybrid_arena {

std::vector<float> FlattenObservation(const EncodedObservation& observation) {
  std::vector<float> output;
  output.reserve(kLocalMapSize + kSelfStateSize + kTeammateStateSize + kGlobalInfoSize + kActionMaskSize);
  output.insert(output.end(), observation.local_map.begin(), observation.local_map.end());
  output.insert(output.end(), observation.self_state.begin(), observation.self_state.end());
  output.insert(output.end(), observation.teammate_states.begin(), observation.teammate_states.end());
  output.insert(output.end(), observation.global_info.begin(), observation.global_info.end());
  output.insert(output.end(), observation.action_mask.begin(), observation.action_mask.end());
  return output;
}

std::array<std::int64_t, 3> DecodeAction(std::int64_t flat_action) {
  if (flat_action < 0 || flat_action >= static_cast<std::int64_t>(kActionMaskSize)) {
    throw std::out_of_range("flat action must be in [0, 323]");
  }
  const std::int64_t move = flat_action / (4 * 9);
  const std::int64_t skill = (flat_action % (4 * 9)) / 9;
  const std::int64_t target = flat_action % 9;
  return {move, skill, target};
}

}  // namespace hybrid_arena
