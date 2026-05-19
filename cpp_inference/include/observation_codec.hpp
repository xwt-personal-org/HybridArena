#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace hybrid_arena {

constexpr std::size_t kLocalMapSize = 11 * 11 * 11;
constexpr std::size_t kSelfStateSize = 20;
constexpr std::size_t kTeammateStateSize = 3 * 15;
constexpr std::size_t kGlobalInfoSize = 10;
constexpr std::size_t kActionMaskSize = 324;

struct EncodedObservation {
  std::array<float, kLocalMapSize> local_map{};
  std::array<float, kSelfStateSize> self_state{};
  std::array<float, kTeammateStateSize> teammate_states{};
  std::array<float, kGlobalInfoSize> global_info{};
  std::array<float, kActionMaskSize> action_mask{};
};

std::vector<float> FlattenObservation(const EncodedObservation& observation);
std::array<std::int64_t, 3> DecodeAction(std::int64_t flat_action);

}  // namespace hybrid_arena
