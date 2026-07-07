#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>

// Minimal on-device adapter weight registry. Holds raw device pointers only
// -- the caller (pybind layer) is responsible for keeping the backing
// tensors alive for as long as the adapter is registered.
class AdapterStore {
 public:
  struct Entry {
    uintptr_t A_ptr;
    uintptr_t B_ptr;
    int32_t rank;
    int32_t d_model;
    int32_t out_features;
    float alpha_scale;
  };

  void register_adapter(int32_t adapter_id, uintptr_t A_ptr, uintptr_t B_ptr, int32_t rank, int32_t d_model,
                         int32_t out_features, float alpha_scale) {
    if ((A_ptr & 0xF) != 0 || (B_ptr & 0xF) != 0) {
      throw std::runtime_error("AdapterStore::register_adapter: A/B pointer not 16-byte aligned");
    }
    entries_[adapter_id] = Entry{A_ptr, B_ptr, rank, d_model, out_features, alpha_scale};
  }

  void evict(int32_t adapter_id) { entries_.erase(adapter_id); }

  bool contains(int32_t adapter_id) const { return entries_.count(adapter_id) != 0; }

  const Entry& lookup(int32_t adapter_id) const {
    auto it = entries_.find(adapter_id);
    if (it == entries_.end()) {
      throw std::runtime_error("AdapterStore::lookup: unknown adapter_id");
    }
    return it->second;
  }

 private:
  std::unordered_map<int32_t, Entry> entries_;
};
