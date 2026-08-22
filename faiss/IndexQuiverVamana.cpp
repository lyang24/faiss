/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexQuiverVamana.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/VisitedTable.h>

namespace faiss {

namespace {

constexpr int32_t kEmpty = -1;

struct NodeDistance {
    uint32_t distance;
    int32_t id;
};

struct CloserFirst {
    bool operator()(const NodeDistance& a, const NodeDistance& b) const {
        return a.distance != b.distance ? a.distance > b.distance : a.id > b.id;
    }
};

struct FartherFirst {
    bool operator()(const NodeDistance& a, const NodeDistance& b) const {
        return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
    }
};

struct BuildGraph {
    idx_t n;
    int capacity;
    std::vector<int32_t> degree;
    std::vector<int32_t> neighbors;

    BuildGraph(idx_t n, int capacity)
            : n(n),
              capacity(capacity),
              degree(static_cast<size_t>(n), 0),
              neighbors(static_cast<size_t>(n) * capacity, kEmpty) {}

    int32_t* begin(idx_t node) {
        return neighbors.data() + static_cast<size_t>(node) * capacity;
    }

    const int32_t* begin(idx_t node) const {
        return neighbors.data() + static_cast<size_t>(node) * capacity;
    }

    bool contains(idx_t node, int32_t value) const {
        const int32_t* values = begin(node);
        return std::find(values, values + degree[node], value) !=
                values + degree[node];
    }

    void set(idx_t node, const std::vector<int32_t>& values) {
        FAISS_THROW_IF_NOT(values.size() <= static_cast<size_t>(capacity));
        int32_t* out = begin(node);
        std::copy(values.begin(), values.end(), out);
        std::fill(out + values.size(), out + capacity, kEmpty);
        degree[node] = static_cast<int32_t>(values.size());
    }

    void append(idx_t node, int32_t value) {
        FAISS_THROW_IF_NOT(degree[node] < capacity);
        begin(node)[degree[node]++] = value;
    }
};

std::vector<NodeDistance> beam_search_codes(
        const IndexQuiver& codes,
        const BuildGraph& graph,
        int32_t target,
        int32_t entry,
        int beam,
        VisitedTable& visited) {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, CloserFirst>
            candidates;
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst>
            results;

    const uint32_t initial = codes.code_distance(target, entry);
    candidates.push({initial, entry});
    results.push({initial, entry});
    visited.set(entry);

    while (!candidates.empty()) {
        const NodeDistance current = candidates.top();
        candidates.pop();
        if (static_cast<int>(results.size()) >= beam &&
            current.distance > results.top().distance) {
            break;
        }

        const int32_t* neighbors = graph.begin(current.id);
        const int degree = graph.degree[current.id];
        for (int j = 0; j < degree; ++j) {
            const int32_t neighbor = neighbors[j];
            if (neighbor < 0 || !visited.set(neighbor)) {
                continue;
            }
            const uint32_t distance = codes.code_distance(target, neighbor);
            if (static_cast<int>(results.size()) < beam ||
                distance < results.top().distance) {
                candidates.push({distance, neighbor});
                results.push({distance, neighbor});
                if (static_cast<int>(results.size()) > beam) {
                    results.pop();
                }
            }
        }
    }

    std::vector<NodeDistance> output;
    output.reserve(results.size());
    while (!results.empty()) {
        output.push_back(results.top());
        results.pop();
    }
    std::sort(output.begin(), output.end(), [](const auto& a, const auto& b) {
        return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
    });
    visited.advance();
    return output;
}

std::vector<int32_t> robust_prune(
        const IndexQuiver& codes,
        int32_t target,
        std::vector<NodeDistance> candidates,
        int max_degree,
        float alpha) {
    std::sort(
            candidates.begin(),
            candidates.end(),
            [](const auto& a, const auto& b) {
                return a.id != b.id ? a.id < b.id : a.distance < b.distance;
            });
    candidates.erase(
            std::unique(
                    candidates.begin(),
                    candidates.end(),
                    [](const auto& a, const auto& b) { return a.id == b.id; }),
            candidates.end());
    candidates.erase(
            std::remove_if(
                    candidates.begin(),
                    candidates.end(),
                    [target](const auto& candidate) {
                        return candidate.id == target;
                    }),
            candidates.end());
    std::sort(
            candidates.begin(),
            candidates.end(),
            [](const auto& a, const auto& b) {
                return a.distance != b.distance ? a.distance < b.distance
                                                : a.id < b.id;
            });

    std::vector<int32_t> selected;
    selected.reserve(max_degree);
    for (const NodeDistance& candidate : candidates) {
        bool dominated = false;
        for (int32_t kept : selected) {
            const uint32_t between = codes.code_distance(candidate.id, kept);
            if (alpha * static_cast<float>(between) <=
                static_cast<float>(candidate.distance)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            selected.push_back(candidate.id);
            if (static_cast<int>(selected.size()) == max_degree) {
                return selected;
            }
        }
    }

    // QuIVer stores a fixed degree whenever the candidate pool permits it.
    // Diversity-selected neighbors stay first; nearest remaining candidates
    // provide local redundancy and deterministic handling of distance ties.
    for (const NodeDistance& candidate : candidates) {
        if (static_cast<int>(selected.size()) == max_degree) {
            break;
        }
        if (std::find(selected.begin(), selected.end(), candidate.id) ==
            selected.end()) {
            selected.push_back(candidate.id);
        }
    }
    return selected;
}

int32_t choose_entry(const IndexQuiver& codes, idx_t n, uint64_t seed) {
    if (n <= 1) {
        return 0;
    }
    std::mt19937_64 rng(seed ^ 0x4d595df4d0f33173ULL);
    const int candidate_count = static_cast<int>(std::min<idx_t>(256, n));
    const int sample_count = static_cast<int>(std::min<idx_t>(4096, n));
    std::vector<int32_t> candidates(candidate_count);
    std::vector<int32_t> samples(sample_count);
    for (int32_t& candidate : candidates) {
        candidate = static_cast<int32_t>(rng() % static_cast<uint64_t>(n));
    }
    for (int32_t& sample : samples) {
        sample = static_cast<int32_t>(rng() % static_cast<uint64_t>(n));
    }

    uint64_t best_sum = std::numeric_limits<uint64_t>::max();
    int32_t best = candidates.front();
    for (int32_t candidate : candidates) {
        uint64_t sum = 0;
        for (int32_t sample : samples) {
            sum += codes.code_distance(candidate, sample);
        }
        if (sum < best_sum || (sum == best_sum && candidate < best)) {
            best_sum = sum;
            best = candidate;
        }
    }
    return best;
}

template <class DistanceFn, class NeighborFn>
std::vector<NodeDistance> beam_search_query(
        DistanceFn&& distance,
        NeighborFn&& neighbors,
        int32_t entry,
        int beam,
        VisitedTable& visited) {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, CloserFirst>
            candidates;
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst>
            results;

    const uint32_t initial = distance(entry);
    candidates.push({initial, entry});
    results.push({initial, entry});
    visited.set(entry);

    while (!candidates.empty()) {
        const NodeDistance current = candidates.top();
        candidates.pop();
        if (static_cast<int>(results.size()) >= beam &&
            current.distance > results.top().distance) {
            break;
        }
        const auto range = neighbors(current.id);
        for (const int32_t* it = range.first; it != range.second; ++it) {
            const int32_t neighbor = *it;
            if (neighbor < 0 || !visited.set(neighbor)) {
                continue;
            }
            const uint32_t next_distance = distance(neighbor);
            if (static_cast<int>(results.size()) < beam ||
                next_distance < results.top().distance) {
                candidates.push({next_distance, neighbor});
                results.push({next_distance, neighbor});
                if (static_cast<int>(results.size()) > beam) {
                    results.pop();
                }
            }
        }
    }

    std::vector<NodeDistance> output;
    output.reserve(results.size());
    while (!results.empty()) {
        output.push_back(results.top());
        results.pop();
    }
    std::sort(output.begin(), output.end(), [](const auto& a, const auto& b) {
        return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
    });
    visited.advance();
    return output;
}

} // namespace

IndexQuiverVamana::IndexQuiverVamana()
        : IndexNSG(0, 64, METRIC_INNER_PRODUCT) {}

IndexQuiverVamana::IndexQuiverVamana(int d_in, int m_in)
        : IndexNSG(new IndexQuiver(d_in), 2 * m_in), m(m_in) {
    FAISS_THROW_IF_NOT_MSG(m > 0, "QuIVer Vamana m must be positive");
    own_fields = true;
    is_trained = true;
    nsg.search_L = search_ef;
}

void IndexQuiverVamana::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            !is_built && ntotal == 0,
            "IndexQuiverVamana supports one-shot construction only");
    FAISS_THROW_IF_NOT_MSG(n > 0, "cannot build an empty Vamana graph");
    FAISS_THROW_IF_NOT_MSG(
            n <= std::numeric_limits<int32_t>::max(),
            "IndexQuiverVamana supports at most INT32_MAX vectors");
    FAISS_THROW_IF_NOT(storage && is_trained);
    FAISS_THROW_IF_NOT(construction_ef > 0 && alpha >= 1.0f);

    storage->add(n, x);
    ntotal = storage->ntotal;
    auto* codes = dynamic_cast<IndexQuiver*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            codes, "IndexQuiverVamana requires IndexQuiver storage");

    const int max_degree = nsg.R;
    const int initial_degree =
            static_cast<int>(std::min<idx_t>(max_degree, n - 1));
    const int capacity = std::max(max_degree * 2, 1);
    BuildGraph graph(n, capacity);
    std::mt19937_64 rng(random_seed);

    // Standard Vamana starts from a random bounded-degree graph. This keeps
    // every pre-installed node reachable while BQ RobustPrune rewrites edges.
    for (idx_t node = 0; node < n; ++node) {
        while (graph.degree[node] < initial_degree) {
            const int32_t neighbor =
                    static_cast<int32_t>(rng() % static_cast<uint64_t>(n));
            if (neighbor != node && !graph.contains(node, neighbor)) {
                graph.append(node, neighbor);
            }
        }
    }

    nsg.enterpoint = choose_entry(*codes, n, random_seed);
    std::vector<int32_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::unique_ptr<VisitedTable> visited = VisitedTable::create(n, false);

    for (idx_t position = 0; position < n; ++position) {
        const int32_t node = order[position];
        std::vector<NodeDistance> candidates = beam_search_codes(
                *codes,
                graph,
                node,
                nsg.enterpoint,
                std::max(construction_ef, max_degree),
                *visited);
        const int32_t* old_neighbors = graph.begin(node);
        for (int j = 0; j < graph.degree[node]; ++j) {
            const int32_t neighbor = old_neighbors[j];
            candidates.push_back(
                    {codes->code_distance(node, neighbor), neighbor});
        }

        const std::vector<int32_t> selected = robust_prune(
                *codes, node, std::move(candidates), max_degree, alpha);
        graph.set(node, selected);

        // Reverse links use 2R temporary headroom and are pruned only when
        // full. This is equivalent to batching reverse-edge insertions before
        // RobustPrune and avoids rerunning O(R^2) pruning for every edge.
        for (int32_t neighbor : selected) {
            if (graph.contains(neighbor, node)) {
                continue;
            }
            if (graph.degree[neighbor] < capacity) {
                graph.append(neighbor, node);
                continue;
            }
            std::vector<NodeDistance> reverse;
            reverse.reserve(static_cast<size_t>(capacity) + 1);
            const int32_t* values = graph.begin(neighbor);
            for (int j = 0; j < graph.degree[neighbor]; ++j) {
                reverse.push_back(
                        {codes->code_distance(neighbor, values[j]), values[j]});
            }
            reverse.push_back({codes->code_distance(neighbor, node), node});
            graph.set(
                    neighbor,
                    robust_prune(
                            *codes,
                            neighbor,
                            std::move(reverse),
                            max_degree,
                            alpha));
        }

        if ((position & 0x3fff) == 0) {
            InterruptCallback::check();
        }
    }

    // Converge lazy reverse-link headroom to the advertised degree bound.
#pragma omp parallel for
    for (idx_t node = 0; node < n; ++node) {
        if (graph.degree[node] <= max_degree) {
            continue;
        }
        std::vector<NodeDistance> candidates;
        candidates.reserve(graph.degree[node]);
        const int32_t* values = graph.begin(node);
        for (int j = 0; j < graph.degree[node]; ++j) {
            candidates.push_back(
                    {codes->code_distance(node, values[j]), values[j]});
        }
        graph.set(
                node,
                robust_prune(
                        *codes,
                        static_cast<int32_t>(node),
                        std::move(candidates),
                        max_degree,
                        alpha));
    }

    nsg.final_graph = std::make_shared<nsg::Graph<int32_t>>(
            static_cast<int>(n), max_degree);
    std::fill_n(
            nsg.final_graph->data, static_cast<size_t>(n) * max_degree, kEmpty);
#pragma omp parallel for
    for (idx_t node = 0; node < n; ++node) {
        const int count = std::min(graph.degree[node], max_degree);
        std::copy_n(
                graph.begin(node),
                count,
                nsg.final_graph->data + static_cast<size_t>(node) * max_degree);
    }
    nsg.ntotal = static_cast<int>(n);
    nsg.is_built = true;
    nsg.search_L = search_ef;
    is_built = true;
}

void IndexQuiverVamana::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(is_built && nsg.final_graph && storage);
    FAISS_THROW_IF_NOT(k > 0 && k <= ntotal);
    int beam = search_ef;
    const IDSelector* selector = params ? params->sel : nullptr;
    if (params) {
        if (const auto* quiver_params =
                    dynamic_cast<const SearchParametersQuiverVamana*>(params)) {
            if (quiver_params->search_ef > 0) {
                beam = quiver_params->search_ef;
            }
        } else if (typeid(*params) != typeid(SearchParameters)) {
            FAISS_THROW_MSG("unsupported QuIVer Vamana search parameters");
        }
    }
    beam = std::max<int>(beam, static_cast<int>(k));

    std::exception_ptr exception;
    std::atomic<bool> interrupted{false};
#pragma omp parallel
    {
        try {
            std::unique_ptr<DistanceComputer> dc(
                    storage->get_distance_computer());
            std::unique_ptr<VisitedTable> visited =
                    VisitedTable::create(ntotal, false);
#pragma omp for
            for (idx_t query = 0; query < n; ++query) {
                if (interrupted.load(std::memory_order_relaxed)) {
                    continue;
                }
                dc->set_query(x + query * d);
                auto distance = [&dc](int32_t id) {
                    return static_cast<uint32_t>(-(*dc)(id));
                };
                auto neighbors = [this](int32_t id) {
                    const int32_t* begin = nsg.final_graph->data +
                            static_cast<size_t>(id) * nsg.R;
                    const int32_t* end = begin;
                    while (end != begin + nsg.R && *end >= 0) {
                        ++end;
                    }
                    return std::make_pair(begin, end);
                };
                const std::vector<NodeDistance> found = beam_search_query(
                        distance, neighbors, nsg.enterpoint, beam, *visited);

                idx_t output = 0;
                for (const NodeDistance& candidate : found) {
                    if (selector && !selector->is_member(candidate.id)) {
                        continue;
                    }
                    labels[query * k + output] = candidate.id;
                    distances[query * k + output] =
                            -static_cast<float>(candidate.distance);
                    if (++output == k) {
                        break;
                    }
                }
                while (output < k) {
                    labels[query * k + output] = -1;
                    distances[query * k + output] =
                            -std::numeric_limits<float>::infinity();
                    ++output;
                }
            }
        } catch (...) {
#pragma omp critical
            {
                if (!exception) {
                    exception = std::current_exception();
                }
                interrupted.store(true, std::memory_order_relaxed);
            }
        }
    }
    if (exception) {
        std::rethrow_exception(exception);
    }
    InterruptCallback::check();
}

void IndexQuiverVamana::reset() {
    IndexNSG::reset();
    nsg.search_L = search_ef;
}

int IndexQuiverVamana::get_degree(idx_t i) const {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal && nsg.final_graph);
    int degree = 0;
    while (degree < nsg.R && nsg.final_graph->at(i, degree) >= 0) {
        ++degree;
    }
    return degree;
}

} // namespace faiss
