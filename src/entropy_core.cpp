// entropy_core.cpp
#include "entropy_core.hpp"
#include <vector>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <string>

// Векторные хэш-функции (оставляем как было)
struct VectorHash {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t seed = vec.size();
        for (int value : vec) {
            seed ^= std::hash<int>{}(value)+0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct VectorEqual {
    bool operator()(const std::vector<int>& a, const std::vector<int>& b) const {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
};

double calculate_entropy(const int* data, size_t n_rows, size_t n_cols,
    const int* mask, size_t mask_size)
{
    // Проверка входных указателей
    if (!data || !mask) {
        throw std::invalid_argument("Data or mask pointer is null");
    }

    if (n_rows == 0 || n_cols == 0) {
        return 0.0;  // Пустые данные - энтропия 0
    }

    // Критическая проверка: размер маски должен совпадать с количеством признаков
    if (mask_size != n_cols) {
        throw std::invalid_argument(
            std::string("Mask size (") + std::to_string(mask_size) +
            ") doesn't match number of columns (" + std::to_string(n_cols) + ")"
        );
    }

    // Подсчёт активных признаков
    bool any_active = false;
    size_t active_count = 0;
    for (size_t j = 0; j < n_cols; j++) {
        if (mask[j] != 0) {
            any_active = true;
            active_count++;
        }
    }

    if (!any_active) {
        return 0.0;  // Ни один признак не выбран - энтропия 0
    }

    // Хэш-таблица для подсчёта частот комбинаций
    std::unordered_map<std::vector<int>, int, VectorHash, VectorEqual> counts;
    counts.reserve(std::min(n_rows, size_t(1000)));

    // Подсчёт уникальных комбинаций
    for (size_t i = 0; i < n_rows; i++) {
        std::vector<int> key;
        key.reserve(active_count);

        for (size_t j = 0; j < n_cols; j++) {
            if (mask[j] != 0) {
                key.push_back(data[i * n_cols + j]);
            }
        }
        counts[key]++;
    }

    // Вычисление энтропии: H = (n*log2(n) - Σ count*log2(count)) / n
    double h_sum = 0.0;
    double n = static_cast<double>(n_rows);

    for (const auto& entry : counts) {
        double count = static_cast<double>(entry.second);
        if (count > 0) {
            h_sum += count * std::log2(count);
        }
    }

    if (n <= 0) {
        return 0.0;
    }

    return (n * std::log2(n) - h_sum) / n;
}