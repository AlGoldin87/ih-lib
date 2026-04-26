#include "../include/rule_finder.hpp"
#include "../include/entropy_core.hpp"
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <limits>

// ==================== ВСПОМОГАТЕЛЬНЫЕ СТРУКТУРЫ ====================

struct VectorHash {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t seed = vec.size();
        for (size_t i = 0; i < vec.size(); i++) {
            seed ^= std::hash<int>{}(vec[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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

// ==================== ВНУТРЕННЯЯ ФУНКЦИЯ ЭНТРОПИИ ====================

double calc_entropy_internal(const std::vector<std::vector<int>>& data, const std::vector<int>& mask) {
    size_t n_rows = data.size();
    if (n_rows == 0) return 0.0;
    
    size_t n_cols = data[0].size();
    if (mask.size() != n_cols) return 0.0;
    
    std::unordered_map<std::vector<int>, int, VectorHash, VectorEqual> counts;
    
    for (size_t i = 0; i < n_rows; i++) {
        std::vector<int> key;
        for (size_t j = 0; j < n_cols; j++) {
            if (mask[j]) {
                key.push_back(data[i][j]);
            }
        }
        counts[key]++;
    }
    
    double h_sum = 0.0;
    double n = static_cast<double>(n_rows);
    
    for (auto it = counts.begin(); it != counts.end(); ++it) {
        double c = static_cast<double>(it->second);
        h_sum += c * std::log2(c);
    }
    
    return (n * std::log2(n) - h_sum) / n;
}

// ==================== ОЦЕНКА ДЛЯ КОЛИЧЕСТВЕННОГО ПРИЗНАКА ====================

// Для количественного признака: data_2d строится из уже дискретизированного столбца
BinarySplitResult evaluate_quantitative_mask(
    const std::vector<int>& discrete_feature,
    const std::vector<int>& y_binary,
    const std::vector<int>& mask,
    int L, int R, bool inverted,
    const std::string& feature_name) {
    
    size_t rows = discrete_feature.size();
    std::vector<std::vector<int>> data_2d(rows, std::vector<int>(2));
    
    int class0 = 0, class1 = 0;
    
    for (size_t i = 0; i < rows; i++) {
        int x_val = discrete_feature[i];
        data_2d[i][0] = mask[x_val];
        data_2d[i][1] = y_binary[i];
        
        if (y_binary[i] == 0) class0++;
        else class1++;
    }
    
    std::vector<int> mask_x = {1, 0};
    std::vector<int> mask_y = {0, 1};
    std::vector<int> mask_xy = {1, 1};
    
    double Hx = calc_entropy_internal(data_2d, mask_x);
    double Hy = calc_entropy_internal(data_2d, mask_y);
    double Hxy = calc_entropy_internal(data_2d, mask_xy);
    
    double Ixy = Hx + Hy - Hxy;
    if (Ixy < 0 && Ixy > -1e-10) Ixy = 0.0;
    double Rxy = (Hx > 1e-10) ? Ixy / Hy : 0.0;
    
    std::ostringstream rule;
    if (inverted) {
        rule << feature_name << " NOT in interval [" << L << "," << R << "] → class 1";
    } else {
        rule << feature_name << " in interval [" << L << "," << R << "] → class 1";
    }
    rule << " (Rxy=" << Rxy << ")";
    
    BinarySplitResult res;
    res.Rxy = Rxy;
    res.L = L;
    res.R = R;
    res.inverted = inverted;
    res.Hx = Hx;
    res.Hy = Hy;
    res.Ixy = Ixy;
    res.class0_count = class0;
    res.class1_count = class1;
    res.interval_start = static_cast<double>(L);
    res.interval_end = static_cast<double>(R);
    res.feature_name = feature_name;
    res.rule_text = rule.str();
    
    return res;
}

// ==================== ПОИСК ДЛЯ КОЛИЧЕСТВЕННОГО ====================

BinarySplitResult find_best_quantitative(
    const std::vector<int>& discrete_feature,
    const std::vector<int>& y_binary,
    int n_intervals,
    const std::string& feature_name) {
    
    BinarySplitResult best;
    best.Rxy = -1.0;
    
    for (int L = 0; L < n_intervals; L++) {
        for (int R = L; R < n_intervals; R++) {
            // Маска для интервала [L, R]
            std::vector<int> mask1(n_intervals, 0);
            for (int k = L; k <= R; k++) mask1[k] = 1;
            
            BinarySplitResult res1 = evaluate_quantitative_mask(
                discrete_feature, y_binary, mask1, L, R, false, feature_name);
            
            if (res1.Rxy > best.Rxy) best = res1;
            
            // Инвертированная маска (НЕ в интервале)
            std::vector<int> mask2(n_intervals, 1);
            for (int k = L; k <= R; k++) mask2[k] = 0;
            
            BinarySplitResult res2 = evaluate_quantitative_mask(
                discrete_feature, y_binary, mask2, L, R, true, feature_name);
            
            if (res2.Rxy > best.Rxy) best = res2;
        }
    }
    
    return best;
}

// ==================== ПЕРЕКОДИРОВКА КАТЕГОРИЙ В 0..K-1 ====================

std::pair<std::vector<int>, int> renumber_categories(const std::vector<int>& raw_categories) {
    std::unordered_map<int, int> code_map;
    std::vector<int> encoded;
    encoded.reserve(raw_categories.size());
    
    int next_code = 0;
    for (int val : raw_categories) {
        auto it = code_map.find(val);
        if (it == code_map.end()) {
            code_map[val] = next_code;
            encoded.push_back(next_code);
            next_code++;
        } else {
            encoded.push_back(it->second);
        }
    }
    
    return {encoded, next_code};
}

// ==================== ОЦЕНКА ДЛЯ КАТЕГОРИАЛЬНОГО ====================

BinarySplitResult evaluate_categorical_mask(
    const std::vector<int>& categorical_feature,
    const std::vector<int>& y_binary,
    const std::vector<int>& category_mask,
    const std::string& feature_name) {
    
    size_t rows = categorical_feature.size();
    std::vector<std::vector<int>> data_2d(rows, std::vector<int>(2));
    
    int class0 = 0, class1 = 0;
    
    for (size_t i = 0; i < rows; i++) {
        int cat = categorical_feature[i];
        data_2d[i][0] = category_mask[cat];
        data_2d[i][1] = y_binary[i];
        
        if (y_binary[i] == 0) class0++;
        else class1++;
    }
    
    std::vector<int> mask_x = {1, 0};
    std::vector<int> mask_y = {0, 1};
    std::vector<int> mask_xy = {1, 1};
    
    double Hx = calc_entropy_internal(data_2d, mask_x);
    double Hy = calc_entropy_internal(data_2d, mask_y);
    double Hxy = calc_entropy_internal(data_2d, mask_xy);
    
    double Ixy = Hx + Hy - Hxy;
    if (Ixy < 0 && Ixy > -1e-10) Ixy = 0.0;
    double Rxy = (Hx > 1e-10) ? Ixy / Hy : 0.0;
    
    std::ostringstream cat_list;
    cat_list << "{";
    bool first = true;
    for (size_t k = 0; k < category_mask.size(); k++) {
        if (category_mask[k]) {
            if (!first) cat_list << ",";
            cat_list << k;
            first = false;
        }
    }
    cat_list << "}";
    
    std::ostringstream rule;
    rule << feature_name << " in " << cat_list.str() << " → class 1 (Rxy=" << Rxy << ")";
    
    BinarySplitResult res;
    res.Rxy = Rxy;
    res.Hx = Hx;
    res.Hy = Hy;
    res.Ixy = Ixy;
    res.class0_count = class0;
    res.class1_count = class1;
    res.feature_name = feature_name;
    res.rule_text = rule.str();
    res.category_mask = category_mask;
    
    return res;
}

// ==================== ПОИСК ДЛЯ КАТЕГОРИАЛЬНОГО ====================

BinarySplitResult find_best_categorical(
    const std::vector<int>& categorical_feature,
    const std::vector<int>& y_binary,
    const std::string& feature_name) {
    
    // Перекодируем в последовательные 0,1,2,...,K-1
    auto [encoded, n_categories] = renumber_categories(categorical_feature);
    
    // Защита от слишком большого числа категорий
    if (n_categories > 12) {
        BinarySplitResult dummy;
        dummy.Rxy = 0.0;
        dummy.feature_name = feature_name;
        dummy.rule_text = feature_name + ": too many categories (" + std::to_string(n_categories) + "), skipping exhaustive search";
        return dummy;
    }
    
    BinarySplitResult best;
    best.Rxy = -1.0;
    
    // Перебираем все непустые и неполные подмножества
    for (int subset = 1; subset < (1 << n_categories) - 1; subset++) {
        std::vector<int> mask(n_categories, 0);
        for (int k = 0; k < n_categories; k++) {
            if (subset >> k & 1) mask[k] = 1;
        }
        
        BinarySplitResult res = evaluate_categorical_mask(
            encoded, y_binary, mask, feature_name);
        
        if (res.Rxy > best.Rxy) best = res;
    }
    
    return best;
}

// ==================== ГЛАВНАЯ ФУНКЦИЯ (НОВАЯ СИГНАТУРА) ====================

std::vector<BinarySplitResult> find_best_rules(
    const std::vector<std::vector<int>>& prepared_data,
    const std::vector<int>& feature_mask,
    int y_index,
    const std::vector<std::string>& feature_names) {
    
    std::vector<BinarySplitResult> results;
    
    if (prepared_data.empty() || feature_mask.empty()) {
        return results;
    }
    
    size_t n_rows = prepared_data.size();
    size_t n_cols = prepared_data[0].size();
    
    if (feature_mask.size() != feature_names.size() ||
        feature_mask.size() != n_cols) {
        return results;
    }
    
    // Извлекаем Y (уже бинарный 0/1)
    std::vector<int> y_binary(n_rows);
    for (size_t i = 0; i < n_rows; i++) {
        y_binary[i] = prepared_data[i][y_index];
    }
    
    // Определяем количество интервалов для каждого количественного признака
    // (нужно для перебора)
    std::vector<int> n_intervals_per_feature(n_cols, 0);
    for (size_t col = 0; col < n_cols; col++) {
        if (col == (size_t)y_index) continue;
        int max_val = 0;
        for (size_t i = 0; i < n_rows; i++) {
            max_val = std::max(max_val, prepared_data[i][col]);
        }
        n_intervals_per_feature[col] = max_val + 1;
    }
    
    // Обрабатываем каждый признак
    for (size_t col = 0; col < n_cols; col++) {
        if (col == (size_t)y_index) continue;
        if (feature_mask[col] == 0) continue;  // исключенный признак
        
        std::string feature_name = feature_names[col];
        
        // Извлекаем столбец
        std::vector<int> feature_col(n_rows);
        for (size_t i = 0; i < n_rows; i++) {
            feature_col[i] = prepared_data[i][col];
        }
        
        BinarySplitResult res;
        
        if (feature_mask[col] == 1) {
            // Количественный признак
            int n_intervals = n_intervals_per_feature[col];
            res = find_best_quantitative(feature_col, y_binary, n_intervals, feature_name);
        } else if (feature_mask[col] == 2) {
            // Категориальный признак
            res = find_best_categorical(feature_col, y_binary, feature_name);
        } else {
            continue;  // неизвестный тип
        }
        
        results.push_back(res);
    }
    
    return results;
}
