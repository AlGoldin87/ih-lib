#ifndef RULE_FINDER_HPP
#define RULE_FINDER_HPP

#include <vector>
#include <string>

struct BinarySplitResult {
    double Rxy;
    int L;
    int R;
    bool inverted;
    double Hx;
    double Hy;
    double Ixy;
    int class0_count;
    int class1_count;
    double interval_start;      // для количественных (в исходных единицах)
    double interval_end;        // для количественных
    std::string feature_name;
    std::string rule_text;
    std::vector<int> category_mask;  // для категориальных: какие категории в классе 1
};

// Новая сигнатура: принимает уже подготовленные данные (int32 от ih-prep)
std::vector<BinarySplitResult> find_best_rules(
    const std::vector<std::vector<int>>& prepared_data,  // матрица int32
    const std::vector<int>& feature_mask,                // 0=не участвует, 1=колич, 2=категор
    int y_index,                                         // индекс целевой (уже 0/1)
    const std::vector<std::string>& feature_names        // имена признаков
);

#endif // RULE_FINDER_HPP
