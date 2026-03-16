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
    double interval_start;
    double interval_end;
    std::string feature_name;
    std::string rule_text;
};

std::vector<BinarySplitResult> find_best_rules(
    const std::vector<std::vector<float>>& raw_data,
    const std::vector<int>& feature_mask,
    int y_index,
    double y_class_low,
    double y_class_high,
    const std::vector<double>& sharpness_list,
    const std::vector<std::string>& feature_names
);

#endif // RULE_FINDER_HPP