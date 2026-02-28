// entropy_core.hpp
#ifndef ENTROPY_CORE_HPP
#define ENTROPY_CORE_HPP

#include <cstddef>  // Для size_t

/**
 * @brief Вычисляет энтропию группы признаков
 *
 * @param data Указатель на плоский массив данных в row-major порядке
 * @param n_rows Количество строк (образцов)
 * @param n_cols Количество столбцов (признаков)
 * @param mask Указатель на маску выбора признаков (ненулевые значения = выбранные)
 * @param mask_size Размер маски (должен быть равен n_cols)
 * @return double Энтропия выбранных признаков
 * @throws std::invalid_argument Если mask_size != n_cols
 */
double calculate_entropy(const int* data, size_t n_rows, size_t n_cols,
    const int* mask, size_t mask_size);

#endif // ENTROPY_CORE_HPP