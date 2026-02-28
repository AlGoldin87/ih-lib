# test_108x7.py
import numpy as np
from ih import calculate_entropy

# ПОЛНЫЕ ДАННЫЕ 108x7 (все строки из вашего C++ кода)
data = np.array([
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 1, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0]
], dtype=np.int32)

print("=" * 60)
print("EXACT TEST ON REFERENCE DATA 108x7")
print("=" * 60)
print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
print()

# TEST 1: H(X1) = 0.908757
print("TEST 1: H(X1)")
mask1 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
h1 = calculate_entropy(data, mask1)
print(f"  Calculated: {h1:.6f}")
print(f"  Expected:   0.908757")
print(f"  Difference: {abs(h1 - 0.908757):.6f}")
test1_passed = abs(h1 - 0.908757) < 0.00001
print(f"  Result:     {'PASSED' if test1_passed else 'FAILED'}")

# TEST 2: H(X1,X2) = 1.701814
print("\nTEST 2: H(X1,X2)")
mask2 = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
h2 = calculate_entropy(data, mask2)
print(f"  Calculated: {h2:.6f}")
print(f"  Expected:   1.701814")
print(f"  Difference: {abs(h2 - 1.701814):.6f}")
test2_passed = abs(h2 - 1.701814) < 0.00001
print(f"  Result:     {'PASSED' if test2_passed else 'FAILED'}")

# TEST 3: H(X1,X2,X7) = 2.348747
print("\nTEST 3: H(X1,X2,X7)")
mask3 = np.array([1, 1, 0, 0, 0, 0, 1], dtype=np.int32)
h3 = calculate_entropy(data, mask3)
print(f"  Calculated: {h3:.6f}")
print(f"  Expected:   2.348747")
print(f"  Difference: {abs(h3 - 2.348747):.6f}")
test3_passed = abs(h3 - 2.348747) < 0.00001
print(f"  Result:     {'PASSED' if test3_passed else 'FAILED'}")

# TEST 4: Empty mask
print("\nTEST 4: Empty mask")
mask_empty = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
h_empty = calculate_entropy(data, mask_empty)
print(f"  Calculated: {h_empty:.6f}")
print(f"  Expected:   0.000000")
print(f"  Difference: {abs(h_empty):.6f}")
test4_passed = abs(h_empty) < 0.00001
print(f"  Result:     {'PASSED' if test4_passed else 'FAILED'}")

print("\n" + "=" * 60)
all_passed = test1_passed and test2_passed and test3_passed and test4_passed

if all_passed:
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("Python library works CORRECTLY")
    print("Results match 1993 calculations")
    print("Library is ready for ML projects")
else:
    print("SOME TESTS FAILED")
    print("Debugging required")

print("=" * 60)