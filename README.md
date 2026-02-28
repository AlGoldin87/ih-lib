# ih-lib

Python библиотека для энтропийного анализа данных.

## Установка

```bash
pip install git+https://github.com/ВАШ-ЛОГИН/ih-lib.git
```

## Использование

```python
import numpy as np
from ih import calculate_entropy

data = np.array([[0,0,1], [1,0,1]], dtype=np.int32)
mask = np.array([1,0,1], dtype=np.int32)
h = calculate_entropy(data, mask)
print(h)
```

## Примеры

См. папку [`examples/`](examples/).

## Лицензия

MIT