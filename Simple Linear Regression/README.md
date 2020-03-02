# SIMPLE LINEAR REGRESSION

## Các bước thực hiện:

### 1. Khởi tạo bộ data để mô phỏng thuật toán hồi quy tuyến tính
```python
      import numpy as np
      import matplotlib.pyplot as plt
      rng = np.random.RandomState(42)
      x = 10 * rng.rand(50)
      y = 2 * x -1 + rng.randn(50)
```
### 2. Chọn class của model

  **Để tính toán một hàm hồi quy cơ bản ta nạp vào thư viện hồi quy**

```python
      from sklearn.linear_model import LinearRegression
```

