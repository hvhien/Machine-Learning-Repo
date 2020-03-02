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

  **Để tính toán dự đoán kết quả mô hình hồi quy tuyến tính, ta có thể import mô hình hồi quy tuyến tính đơn giản như sau**

```python
from sklearn.linear_model import LinearRegression
```
  **Tuy nhiên, mô hình hồi quy tuyến tính tổng quát hơn có tồn tại, ta có thể đọc [sklearn.linear_model module documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)**

### 3. Chọn model hyperparameter
  #### Ta cần phân biệt giữa Model Parameter và Model Hyperparameter**

**Model Parameter là các giá trị của model được sinh ra từ dữ liệu huấn luyện giúp thể hiện mối liên hệ giữa các đại lượng trong dữ liệu. Như vậy khi chúng ta nói tìm được mô hình tốt nhất cho bài toán thì nên ngầm hiểu rằng chúng ta đã tìm ra được các Model parameter phù hợp nhất cho bài toán trên tập dữ liệu hiện có.**
**Chúng sẽ có một số đặc điểm nhận dạng sau:**
  * Dùng để dự đoán đối với một tệp dữ liệu mới.
  * Thể hiện sức mạnh của mô hình đang được sử dụng thông qua tỷ lệ accuracy 
  * Học trực tiếp từ tập dữ liệu huấn luyện
  * Thường không được tạo thủ công

```

```
