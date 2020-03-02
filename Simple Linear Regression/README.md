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
  #### Ta cần phân biệt giữa Model Parameter và Model Hyperparameter

**Model Parameter là các giá trị của model được sinh ra từ dữ liệu huấn luyện giúp thể hiện mối liên hệ giữa các đại lượng trong dữ liệu. Như vậy khi chúng ta nói tìm được mô hình tốt nhất cho bài toán thì nên ngầm hiểu rằng chúng ta đã tìm ra được các Model parameter phù hợp nhất cho bài toán trên tập dữ liệu hiện có. Chúng sẽ có một số đặc điểm nhận dạng sau:**
  * Dùng để dự đoán đối với một tệp dữ liệu mới.
  * Thể hiện sức mạnh của mô hình đang được sử dụng thông qua tỷ lệ accuracy 
  * Học trực tiếp từ tập dữ liệu huấn luyện
  * Thường không được tạo thủ công
  
**Model Hyperparamter hoàn toàn nằm ngoài mô hình và không phụ thuộc vào tệp huấn luyện. Chúng sẽ có một số nhiệm vụ sau:**
  * Được dùng để huấn luyện và tìm ra parameter hợp lý nhất
  * Nó được tạo thủ công bởi con người trong việc huấn luyện
  * Nó được định nghĩa dựa trên **Heuristics**
  
  #### Chúng ta khởi tạo *LinearRegression class* và chỉ định phù hợp với intercept bằng cú pháp *fit_intercept*
  ```python
      model = LinearRegression(fit_intercept = True)
  ```
  **Lưu ý: Khi khởi tạo mô hình, chỉ có một hành động duy nhất là lưu trữ những giá trị của hyperparameter. Cụ thể, ta không thể áp dụng model cho bất kì data nào. Scikit-learn API phân biệt rõ ràng giữa lựa chọn mô hình và áp dụng mô hình vào data.**

### 4. Sắp xếp dữ liệu thành ma trận tính năng và vector target:
  **Scikit learn mô tả data là một ma trận hai chiều và mảng một chiều. Để sắp xếp lại dữ liệu ta cần phải thay đổi ma trận thành mảng một chiều bằng cú pháp:**
```python
      X = x[:, np.newaxis]
```
**Và nó dùng để fit model vào bộ data của chúng ta.**

### 5.Fit mô hình vào bộ data:
  **Để áp dụng mô hình vào data t sẽ dung *fit()* method của model:**
```python
model.fit(X,y)
```
**Trong quá trình fit model, ta luôn nhận được về hai đại lượng đó chính là hệ số hồi quy *coef_* (hay còn gọi là độ dốc) và sai số *intercept_*.**
```python
      model.coef_
      model.intercept_
```
### 6. Dự đoán nhãn cho những dữ liệu chưa biết (dự đoán kết quả).
  **Để dự đoán nhãn cho những dữ liệu chưa biết, ta sẽ xây dựng training set. Trong scikit learn, ta dùng *predict()* method để dự đoán.**
```python
      xfit = np.linspace(-1,11)
```
**Và trước hết chúng ta phải sắp xếp lại dữ liệu trong *xfit* giống như với bộ data *X***
```python
      Xfit = xfit[:, np.newaxis]
```
**Tiếp theo ta sẽ sử dụng model vừa tạo để tiến hành dự đoán.**
```python
      yfit = model.predict(Xfit)
```

### 6. Trực quan hoá mô hình.
  **Ta sẽ trực quan hoá kết quả bao gồm bộ data đầu và phương trình hồi quy của mô hình.**
```python
      plt.scatter(x,y)
      plt.plot(xfit,yfit)
```
### [Full source code ở đây](https://github.com/tandathcmute/MLrepo/blob/master/Simple%20Linear%20Regression/demoSimpleLinearRegression.ipynb)
### Xong. :)))
Viết bởi [Trịnh Tấn Đạt](https://www.facebook.com/ttd.lvc)
### Nguồn tham khảo:
[Scikit-learn](https://scikit-learn.org)
[Python Data Science Handbook O'REILLY](https://libgen.is/book/index.php?md5=B72D6570421B823BA68C6D4B2F7BF2A4)
  

