# Multiple Linear Regression
Một chút thông tin về thuật toán **Linear Regression** ta xem ở [đây](https://github.com/tandathcmute/MLrepo/tree/master/Simple%20Linear%20Regression/Math%20for%20Linear%20Regression)
## Các bước thực hiện:

### 1. Import library and dataset
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
```
### 2. Encoding the independent variable
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
```
Dựa vào cột ***State***  ta có thể nhận thấy có tất cả 3 thành phố riêng biệt. Dùng **LabelEncoder** giúp chúng ta ánh xạ từng danh mục đó thành các cột được đánh số từ 0 tới 2.
**One Hot Encoding Scheme** giúp biến đổi các giá trị đặc thành các đặc trưng nhị phân chỉ chứa giá trị 0 và 1 với giá trị 1 là giá trị active

### 3. Avoid the dummy variable trap
```python
X = X[:,1:]
```
Bẫy này xảy ra khi các biến Dummy được sử dụng để làm biến giải thích. Khi ta đưa tất cả các biến Dummy vào model thì sẽ xảy ra hiện tượng ***đa cộng tuyến hoàn hảo*** giữa chúng và intercept_ (hay còn gọi là sai số).

### 4. Split the dataset into Training set and Test set:
Ta sẽ chia bộ data với 20% cho ***Test set*** và 80% cho ***Training set***
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
```
### 5. Fit Training set to model
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
### 6. Building the optimal model using Backward Elimination.
Đối với Linear Regression trong Multivariate. Có thể tất cả các cột thành phần trong data sẽ có quan hệ hồi quy với cột output nhưng phần lớn các cột trong bộ data sẽ không có quan hệ hồi quy với cột output. Để tìm được các cột thành phần ta sử dụng phương pháp Backward Elimination.

Để làm được ta cần phải tìm Ordinary Least Square (OSL) hay bình phương tối thiểu. Ta có thể xem ở [đây](https://vi.wikipedia.org/wiki/B%C3%ACnh_ph%C6%B0%C6%A1ng_t%E1%BB%91i_thi%E1%BB%83u)

Tuy nhiên giả định OLS với model hồi quy đơn giản, Linear Regression trong Multivariate bổ sung thêm một điều kiện: không có ***đa cộng biến hoàn hảo***

Khi ta nhìn biểu thức Linear Regression trong Multivariate thì θo sẽ không có giá trị x0 tương ứng do đó ta sẽ thêm 1 cột bias có giá trị một và có số dòng bằng đúng số dòng của cột X
```python
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
```

Để thực hiện quá trình này ta sử dụng thư viện **statsmodels.api**
```python
import statsmodels.api as sm
```
Backward Elimination được thực hiện theo các bước:
  * Chọn một hạng mức để giữ lại cho model. Thông thường ta sẽ chon SL = 0.05
  * Fit các giá trị có thể dự đoán vào model
  * So sánh với giá trị P_values
    * Nếu P_values < SL thì chúng ta sẽ loại bỏ giá trị dự đoán đó
      * Fit lại model sau khi loại bỏ giá trị dự đoán.
    * Nếu P_values > SL ta sẽ fit model trực tiếp mà không loại bỏ giá trị dự đoán nào nữa.
 
Vậy giá trị P_values là gì. Ta có thể tham khảo video [này](https://www.youtube.com/watch?v=eyknGvncKLw)

Và khi giá trị P_values > SL với SL = 0.05 như ảnh 

![](https://github.com/tandathcmute/MLrepo/blob/master/Multiple%20Linear%20Regression/Capture.PNG)

Ta sẽ bỏ giá trị P_values lớn nhất đó chính là 0.99. Ta sẽ loại bỏ tới khi không còn giá trị P_values > SL nữa.

![](https://github.com/tandathcmute/MLrepo/blob/master/Multiple%20Linear%20Regression/Capture2.PNG)

Giá trị P_values có thể nhìn thấy nó bằng 0 nhưng giá trị P_values thật sự **không bao giờ bằng 0**

Sau khi loại bỏ hết bộ data của chúng ta sẽ như thế này:

![](https://github.com/tandathcmute/MLrepo/blob/master/Multiple%20Linear%20Regression/after.png)

Bài toán sẽ đưa về Simple Linear Regression. Và ta sẽ visualization bằng matplotlib đơn thuần. 

`Lưu ý: ` Ta không thể trực quan hoá để dự đoán trong multivariable vì ta sẽ có nhiều hồi quy hơn trong một biểu đồ. Bằng cách đưa về ***Simple Linear Regression*** ta sẽ dễ dàng dự đoán kết quả hơn.

### 7. Đưa bài toán về Simple Linear Regression.
Để đơn giản ta sẽ tạo một bản sao data khác và xoá đi các cột không cần thiết. Sau đó trực quan hoá và dự đoán kết quả như ***Simple Linear Regression***
```python
import numpy as nm  
import matplotlib.pyplot as mpl  
import pandas as pd

data_set_after = pd.read_csv('50_Startups_after.csv')
X_after = data_set_after.iloc[:,:-1].values
y_after = data_set_after.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_after_train, X_after_test, y_after_train,y_after_test = train_test_split(X_after, y_after, test_size = 0.02,
                                                                           random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_after_train, y_after_train)

mpl.scatter(X_after_train,y_after_train)
mpl.plot(X_after_train, lin_reg.predict(X_after_train))
mpl.xlabel("R&D Spend")
mpl.ylabel("Profit")
mpl.title("R&D Spend vs Profit (Training set)", color = 'darkred')
mpl.show()
```

Và đây là kết quả

![](https://github.com/tandathcmute/MLrepo/blob/master/Multiple%20Linear%20Regression/Figure_1.png)

[Full source code ở đây](https://github.com/tandathcmute/MLrepo/blob/master/Multiple%20Linear%20Regression/test.py)

Viết bởi [Trịnh Tấn Đạt](https://www.facebook.com/ttd.lvc)

[Scikit-learn](https://scikit-learn.org)

[Statsmodels](https://www.statsmodels.org/stable/index.html)
