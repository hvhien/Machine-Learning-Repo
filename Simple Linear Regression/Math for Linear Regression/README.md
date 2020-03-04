# Kiến thức toán về Linear Regression

## Các kiến thức cần phải học:
  * Đại số tuyến tính và cấu trúc đại số
  * Toán cơ bản bao gồm đạo hàm, tích phân, vi phân
## Công thức cơ bản cân nắm:
  ### Dạng của Linear Regression:
![](https://github.com/tandathcmute/MLrepo/blob/master/Simple%20Linear%20Regression/Math%20for%20Linear%20Regression/hthetax.PNG)

Trong đó x1,x2,...,xn là các input. 

Các theta là các parameter để điều chỉnh hàm số sao cho phù hợp với trainning set.

Tuy nhiên, nếu dùng công thức này rất là lâu khi ta phải nhân từng phần tử sau đó cộng lại. Lúc này ta sẽ dùng đại số tuyến tính, cụ thể là phép nhân hai ma trận.

Ta sẽ có X là ma trận gồm các input. Và nó là một vector ngang 1 * n. Theta là một vector chứa tất cả các theta

Để nhân hai ma trận này ta phải có số cột của X bằng với số hàng của theta. Tuy nhiên sẽ xảy ra trường hợp là θo không có x0 tương ứng. Để giải quyết trường hợp này ta sẽ thêm một cột có giá trị luôn băng 1 ở đầu cột X
```python
    X = np.copy(raw)
    X[:,1] = X[:,0]
```
và ta sẽ có cột y là chứa các output trong bộ data của chúng ta.

Để dự đoán được ta chỉ cần thực hiện phép toán nhân hai ma trận X và theta bằng toán tử ***@***
```python
    def predict(X,theta):
        return X@theta
```

Mà nếu ta để ý, ta sẽ không có bộ theta để chúng ta thực hiện quá trình này. Chúng ta sẽ tìm nó ngay sau đây.
  ### Hàm mất mát J(θ)
![](https://github.com/tandathcmute/MLrepo/blob/master/Simple%20Linear%20Regression/Math%20for%20Linear%20Regression/jtheta.PNG)

J(θ) giúp chúng ta tìm ra độ chính xác của kết quả. Khi chúng ta có hàm hθ(x) chính xác thì hàm J(θ) sẽ bằng 0. Và mô phỏng đơn giản hàm mất mát sẽ như thế này:

![Đồ thị đơn giản của hàm J(θ)](https://github.com/tandathcmute/MLrepo/blob/master/Simple%20Linear%20Regression/Math%20for%20Linear%20Regression/DothiJtheta.jpg)

Khi chính xác, điểm dừng của chúng ta sẽ chạm điểm thấp nhất của đồ thị.

Để áp dụng công thức vào python ta sẽ có cú pháp như thế này:
```python
    def computeCost(X,y,Theta):
	sqr_error = (predicted – y)**2
	sum_error = np.sum(sqr_error)
	m = np.size(y)
	J = (1/(2*m))*sum_error
	return J
```
Hàm mất mát chỉ để tìm được θ nhưng không thể tìm được bộ θ tốt nhất. Nó sinh ra chỉ để tìm theta nhưng để tìm được θ tốt nhất đó chính là việc của ***Gradient Descent***
  ### Gradient Descent
![](https://github.com/tandathcmute/MLrepo/blob/master/Simple%20Linear%20Regression/Math%20for%20Linear%20Regression/congthucgradientdescent.PNG)

Gradient Descent đơn giản là từng bước bước từ điểm cao nhất của đồ thị xuống tới điểm thấp nhất. Và khi gặp điểm thấp nhất nó sẽ dừng lại.

```python
def gradientdescent(X,y,alpha=0.02,iter=5000):
    theta = np.zeros(np.size(X,1))
    m = np.size(y)
    X_T = np.transpose(X)
    precost = computeCost(X,y,theta)
    for i in range(0,iter):
        error = predict(X,theta) - y
        theta = theta - (alpha/m) * (X_T@error)
        cost = computeCost(X,y,theta)
        if np.round(cost,30) == np.round(precost,30):
            break
        precost = cost
    yield theta

```

Trong đó: 
  * *alpha* bước chạy mặc định là 0.02. Tuy nhiên khi ta tăng lên có thể xáy ra trường hợp nó bước qua điểm lõm còn khi giảm thấp nó sẽ đi rất chậm dẫn đến tốn thời gian.
  * *inter* là số lần lặp. Chúng ta có thể điều chỉnh được số lần lặp này.
  * *theta* chính là ma trận cần tìm, và nó phải bằng đúng số cột của X
  * *precost* để kiểm tra giá trị của điểm đang xét lúc đầu
  * *cost* để kiểm tra giá trị của điểm đang xét lúc sau
Để hàm có thể tìm đúng vị trí điểm lõm của đồ thị thì hai giá trị *precost* và *cost* sẽ gần bằng nhau. Và dựa theo trên mình sẽ lấy 30 chứ số thập phân. Nếu hai giá trị này bằng nhau thì sẽ ngắt vòng lặp và cập nhật lại precost. Khi ngắt giá trị *theta* tốt nhất sẽ được cập nhật

```python
    error = predict(X,theta) - y
    theta = theta - (alpha/m) * (X_T@error)
```

Hàm trên chính là mô phỏn lại công thức của ***Gradient Descent***

Để lấy giá trị của bộ θ ta sẽ viết:
```python
    [Theta] = gradientdescent(X,y)
```

## Thực hiện quá trình tìm hàm số của Linear Regression:

Theo công thức tìm hθ(x), ***Linear Regression*** chỉ đơn giản là nhân hai ma trận X và Theta sau khi tìm được.
```python
    predicted = X @ Theta
```

## Trực quan hoá đồ thị để dự đoán kết quả:
```python
    plt.plot(X[:,1:],y,'rx')
plt.plot(X[:,1:],predicted,'b')
plt.show()
```
![](https://github.com/tandathcmute/MLrepo/blob/master/Simple%20Linear%20Regression/Math%20for%20Linear%20Regression/dothiketqua.png)

## Một vài lưu ý:
Khi ta dùng file [univariate_theta.txt](https://github.com/tandathcmute/MLrepo/blob/master/Simple%20Linear%20Regression/Math%20for%20Linear%20Regression/univariate_theta.txt) ta vẫn thu được đồ thị tương xứng với nhau trong khi giá trị *Theta* trong file này khác xa với *Theta* mà ta tìm được.

Viết bởi [Trịnh Tấn Đạt](https://www.facebook.com/ttd.lvc)
## Nguồn: 
[Howkteam](https://www.howkteam.vn/)

[Machine Learning cơ bản](https://machinelearningcoban.com/2016/12/28/linearregression/)
