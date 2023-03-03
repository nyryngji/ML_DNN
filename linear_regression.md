## 경력으로 연봉 예측하기 

- data : https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression
<br>

> **1. 데이터 불러오기**
>
    import pandas as pd
    data = pd.read_csv('Salary_dataset.csv')

<br>

> **2. 독립변수와 종속변수 설정**
- 머신러닝 : 독립 변수(경력)로 종속 변수(연봉) 예측

>
    from sklearn.model_selection import train_test_split
    X = data[['YearsExperience']].to_numpy() # 독립변수 
    Y = data['Salary'].to_numpy() # 종속변수
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

<br>

> **3. 모델 학습**

>
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
<br>

> **4. 모델 평가**
>
    model.score(X,Y)

<br>

> **about model**
- 정확도 95%
