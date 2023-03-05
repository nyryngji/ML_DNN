## Heart Disease Prediction - 심장병 예측

> **1. 데이터 불러오기**
>
    import pandas as pd 
    data = pd.read_csv('framingham.csv')

<br>

> **2. 결측값 제거**
>
    data.info() # 데이터에 결측값이 있는 것을 확인
    data = data.dropna(axis=1) # 결측값이 있는 열 제거
    data.head()    

<br>

> **3. 히트맵으로 변수 간 상관관계 파악**
>
    import matplotlib.pyplot as plt 
    import seaborn as sns

    sns.heatmap(data.corr(), cmap='Blues',vmin=-1, vmax=1, annot=True)
    plt.show()

<br>

> **4. 피처 엔지니어링**

- 상관관계가 높은 변수끼리 묶어 새로운 변수 만들기
- 정확도를 0.8455 -> 0.8496으로 올리는데 성공 

>
    data['BP'] = data['sysBP'] + data['diaBP']
    data.drop(['sysBP', 'diaBP'], axis=1, inplace=True)

<br>

>
    sns.heatmap(data.corr(), cmap='Blues',vmin=-1, vmax=1, annot=True)
    plt.show()
    data['preBP'] = data['BP'] + data['prevalentHyp']
    data.drop(['BP','prevalentHyp'], axis=1, inplace=True)
    
<br>

> **5. 독립변수, 종속변수 분리하기**
>
    from sklearn.model_selection import train_test_split

    X = data[['male','age','currentSmoker','prevalentStroke','diabetes','preBP']]
    Y = data['TenYearCHD']
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4,random_state=100)

<br>

> **6. 모델 학습**
>
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

<br>

> **about model**

- 정확도 85%
- MinMaxScaler로 스케일링을 시도하였으나 정확도에 영향 x 
