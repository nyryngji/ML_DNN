## 영화 리뷰 분류

- 데이터 : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

> **1. 데이터 불러오기**
>
    import pandas as pd 

    data = pd.read_csv('IMDB Dataset.csv')
    data.head()
<br>

> **2. 독립변수, 종속 변수 설정**
>
    x = data['review']
    y = data['sentiment']

> **3. review 데이터 전처리** 
>
    import string 
    def remove_punc(x):
        new_string = []
        for i in x:
            if i not in string.punctuation: # 특수 기호 제거
                new_string.append(i)
        new_string = ''.join(new_string)
        return new_string

    data['review'] = data['review'].apply(remove_punc)

>
    def stop_words(x):
    new_string = [i.lower() for i in x.split()] # 소문자로 문자 변경 
    new_string = ' '.join(new_string)
    return new_string
    
    data['review'] = data['review'].apply(stop_words)

<br>

> **4. sentiment 변수 전처리**
>
    data['sentiment'] = data['sentiment'].map({'positive':1, 'negative':0})
    # positive, negative를 1, 0으로 변경

<br>

> **5. 카운트 벡터화** 

- 카운트 벡터화 : 문자를 개수 기반으로 벡터화 
  
  - 모든 단어에 인덱스를 부여하고 문장마다 속한 단어가 있는 인덱스를 카운트 

>
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    cv.fit(x) # 학습하기 
    x = cv.transform(x) # review 데이터 변환

<br>

> **6. 훈련, 학습데이터로 나누기**

>
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=100)
<br>

> **7. 모델 학습**
>
    from sklearn.naive_bayes import MultinomialNB 
    from sklearn.metrics import accuracy_score

    model = MultinomialNB()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy_score(y_test, pred)
<br>

> **about model**

- 정확도 85.5%
- 학습, 훈련 데이터로 분리할 때 random_state를 적용했더니 정확도가 0.07% 증가 
