# ML Fitting App

![image](https://github.com/taturabe/ml_fitting_app/blob/master/ml_fitting_app.png)

機械学習の基本である線形多項式回帰を体験するアプリです。

WebアプリのPythonのstreamlitライブラリを使用しています。

## 使用ライブラリ

- matplotlib==3.1.3
- numpy==1.18.1
- streamlit==0.63.0

## 使用方法

```
streamlit run polynomial.py
```

Best fitボタンがオフの時は、スライダーが現れ手動で多項式モデルを動かすことができます。

Best fitボタンがオン（チェック）の時は、自乗誤差を最小化するように最適化を行います。
