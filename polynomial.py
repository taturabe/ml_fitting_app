import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)






N = 15

st.title("ML Fitting App")
st.subheader("多項式回帰モデル")
st.latex(r'''
    y = a_0  + a_1 x + a_2 x^2 + a_3 x^3 ... = \sum_{k=0}^{m}a_kx^k
    ''')
st.subheader("")
st.subheader("")



num = st.number_input("次数", min_value=1, max_value=30, step=1)

fit_box = st.checkbox('Best fit (red line will appear)')

if not fit_box:
    st.write("調整したい重み係数")
    slider_list = [st.slider('a0', -5.0, 5.0, 0.0, step=0.01)]
    for i in range(num):
        key = 'a%d' % (i+1)
        slider_list.append(st.slider(key, -10.0, 10.0, 1.0, step=0.01))

a = 1.5 # ground truth
b = 2   # ground truth
s = 20   # marker size
lowend = 0
highend = 10
sigma = 0.8

# sample plot
np.random.seed(seed=123)
X = np.random.uniform(low=lowend, high=highend, size=N)
noise = np.random.normal(loc=0, scale=sigma, size=N)
Y = 10*np.sin(X/((highend-lowend)*3.5)*2*np.pi) + b + noise  # sampler
plt.scatter(X,Y, s=s)

@st.cache(suppress_st_warning=True)
def multi_nominal(coef_list, X):
    val = 0
    for i,c in enumerate(coef_list):
        val += c * X**i
    return val


@st.cache(suppress_st_warning=True)
def fit(X, Y, m):
    assert len(X) == len(Y)
    normal = np.ones((len(X), m+1))
    for i in range(1,m+1):
        normal[:,i] = X**(i)
    #w = LA.inv((normal.T).dot(normal)).dot(normal.T).dot(Y)
    w = LA.solve((normal.T).dot(normal), (normal.T).dot(Y))
    return w
   

xx = np.linspace(0, 10, 200)

if fit_box: # use best fit
    w_hat = fit(X,Y,num)
    fit_line = multi_nominal(w_hat, xx)
    plt.plot(xx, fit_line, c='red')
    Y_pred = multi_nominal(w_hat, X)
else:  # use slider_bar value
    fit_line = multi_nominal(slider_list, xx)
    plt.plot(xx, fit_line, c="deeppink")
    Y_pred = multi_nominal(slider_list, X)
    
# calculate RMSE
rmse = np.sqrt(sum((Y - Y_pred)**2)/N)

plt.title("m=%d    RMSE=%.4f" % (num, rmse))
plt.xlim(lowend, highend)
plt.ylim(0,16)



st.pyplot()
