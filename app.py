import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from math import sqrt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import xgboost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score as r2
import tempfile
col1, col2, col3 = st.columns([1,6,1])
st.title('基于卷积神经网络的高温钛合金蠕变断裂寿命预测软件 V1.0')
##up load training data
st.sidebar.header('上传高温钛合金蠕变断裂寿命实验数据')
uploaded_file = st.sidebar.file_uploader("选择一个文件")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.subheader('数据展示')
    st.sidebar.write(df)
    df = df.drop(['class'], axis=1)
    x = df.iloc[:, :34]
    y = df.iloc[:, 34]
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    ## 中间面板选择模型
    st.subheader('机器学习模型')
    model_chooses = st.selectbox(
        '想要使用的机器学习模型',
        ('CNN', 'RF', 'SVR', 'GBM')
    )

    if model_chooses == 'CNN':
        uploaded_file_model_params = st.file_uploader("请选择一个CNN模型参数文件", type=["h5"])
        if uploaded_file_model_params is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(uploaded_file_model_params.read())
                tmp_file_path = tmp_file.name
            try:
                m = load_model(tmp_file_path)
                st.success("模型加载成功")
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                st.stop()
            x_data = []
            for i in range(len(x)):
                a = x[i, :]
                a = np.pad(a, (0, 2), 'constant', constant_values=(0, 0))
                a = a.reshape(6, 6, 1)
                x_data.append(a)
            x_data = np.array(x_data)

            x_train, x_test, y_train, y_test_source = train_test_split(x_data, y, train_size=0.8)

            # 加载模型并进行预测
            predict_train = m.predict(x_train).flatten()
            predict_test = m.predict(x_test).flatten()
            R2_train = r2(predict_train, y_train)
            R2_test = r2(predict_test,y_test_source)
            st.write("训练集R$^2$：",R2_train)
            st.write("测试集R$^2$：", R2_test)
            plt.figure(figsize=(8, 7))
            matplotlib.rcParams['font.family']=['Arial']
            matplotlib.rcParams['font.sans-serif']=['Arial']
            matplotlib.rcParams['font.sans-serif']=['SimHei']
            plt.title('Training Results of CNN', fontsize=24)
            plt.scatter(y_train, predict_train, label="Training", marker='*',
                        c="blue", alpha=0.7, s=60)
            plt.scatter(y_test_source, predict_test, label="Test",
                        c="green", s=60, alpha=0.9, marker='*')
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            ax.tick_params(axis='both', which='major', labelsize=22, direction='in', width=1.5, length=7)
            plt.plot([-4, 6], [-4, 6], '--')
            plt.xlim((-4, 6))
            plt.ylim((-4, 6))
            plt.ylabel('Truth', fontsize=25)
            plt.xlabel('Prediction', fontsize=25)
            plt.legend(fontsize = 18)
            st.pyplot(plt)
        else:
            st.info("请上传 CNN 模型参数文件")
        st.subheader("新成分、处理方式材料蠕变寿命预测")
        test_data_file = st.file_uploader("请选择需要测试的材料成分、工艺及测试条件文件",type = ["CSV"])
        if test_data_file is not None:
            test_data = pd.read_csv(test_data_file)
            st.write(test_data)
            test_data = test_data.drop(['class'], axis=1)
            X_test = test_data.iloc[:,:34]
            X_test_scaled = scaler.transform(X_test)
            X_test_processed = []
            for i in range(len(X_test_scaled)):
                a = X_test_scaled[i, :]
                a = np.pad(a, (0, 2), 'constant', constant_values=(0, 0))
                a = a.reshape(6, 6, 1)
                X_test_processed.append(a)
            X_test_processed = np.array(X_test_processed)
            Y_predicted = m.predict(X_test_processed).flatten()
            Y_predicted_power = 10**Y_predicted
            df = pd.DataFrame(Y_predicted_power, columns=["蠕变断裂寿命预测（小时）"])
            st.write(df)


    elif model_chooses =='RF':
        x_train, x_test, y_train, y_test_source = train_test_split(x, y, train_size=0.8)
        model = RandomForestRegressor(random_state=1)
        param_grid = {
            "max_depth":np.arange(1,11,step=1),
            "n_estimators":np.arange(10,51,step=5),
        }
        random_cv = GridSearchCV(
            model, param_grid, cv=5, scoring="r2", n_jobs=-1
        )
        random_cv.fit(x_train,y_train)
        y_hat_train = random_cv.predict(x_train)
        y_hat_test = random_cv.predict(x_test)
        R2_train = r2(y_hat_train, y_train)
        R2_test = r2(y_hat_test, y_test_source)
        st.write("训练集R$^2$：", R2_train)
        st.write("测试集R$^2$：", R2_test)
        plt.figure(figsize=(8, 7))
        matplotlib.rcParams['font.family'] = ['Arial']
        matplotlib.rcParams['font.sans-serif'] = ['Arial']
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot([-4, 6], [-4, 6],  linestyle='--')
        plt.title("Training results of RF",fontsize = 24)
        plt.scatter(y_train, y_hat_train, label="Training", marker='*',
                    c="blue", alpha=0.7, s=60)
        plt.scatter(y_test_source, y_hat_test, label="Test",
                    c="green", s=60, alpha=0.9, marker='*')
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        ax.tick_params(axis='both', which='major', labelsize=22, direction='in', width=1.5, length=7)
        plt.xlabel("True Values",fontsize=25)
        plt.ylabel("Predicted Values",fontsize=25)
        plt.legend(fontsize =18)
        st.pyplot(plt)
        st.subheader("新成分、处理方式材料蠕变寿命预测")
        test_data_file = st.file_uploader("请选择需要测试的材料成分、工艺及测试条件文件",type=["CSV"])
        if test_data_file is not None:
            test_data = pd.read_csv(test_data_file)
            st.write(test_data)
            test_data = test_data.drop(['class'], axis=1)
            X_test = test_data.iloc[:,:34]
            X_test_scaled = scaler.transform(X_test)
            Y_predicted = random_cv.predict(X_test_scaled)
            Y_predicted_power = 10**Y_predicted
            df = pd.DataFrame(Y_predicted_power, columns=["蠕变断裂寿命预测（小时）"])
            st.write(df)
    elif model_chooses == 'SVR':
        x_train, x_test, y_train, y_test_source = train_test_split(x, y, train_size=0.8)
        model = svm.SVR()
        param_grid = {
            "C": np.arange(1, 10, step=1),
            "kernel": ['rbf', 'sigmoid'],#, 'precomputed', 'sigmoid'
            "gamma": np.arange(0.01, 10, step=0.05),
        }
        random_cv = GridSearchCV(
            model, param_grid, cv=5, scoring="r2", n_jobs=-1
        )
        random_cv.fit(x_train,y_train)
        y_hat_train = random_cv.predict(x_train)
        y_hat_test = random_cv.predict(x_test)
        R2_train = r2(y_hat_train, y_train)
        R2_test = r2(y_hat_test, y_test_source)
        st.write("训练集R2：", R2_train)
        st.write("测试集R2：", R2_test)
        plt.figure(figsize=(8, 7), dpi=150)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        ax.tick_params(axis='both', which='major', labelsize=22, direction='in', width=1.5, length=7)
        plt.plot([-4, 6], [-4, 6], c='red', linestyle='--')
        plt.title("Training results of SVR",fontsize = 24)
        plt.scatter(y_train, y_hat_train, label="Training", marker='*',
                    c="blue", alpha=0.7, s=60)
        plt.scatter(y_test_source, y_hat_test, label="Test",
                    c="green", s=60, alpha=0.9, marker='*')
        plt.xlabel("True Values",fontsize=24)
        plt.ylabel("Predicted Values",fontsize=24)
        plt.legend(fontsize = 18)
        st.pyplot(plt)
        st.subheader("新成分、处理方式材料蠕变寿命预测")
        test_data_file = st.file_uploader("请选择需要测试的材料成分、工艺及测试条件文件",type=["CSV"])
        if test_data_file is not None:
            test_data = pd.read_csv(test_data_file)
            st.write(test_data)
            test_data = test_data.drop(['class'], axis=1)
            X_test = test_data.iloc[:,:34]
            X_test_scaled = scaler.transform(X_test)
            Y_predicted = random_cv.predict(X_test_scaled)
            Y_predicted_power = 10 ** Y_predicted
            df = pd.DataFrame(Y_predicted_power, columns=["蠕变断裂寿命预测（小时）"])
            st.write(df)
    elif model_chooses == 'GBM':
        x_train, x_test, y_train, y_test_source = train_test_split(x, y, train_size=0.8)
        model = GradientBoostingRegressor()
        param_grid = {
            'n_estimators':np.arange(50,301,step=50),
            'max_depth':np.arange(1,11,step=1),
        }
        random_cv = GridSearchCV(
            model, param_grid, cv=5, scoring="r2", n_jobs=-1
        )
        random_cv.fit(x_train,y_train)
        y_hat_train = random_cv.predict(x_train)
        y_hat_test = random_cv.predict(x_test)
        R2_train = r2(y_hat_train, y_train)
        R2_test = r2(y_hat_test, y_test_source)
        st.write("训练集R$^2$：", R2_train)
        st.write("测试集R$^2$：", R2_test)
        plt.figure(figsize=(8, 7), dpi=150)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        ax.tick_params(axis='both', which='major', labelsize=22, direction='in', width=1.5, length=7)
        plt.title("Traning results of GBM",fontsize = 24)
        plt.plot([-4, 6], [-4, 6], c='red', linestyle='--')
        plt.scatter(y_train, y_hat_train, label="Training", marker='*',
                    c="blue", alpha=0.7, s=60)
        plt.scatter(y_test_source, y_hat_test, label="Test",
                    c="green", s=60, alpha=0.9, marker='*')
        plt.xlabel("True Values",fontsize=24)
        plt.ylabel("Predicted Values",fontsize=24)
        plt.legend(fontsize = 18)
        st.pyplot(plt)
        st.subheader("新成分、处理方式材料蠕变寿命预测")
        test_data_file = st.file_uploader("请选择需要测试的材料成分、工艺及测试条件文件",type=["CSV"])
        if test_data_file is not None:
            test_data = pd.read_csv(test_data_file)
            st.write(test_data)
            test_data = test_data.drop(['class'], axis=1)
            X_test = test_data.iloc[:,:34]
            X_test_scaled = scaler.transform(X_test)
            Y_predicted = random_cv.predict(X_test_scaled)
            Y_predicted_power = 10**Y_predicted
            df = pd.DataFrame(Y_predicted_power, columns=["蠕变断裂寿命预测（小时）"])
            st.write(df)

else:
    st.sidebar.info('请上传数据')






