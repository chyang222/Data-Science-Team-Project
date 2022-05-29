import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV


def exploation(df):
    print(df.info())
    print("-"*10, "data basic statistics", "-"*10)
    print(df.describe())


def nul(df):
    print(df.isna().sum())
    df.isna().mean().sort_values(ascending=False).plot( kind='bar', figsize=(15,5),  grid=True, color='blue', edgecolor='black', linewidth=2, rot=42)
    plt.title('NaN')
    plt.xlabel('Name of column')
    plt.ylabel('Frac of Nan')
    plt.show()


def go_pie(df, columns):
    df = df[columns].value_counts()
    trace=go.Pie(labels=df.index,
            values=df.values,
            hole=0.3)

    layout=go.Layout(title='Fraction of games of each {}'.format(columns))

    fig=go.Figure(data=[trace],layout=layout)
    iplot(fig)

def go_plot(df1, df2, df3, df4, df5):
    trace1 = go.Scatter(x=df1.index, y=df1.values, mode='lines+markers', name='action')
    trace2 = go.Scatter(x=df2.index, y=df2.values, mode='lines+markers', name='sports')
    trace3 = go.Scatter(x=df3.index, y=df3.values, mode='lines+markers', name='Role-Playing')
    trace4 = go.Scatter(x=df4.index, y=df4.values, mode='lines+markers', name='Shooter')
    trace5 = go.Scatter(x=df5.index, y=df5.values, mode='lines+markers', name='Platform')

    layout = go.Layout(title='Number of games of top 5 Genre released each year', xaxis=dict(title='Year'), yaxis=dict(title='Number of games'))

    fig = go.Figure(data=[trace1,trace2,trace3,trace4,trace5],layout=layout)

    iplot(fig)

def rank_plot(df, column1, column2, rank):
    top_10_games_by_gs = df[[column1, column2]].groupby([column1],as_index=False).mean().sort_values(by=column2, ascending=False).round(2).head(rank)
    plt.figure(figsize=(10, 7))        
 
    ax = sns.barplot(x=column2, y=column1, data=top_10_games_by_gs)
    ax.set_xlabel('Global Sales in Millions')
    ax.set_ylabel('Game')

    for index, value in enumerate(top_10_games_by_gs[column2]):
        plt.text(value, index, str(value))
    plt.title(column2)
    plt.show()

def compare_plot(df, column1, column2, column3, column4, column5):
    comp_genre = df[[column1, column2, column3, column4, column5]]

    comp_map = comp_genre.groupby(by=[column1]).sum()
    comp_table = comp_map.reset_index()
    comp_table = pd.melt(comp_table, id_vars=[column1], value_vars=[column2, column3, column4, column5], var_name='Sale_Area', value_name='Sale_Reveneu')


    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x=column1, y="Sale_Reveneu", hue='Sale_Area', data=comp_table)
    ax.set_ylabel('Total Sales in Millions')
    ax.set_xlabel(column1)
    plt.show()

def standard_scale(df, column1, column2):
    Global_Sales=pd.DataFrame(df,columns=[column1])
    other_Sales=pd.DataFrame(df,columns=[column2])
    standard = StandardScaler() #Global sales_ Standard scaler
    Sscale_Global_Sales = standard.fit_transform(Global_Sales)
    Oother_scale_scale = standard.fit_transform(other_Sales)
    return Sscale_Global_Sales, Oother_scale_scale


def minmax_scale(df, column1, column2):
    Global_Sales=pd.DataFrame(df,columns=[column1])
    other_Sales=pd.DataFrame(df,columns=[column2])
    minmax = MinMaxScaler() #Global sales_ MinMax scaler
    Mscale_Global_Sales = minmax.fit_transform(Global_Sales)
    Oother_minmax_scale = minmax.fit_transform(other_Sales)
    return Mscale_Global_Sales, Oother_minmax_scale

def robust_scale(df, column1, column2):
    Global_Sales=pd.DataFrame(df,columns=[column1])
    other_Sales=pd.DataFrame(df,columns=[column2])
    minmax = RobustScaler() #Global sales_ MinMax scaler
    Rscale_Global_Sales = minmax.fit_transform(Global_Sales)
    Oother_robust_scale = minmax.fit_transform(other_Sales)
    return Rscale_Global_Sales, Oother_robust_scale

def line_reg(df_train, column1, column2, column3, column4, column5):
    X_na = df_train[[column1]]
    X_eu = df_train[[column2]]
    X_jp = df_train[[column3]]
    X_other = df_train[[column4]]
    y = df_train[column5]

    line_fitter_na = LinearRegression().fit(X_na, y)
    line_fitter_eu = LinearRegression().fit(X_eu, y)
    line_fitter_jp = LinearRegression().fit(X_jp, y)
    line_fitter_other = LinearRegression().fit(X_other, y)

    train_predict_na = line_fitter_na.predict(X_na)
    train_predict_eu = line_fitter_eu.predict(X_eu)
    train_predict_jp = line_fitter_jp.predict(X_jp)
    train_predict_other = line_fitter_other.predict(X_other)

    return train_predict_na, train_predict_eu, train_predict_jp, train_predict_other

def mse(df_train, column1, column2, column3, column4, column5):
    X_na = df_train[[column1]]
    X_eu = df_train[[column2]]
    X_jp = df_train[[column3]]
    X_other = df_train[[column4]]
    y = df_train[column5]

    line_fitter_na = LinearRegression().fit(X_na, y)
    line_fitter_eu = LinearRegression().fit(X_eu, y)
    line_fitter_jp = LinearRegression().fit(X_jp, y)
    line_fitter_other = LinearRegression().fit(X_other, y)

    train_predict_na, train_predict_eu, train_predict_jp, train_predict_other = line_reg(df_train, column1, column2, column3, column4, column5)
    test_predict_na = line_fitter_na.predict(df_train[[column1]])
    test_predict_eu = line_fitter_eu.predict(df_train[[column2]])
    test_predict_jp = line_fitter_jp.predict(df_train[[column3]])
    test_predict_other = line_fitter_other.predict(df_train[[column4]])
    y_test = df_train[column5]

    print("----------------MSE of train set----------------")
    print("Global sales by NA_Sales :", mean_squared_error(y, train_predict_na))
    print("Global sales by EU_Sales :", mean_squared_error(y, train_predict_eu))
    print("Global sales by JP_Sales :", mean_squared_error(y, train_predict_jp))
    print("Global sales by Other_Sales :", mean_squared_error(y, train_predict_other))

    print("----------------MSE of test set----------------")
    print("Global sales by NA_Sales :", mean_squared_error(y_test, test_predict_na))
    print("Global sales by EU_Sales :", mean_squared_error(y_test, test_predict_eu))
    print("Global sales by JP_Sales :", mean_squared_error(y_test, test_predict_jp))
    print("Global sales by Other_Sales :", mean_squared_error(y_test, test_predict_other))


def stand_line(X_train, X_test, y_train, y_test):
    standard_scaler1 = StandardScaler().fit(X_train)
    standard_scaler2 = StandardScaler().fit(X_test)
    X_train = standard_scaler1.transform(X_train)
    X_test = standard_scaler2.transform(X_test)

    stdScaler = LinearRegression().fit(X_train, y_train)
    y_pred = stdScaler.predict(X_test)

    print("MSE of Standard scaled test set is ", mean_squared_error(y_test, y_pred))

def minmax_line(X_train, X_test, y_train, y_test):
    mm_scaler1 = MinMaxScaler().fit(X_train)
    mm_scaler2 = MinMaxScaler().fit(X_test)
    X_train = mm_scaler1.transform(X_train)
    X_test = mm_scaler2.transform(X_test)

    mmScaler = LinearRegression().fit(X_train, y_train)
    y_pred = mmScaler.predict(X_test)

    print("MSE of MinMax scaled test set is ", mean_squared_error(y_test, y_pred))

def robust_line(X_train, X_test, y_train, y_test):
    r_scaler1 = RobustScaler().fit(X_train)
    r_scaler2 = RobustScaler().fit(X_test)
    X_train = r_scaler1.transform(X_train)
    X_test = r_scaler2.transform(X_test)

    rScaler = LinearRegression().fit(X_train, y_train)
    y_pred = rScaler.predict(X_test)

    print("MSE of Robust scaled test set is ", mean_squared_error(y_test, y_pred))

def line_kfold(df_train, column1, column2):
    x = df_train[[column1]]
    y = df_train[column2]
    model = LinearRegression()

    scores = cross_val_score(model, x, y, cv=KFold(n_splits=5), scoring='r2')

    print('5-fold cv score: ', scores)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts = True)
    entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name,target_name): 
    # 전체 엔트로피 계산
    total_entropy = entropy(data[target_name])
    print('Entropy(D) = ', round(total_entropy, 5))
    
    # 가중 엔트로피 계산
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*
                               entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    print('H(', split_attribute_name, ') = ', round(Weighted_Entropy, 5))
 
    
    # 정보이득 계산
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain
 
def ID3(data,originaldata,features,target_attribute_name,parent_node_class = None):
    # 중지기준 정의
    # 1. 대상 속성이 단일값을 가지면: 해당 대상 속성 반환
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
 
    # 2. 데이터가 없을 때: 원본 데이터에서 최대값을 가지는 대상 속성 반환
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])\
               [np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
 
    # 3. 기술 속성이 없을 때: 부모 노드의 대상 속성 반환
    elif len(features) ==0:
        return parent_node_class

    # 트리 성장
    else:
        # 부모노드의 대상 속성 정의(예: Good)
        parent_node_class = np.unique(data[target_attribute_name])\
                            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # 데이터를 분할할 속성 선택
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # 트리 구조 생성
        tree = {best_feature:{}}
        
        # 최대 정보이득을 보인 기술 속성 제외
        features = [i for i in features if i != best_feature]
        
        # 가지 성장
        for value in np.unique(data[best_feature]):
            # 데이터 분할. dropna(): 결측값을 가진 행, 열 제거
            sub_data = data.where(data[best_feature] == value).dropna()
            
            # ID3 알고리즘
            subtree = ID3(sub_data,data,features,target_attribute_name,parent_node_class)
            tree[best_feature][value] = subtree
            
        return(tree)