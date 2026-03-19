############################数据导入
import pandas as pd

train0 = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test0 = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
train0.drop('Id',axis=1,inplace=True)
test0.drop('Id',axis=1,inplace=True)
sum = pd.concat([train0,test0],ignore_index=True)
############################字符转数字
import numpy as np

print(sum.dtypes)
unique0 = set()
for column in sum.select_dtypes(include=['object']).columns:
    unique0.update(sum[column].unique())
print(unique0)
unique0 = list(unique0)
list0 = list(range(1,len(unique0)+1))
unique0_dict = dict(zip(unique0,list0))
print(unique0_dict)##########这个地方写论文的时候要用
def str_num(data):
    return unique0_dict.get(data,np.nan)
for column in sum.select_dtypes(include=['object']).columns:
    sum[column] = sum[column].apply(str_num)
print(sum)

# ##############################缺失值 用每列的缺失值补充缺失值
sum = sum.fillna(sum.median())
#############################标准化
from sklearn.preprocessing import StandardScaler

xsum = sum.values
ysum = np.array(sum.loc[:,'MSSubClass'])
xsum = np.delete(xsum,0,axis=1)
xsum = StandardScaler().fit_transform(xsum)
xtrain = xsum[:1460,:]
xtest = xsum[1460:,:]
ytrain = ysum[:1460]
ytest = ysum[1460:]



########################################################决策树
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt

test = []
for i in range (20):
    clf = tree.DecisionTreeClassifier(max_depth=i+1
                                      ,criterion='entropy'
                                      ,random_state=30)
    clf = clf.fit(xtrain,ytrain)
    score = clf.score(xtest,ytest)
    test.append(score)
plt.plot(range(1,21),test,color='red',label='max_depth')
plt.legend()
plt.show()
print(test.index(max(test)) , max(test))
clf = tree.DecisionTreeClassifier(max_depth=5,criterion='entropy',random_state=30)
clf = clf.fit(xtrain,ytrain)
print(clf.predict(xtest))

print(sum.columns)
targets = set()
targets.update(sum['MSSubClass'].unique())
print(targets)

feature_name = ['一般分区分类', '与物业的街道的线性英尺', '地段面积', '街道', '胡同',
       '属性的一般形状', '房产的平坦度', '实用程序', '批次配置', '房产坡度',
       '邻里', '靠近主要公路或铁路', '靠近主要公路或铁路2', '住宅类型', '住宅风格',
       '整体材料和表面处理质量', '整体状况品级', '建造年份', '改造日期', '房顶类型',
       '房顶材料', '外部覆盖物', '外部覆盖物2', '砌体单板类型', '砌体单板面积',
       '外部材料质量', '外部材料的现状', '基金会', '地下室高度', '地下室的一般状况',
       '地下室墙壁', '地下室完工面积质量', '类型1成品平方英尺', '第二个成品区域的质量',
       '类型2成品平方英尺', '地下室面积未完工平方英尺', '地下室总面积平方英尺', '加热类型', '加热质量和条件',
       '中央空调', '电气', '一楼平方英尺', '二楼平方英尺', '低质量成品面积',
       '地上居住面积', '地下室全套浴室', '地下室半套浴室', '地上浴室', '地上半浴室',
       '卧室数量', '厨房数量', '厨房质量', '地上客房总数',
       '功能', '壁炉数量', '壁炉质量', '车库位置', '车库建成年份',
       '车库内部装饰', '车位大小', '车库面积', '车库质量', '车库状况',
       '铺砌车道', '木甲板面积', '开放式门廊面积', '封闭式门廊面积', '三季门廊面积',
       '屏幕门廊面积', '泳池面积', '泳池质量', '围栏质量', '其他', '杂项',
       '已售月份', '售出年份', '销售类型', '销售条件', '售价']
class_name = ['160', '70', '40', '75', '45', '80', '50', '20', '85', '180', '30', '120', '150', '90', '60', '190']
dot_data = tree.export_graphviz(clf
                                ,feature_names=feature_name
                                ,class_names=class_name
                                ,filled=True,rounded=True,special_characters=True)

dot_data = dot_data.replace('fontname="helvetica"', 'fontname="SimHei"')
graph = graphviz.Source(dot_data)
graph.render('HouseTree')

#重要属性和接口
print(clf.feature_importances_)
print({key: value for key, value in dict(zip(feature_name,clf.feature_importances_)).items() if value != 0})
print(clf.apply(xtest))
print(clf.predict(xtest))
"""最重要的特征
'BldgType' 'HouseStyle' 'YearBuilt'
"""
#######################随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ##这个要跑几分钟
# superpa = []
# for i in range(100):
#       rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
#       rfc_s = cross_val_score(rfc,xtrain,ytrain,cv=4).mean()
#       superpa.append(rfc_s)
# print(max(superpa),superpa.index(max(superpa))+1)
# plt.figure(figsize=[20,5])
# plt.plot(range(1,101),superpa)
# plt.show()

rfc = RandomForestClassifier(n_estimators=78,random_state=0)
rfc = rfc.fit(xtrain,ytrain)
score_r_c = cross_val_score(rfc,xtrain,ytrain,cv=4).mean()
print(score_r_c)
print(rfc.predict(xtest))

#############################################聚类算法 效果一般
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

BT_HS = test0[['BldgType','HouseStyle']].values
x_FS = test0[['1stFlrSF', '2ndFlrSF']].values
x_FS_train = train0[['1stFlrSF', '2ndFlrSF']].values
x_BH = sum[['BldgType','HouseStyle']].values
y_MSSC = ytest
y_MSSC_train = ytrain

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(BT_HS[:,0],BT_HS[:,1],marker='o',s=8)
cluster = KMeans(n_clusters=len(class_name),random_state=0).fit(x_BH)
y_pred = cluster.labels_##预测结果
centriod = cluster.cluster_centers_##聚类中心
inertia = cluster.inertia_##评分标准，越小越好
color = ['red','pink','orange','gray','green','blue']
for i in range(len(class_name)):
    ax2.scatter(x_BH[y_pred==i,0],x_BH[y_pred==i,1],marker='o',s=8,c=color[i%6])
ax2.scatter(centriod[:,0],centriod[:,1],marker='x',s=30,c='black')
plt.show()

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(BT_HS[:,0],BT_HS[:,1],marker='o',s=8)
cluster = KMeans(n_clusters=len(class_name),random_state=0).fit(x_FS)
y_pred = cluster.labels_##预测结果
centriod = cluster.cluster_centers_##聚类中心
print(cluster.inertia_)##评分标准，越小越好
color = ['red','pink','orange','gray','green','blue',]
ax2.scatter(x_FS[:,0],x_FS[:,1],marker='o',s=20,c=y_pred,cmap='tab20')
ax2.scatter(centriod[:,0],centriod[:,1],marker='x',s=30,c='black')
plt.show()
print()
####根据轮廓系数选择n_clusters
for n_clusters in range(2,len(class_name)+1):
    n_clusters = n_clusters
    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(18,7)
    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0,x_FS.shape[0]+(n_clusters+1)*10])
    clusterer = KMeans(n_clusters=n_clusters,random_state=10).fit(x_FS)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(x_FS,cluster_labels)
    print('For n_clusters =',n_clusters,'The average silhouette_score is :', silhouette_avg)
    sample_silhouette_values = silhouette_samples(x_FS,cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower,y_upper),ith_cluster_silhouette_values
                          ,facecolor=color,alpha=0.7)
        ax1.text(-0.05,y_lower +0.5*size_cluster_i,str(i))
        y_lower = y_upper +10
    ax1.set_title('The silhouette plot for the various clusters.')
    ax1.set_xlabel('The silhouette coefficient values')
    ax1.set_ylabel('Cluster label')
    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(x_FS[:, 0], x_FS[:, 1], marker='o', s=8, c=colors)
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', alpha=1, s=200)
    ax2.set_title('The visualization of the clustered data.')
    ax2.set_xlabel('Feature space for the 1st feature')
    ax2.set_ylabel('Feature space for the 2nd feature')
    plt.suptitle(('Silhouette analysis for KMeans clustering on sample data'
                   'with n_clusters = %d' % n_clusters)
                  ,fontsize=14,fontweight='bold')
    plt.show()

###三个特征的聚类
x_BH2=sum[['YearBuilt','1stFlrSF','2ndFlrSF']].values[:1460,:]
##学习曲线
inertia = []
for i in range(2,len(class_name)+1):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(x_BH2)
    inertia.append(kmeans.inertia_)
index = list(range(2,len(class_name)+1))
plt.plot(index,inertia)
plt.title('学习曲线', fontproperties='SimHei', fontsize=14)
plt.xlabel('n_clusters')  ,plt.ylabel('inertia')
##可视化
kmeans = KMeans(n_clusters=inertia.index(min(inertia)),random_state=0).fit(x_BH2)
pred = kmeans.predict(x_BH2)
center = kmeans.cluster_centers_
plt.figure(figsize=(10,10))
ax = plt.subplot(111,projection='3d')
ax.scatter(x_BH2[:,0],x_BH2[:,1],x_BH2[:,2],c=pred,s=20,cmap='cool')
ax.scatter(center[:,0],center[:,1],center[:,2],marker='x',s=30,c='black')
ax.set_xlabel('YearBuilt')  ,ax.set_ylabel('1stFlrSF')  ,ax.set_zlabel('2ndFlrSF')
plt.show()
print(inertia.index(min(inertia)))


################################################SVM 向量机
from sklearn.svm import SVC

plt.scatter(x_FS_train[:,0],x_FS_train[:,1],c=y_MSSC_train,s=5,cmap='rainbow')
plt.xticks([])
plt.yticks([])
plt.xlabel('1stFlrSF')  ,plt.ylabel('2ndFlrSF')
plt.show()

r = np.exp(-(x_FS_train**2).sum(axis=1))#定义一个由x计算出来的新维度r
rlim = np.linspace(min(r),max(r),100)

def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    y,x = np.meshgrid(y,x)
    xy = np.vstack([x.ravel(), y.ravel()]).T
    p = KMeans(n_clusters=len(class_name), random_state=0).fit(xy).labels_.reshape(x.shape)
    ax.contour(x, y, p, colors='k', levels=[-1, 0, 1]
               , alpha=0.5, linestyles=['--', '-', '--'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()
def plot_3D(elev=15,azim=50,x=x_FS_train,y=y_MSSC_train):#elev上下转，azim平行转
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x[:,0],x[:,1],r,c=y,s=5,cmap='rainbow')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()
plot_3D(elev=15,azim=70)
# ###超平面
clf = SVC(kernel='rbf').fit(x_FS,y_MSSC)
plt.scatter(x_FS_train[:,0],x_FS_train[:,1],c=y_MSSC_train,s=50,cmap='rainbow')
plot_svc_decision_function(clf)


score = []
gamma_range = np.logspace(-10,1,50)
for i in gamma_range:
    clf = SVC(kernel='rbf',gamma=i,cache_size=5000).fit(x_FS_train,y_MSSC_train)
    score.append(clf.score(x_FS,y_MSSC))
print(max(score),gamma_range[score.index(max(score))])
plt.plot(gamma_range,score)
plt.show()

###########################################多元线性回归
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
###数据预处理
data_train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
data_test = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
L_train = data_train.select_dtypes(include=['integer', 'float'])###选取了数据类型为数字的
L_test = data_test.select_dtypes(include=['integer', 'float'])
L_train = L_train.fillna(sum.median())
L_test = L_test.fillna(sum.median())
ytrain = np.array(L_train.loc[:,'SalePrice'])
x_train = L_train.drop('SalePrice',axis=1)
x_train = x_train.drop('Id',axis=1)
x_test = L_test.drop('Id',axis=1)
x_train.columns = ['一般分区分类', '与物业的街道的线性英尺', '地段面积', '整体材料和表面处理质量',
        '整体状况品级', '建造年份', '改造日期', '砌体单板面积', '类型1成品平方英尺',
       '类型2成品平方英尺', '地下室面积未完工平方英尺', '地下室总面积平方英尺',
        '一楼平方英尺', '二楼平方英尺', '低质量成品面积',
       '地上居住面积', '地下室全套浴室', '地下室半套浴室', '地上浴室', '地上半浴室',
       '卧室数量', '厨房数量', '地上客房总数', '壁炉数量',
       '车库建成年份', '车位大小', '车库面积', '木甲板面积', '开放式门廊面积',
        '封闭式门廊面积', '三季门廊面积','屏幕门廊面积', '泳池质量', '杂项',
       '已售月份', '售出年份']
x_test.columns = x_train.columns

reg = LinearRegression().fit(x_train,ytrain)
yhat = reg.predict(x_test)
print(yhat)
print([*zip(x_train.columns,reg.coef_)])
print(reg.intercept_)

###可视化
Features = [['与物业的街道的线性英尺', '地段面积'],['建造年份', '改造日期']
    ,['卧室数量', '厨房数量'],['已售月份', '售出年份'],['类型1成品平方英尺',
       '类型2成品平方英尺'],['一楼平方英尺', '二楼平方英尺']]
for i in Features:
    Feature1, Feature2 = i
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    reg = LinearRegression().fit(x_train[[Feature1, Feature2]], ytrain)
    print(reg.coef_)
    z = reg.coef_[0] * x_test[Feature1] + reg.coef_[1] * x_test[Feature2] + reg.intercept_
    x_line = np.linspace(0, 6, 100)
    ax.scatter(x_test[Feature1], x_test[Feature2], z, c=z, s=1, cmap='rainbow', label='线性回归')
    ax.legend()
    ax.set_xlabel(Feature1)
    ax.set_ylabel(Feature2)
    ax.set_zlabel('售价')
    ax.view_init(elev=15, azim=10)
    plt.grid(True)
    plt.show()

