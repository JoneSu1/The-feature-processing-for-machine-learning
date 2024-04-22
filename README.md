# The-feature-processing-for-machine-learning

## 常见数据特征工程方法
### 离散值的处理


首先的离散值就是在数据分布中明显和大多数值差异较大的，且数量较小的部分（特别是在做回归）.

可以在python中将这些表示用”sklearn.preprocessing“包里面的LablelEnconder 公式来将这些实例化出来。

**主要就是使用数字label来对数据中的label进行表示**

```PYTHON
from sklearn.preprocessing import LablelEnconder

gle = LabelEnconder()
genre_labels = gle.fit_transform(vg_df["Genre"])
gene_mappings = {index: label for index, label in enumerate(gle.class)}
genre_mappings

```

也可以用一个更加方便的方法，先建立一个映射的字典

例如
```
# 创建DataFrame
data = {'animal': ['cat', 'dog', 'bird', 'cat', 'dog']}
df = pd.DataFrame(data)
# 创建映射字典
label_mapping = {'cat': 'mammal', 'dog': 'mammal', 'bird': 'bird'}

```

然后就可以使用map函数或者是replace函数，来将label根据字典来进行映射操作。

```
map函数：
map函数将会将DataFrame中某一列的每个元素都传递给提供的映射字典，并返回相应的新值。
如果映射字典中存在当前元素的键，map函数会将该元素替换为对应的值；如果不存在，则保留原始值。
最后，map函数返回一个包含了映射后结果的新Series，你可以将其分配给DataFrame的新列。
replace函数：
replace函数将会在DataFrame中查找某一列的每个元素，并使用提供的映射字典来替换这些元素。
如果映射字典中存在当前元素的键，replace函数会将该元素替换为对应的值；如果不存在，则保留原始值。
replace函数默认是在整个DataFrame中查找并替换指定的值，但你也可以通过设置参数来指定只在某一列中进行替换。
最后，replace函数会返回一个新的DataFrame，其中已经将指定的值替换为了对应的新值。
```

```PYTHON
# 使用map函数进行映射
df['animal_type'] = df['animal'].map(label_mapping)
print("使用map函数进行标签映射:")
print(df)

# 使用replace函数进行映射
df['animal_type'] = df['animal'].replace(label_mapping)
print("使用replace函数进行标签映射:")
print(df)
```
 
在转换完特征之后，我们可以使用one-hot热编码来将他单独特征提取出来。

使用的也是SK库礼貌的OneHotEncoder()来做的.

```PYTHON
gen_ohe = OneHotEncoder()
gen_feature_arr = gen_ohe.fit_transform(poke_df[['Gen_Label']]).toarray()
gen_feature_labels = list(gen_le.classes_)
gen_features = pd.DataFrame(gen_feature_arr, columns=gen_feature_labels)


```

**可以用一些代码来查看数据的情况**

```
#使用type()函数查看数据结构的类型。
data = [1, 2, 3, 4, 5]
print(type(data))

#使用dir()函数查看对象的属性和方法。
data = [1, 2, 3, 4, 5]
print(dir(data))

使用help()函数获取关于对象的帮助文档，包括其属性和方法的说明。
data = [1, 2, 3, 4, 5]
help(data)

#使用Pandas中的info()方法查看DataFrame的摘要信息。

import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
print(df.info())

#使用Pandas中的head()或tail()方法查看DataFrame的前几行或后几行数据。

import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
print(df.head())  # 查看前几行，默认为前5行
print(df.tail())  # 查看后几行，默认为后5行
#describe() 是 Pandas 中 DataFrame 或 Series 对象的一个方法，
#用于生成关于数据的描述性统计信息。它提供了数据的基本统计指标，如均值、标准差、最大值、最小值、中位数等。

import pandas as pd

# 创建一个DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e']}
df = pd.DataFrame(data)

# 使用describe()方法查看DataFrame的描述性统计信息
print(df.describe())

              A
count  5.000000
mean   3.000000
std    1.581139
min    1.000000
25%    2.000000
50%    3.000000
75%    4.000000
max    5.000000

```

**关于使用transform函数来操作列内值的方法**

**对整列数据进行相同的操作：**

```
import pandas as pd

# 创建一个DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                   'B': [10, 20, 30, 40, 50]})

# 对列'A'的数据进行平方操作
df['A_squared'] = df['A'].transform(lambda x: x ** 2)
print(df)

```

**输出：**

```
   A   B  A_squared
0  1  10          1
1  2  20          4
2  3  30          9
3  4  40         16
4  5  50         25

```

**对整列数据进行分组转换：**

```
import pandas as pd

# 创建一个DataFrame
df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B', 'B'],
                   'Value': [1, 2, 3, 4, 5]})
# 计算每个分组的均值，然后将每个分组的值减去均值
df["Group_mean"] = df.groupby('groupby')['Value'].transform('mean')
df['Value_centered'] = df['Value'] - df['Group_mean']
print(df)

```
### 处理连续变化的值

**也是和之前的label数据一样，进行分档**

如果是特别连续的数据，我们可以尝试按照中位数或者比例来分割

**0，25，5，75，1**

这样的形式进行分割

```
quantile_list = [0, .25, .5, 1.]
quantiles = fcc_survey_df['Income'].quantile(quantile_list)
```

**使用对数变换来集中特征**

**在做机器学习的时候我们都会假设，我们的数据分布是符合正态分布的**

 对数变换的意义就在于缩小了数据的偏度，让它更加的趋近于正态分布

 **就直接使用log做就可以了**

 ```
fcc_survey_df['Income_log'] = np.log((1 + fcc_survey_df['Income']))
fcc_survey_df[['ID.x','Age','Income','Income_log']].iloc[4:9]
#这个.iloc[] 就是选择特定行进行展示的
```

**让它接近对数变换的另外一个方法就是cosbox变换**

和log变换差不多

**如果我们的数据是连续化的，我们要经量去把它离散化了，让它能够产生尽可能多的特征，如果是时间类型的数据
我们可以将时间按照年，月，天，时，分，秒来划分，甚至是，月都分个季节，天都分时段，用足够多的特征来撑起预测**


