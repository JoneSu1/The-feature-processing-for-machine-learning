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

