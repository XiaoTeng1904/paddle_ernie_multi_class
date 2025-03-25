from paddlenlp import Taskflow
import pandas as pd

datas_reviews = pd.read_csv('./reviews/test.csv')
reviews_list = datas_reviews['场景提取'].to_list()
# 模型预测
cls = Taskflow("text_classification", task_path='checkpoint/export', is_static_model=True)
data_predicts = cls(reviews_list)

# 初始化空列表
texts = []
labels = []

# 遍历数据，提取text和label
for item in data_predicts:
    texts.append(item['text'])
    labels.append(item['predictions'][0]['label'])

# 创建DataFrame
df = pd.DataFrame({
    '场景提取': texts,
    '一级场景分类': labels
})

# 保存为Excel文件
df.to_csv('./reviews/test_ouput.csv', index=False)