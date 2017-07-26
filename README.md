#### 情感分类

使用微博数据训练情感分类模型，测试LDA等算法，并使用随机森林等对比结果

#### 流程

1. db中获取微博数据正负样本raw数据
2. 预处理数据后，通过nlp接口对所有line数据分词
3. LDA训练model
4. 评估model，输出accuracy、precision、recall