# nlp-seg

Chinese word separation task for NLP

## Dataset

PKU and MSR

| Corpus             | Encoding | Word Types | Words     | Character Types | Characters |
| :----------------- | -------- | ---------- | --------- | --------------- | ---------- |
| Peking University  | CP936    | 55,303     | 1,109,947 | 4,698           | 1,826,448  |
| Microsoft Research | CP936    | 88,119     | 2,368,391 | 5,167           | 4,050,469  |

Download [the whole dataset](https://drive.google.com/file/d/1d23Jxmk33GUFycalL4uplOX1cl2whOUl/view?usp=sharing), and extract to the project root directory.

## Algorithms

Chinese word separation using multiple types of algorithms, including statistical model, machine learning model, deep learning model, and large-scale pre-trained model

### N-gram

N-gram 语言模型基于统计学模型方法，统计文本中的词频，同时统计文本中词与前缀词的出现的条件概率，通过极大似然估计统计分词的概率。

#### Bigram Model

Bigram Model 进行分词考虑当前词与前一个词的条件概率，通过选择合适的分词点，最大化整个词语的最大似然概率，从而根据训练数据实现分词。

#### 实现过程

Bigram 的实现过程总体上主要分为：

1. 数据处理
2. 模型训练
3. 频率平滑计算
4. 极大似然估计分词

##### 数据处理

在进行分词过程中，通过数据可以看到，标点符号对于句子有天然的分词效果，而对于 Bigram 来说，标点符号同样也影响训练时状态转化的频率，所以首先要去除标点符号。

- "，", "。", "？", "！"，等终止符号，则将句子分为两句
- 对于非终止符号，就只去掉符号，避免影响训练

在所有的句子前后加 `<BEG> `和 ``<END>``标记，标识句子的头尾。

##### 模型训练

模型训练过程主要分为两部分：

- 统计词频：统计训练文档中所有词语出现的频率
- 统计状态转换频率：统计训练文档中词状态转换的频率（前词 -> 后词）

##### 频率平滑计算

对于未登录词，若概率设为0则会导致极大似然归零，影响分词，所以要进行平滑。这里使用了最简单的 $+\delta$ 平滑操作，即：

$$
P_{+1}(w_{n}|w_1...w_{n-1}) =\frac{c(w_1 w_2...w_n) + \delta}{c(w_1 w_2...w_{n-1}) + |V|\delta}
$$

在这里取$\delta = 3\times10^{-20}$

##### 极大似然估计分词

对于一个句子进行分词，若穷举所有分词情况，则时间复杂度为$O(n^2)$

所以这里将句子中的每个字视为节点，构造有向无环图，使用动态规划从句子头部计算，选择从句首到句尾，能使得似然函数最大的路径。

而后根据 DAG 的路径得到分词结果。

![DP](https://codle.net/content/images/2019/10/dp.png)

> ref: https://codle.net/chinese-word-cutter-1/

ps: 这里在动态规划求似然函数最大路径的时候，当出现未登录词时会导致程序将未登录词到句子末尾分为一整个词，导致分词出现错误，可见程序的健壮性比较差。所以我在计算词的对数概率时加了正则处理，即对于$P(w_n)$，正则方法：

$$
P_{re}(w_n) = len(w_n) \times \ln(P(w_n))
$$

采取这样的方法可以使程序更倾向于分更短的词，而不是将整个句子分为一个词

#### 结果

##### MSR Dataset

| Method | TOTAL TRUE WORD COUNT | TOTAL TEST WORD COUNT | Recall | Precision | F1-score | OOV Rate | OOV Recall Rate | IV Recall Rate |
| :----: | :-------------------: | :-------------------: | ------ | :-------: | :------: | :------: | :-------------: | :------------: |
| Bigram |        106873        |        108706        | 0.838  |   0.824   |  0.831  |  0.026  |      0.327      |     0.852     |


##### PKU Dataset

| Method | TOTAL TRUE WORD COUNT | TOTAL TEST WORD COUNT | Recall | Precision | F1-score | OOV Rate | OOV Recall Rate | IV Recall Rate |
| :----: | :-------------------: | :-------------------: | ------ | :-------: | :------: | :------: | :-------------: | :------------: |
| Bigram |        104372        |        100841        | 0.843  |   0.873   |  0.858  |  0.058  |      0.527      |     0.863     |

##### 结果分析

TODO


#### 评价

理论到实践的道路还是很曲折的，bigram的理论很简单，在上手之前实现的思路也很清晰，但是当真正面对问题，要用代码去解决的时候，需要处理很多没有预料到的细节问题，比如未登录词导致整个句子分词错误、平滑算法的实现错误和动态规划没能够找到正确路径时的错误排查。在实践过程中遇到的各种问题都不断加深对于方法的理解。

同时，对于数据的预处理也尤为重要，好的预处理给分词结果带来的提升甚至要优与实验算法的差异。

## Evaluation Script

```
perl ./seg-data/scripts/score seg-data/gold/pku_training_words.utf8 ./seg-data/gold/pku_test_gold.utf8 ./output/pku_test.utf8.seg > pku_score.txt

perl ./seg-data/scripts/score seg-data/gold/msr_training_words.utf8 ./seg-data/gold/msr_test_gold.utf8 ./output/msr_test.utf8.seg > msr_score.txt
```
