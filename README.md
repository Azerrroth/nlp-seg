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

在所有的句子前后加 `<BEG> `和 `<END>`标记，标识句子的头尾。

##### 模型训练

模型训练过程主要分为两部分：

- 统计词频：统计训练文档中所有词语出现的频率
- 统计状态转换频率：统计训练文档中词状态转换的频率（前词 -> 后词）

##### 频率平滑计算

对于未登录词，若概率设为 0 则会导致极大似然归零，影响分词，所以要进行平滑。这里使用了最简单的 $+\delta$ 平滑操作，即：

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
| Bigram |        106873         |        108706         | 0.838  |   0.824   |  0.831   |  0.026   |      0.327      |     0.852      |

##### PKU Dataset

| Method | TOTAL TRUE WORD COUNT | TOTAL TEST WORD COUNT | Recall | Precision | F1-score | OOV Rate | OOV Recall Rate | IV Recall Rate |
| :----: | :-------------------: | :-------------------: | ------ | :-------: | :------: | :------: | :-------------: | :------------: |
| Bigram |        104372         |        100841         | 0.843  |   0.873   |  0.858   |  0.058   |      0.527      |     0.863      |

##### 结果分析

TODO

#### 评价

理论到实践的道路还是很曲折的，bigram 的理论很简单，在上手之前实现的思路也很清晰，但是当真正面对问题，要用代码去解决的时候，需要处理很多没有预料到的细节问题，比如未登录词导致整个句子分词错误、平滑算法的实现错误和动态规划没能够找到正确路径时的错误排查。在实践过程中遇到的各种问题都不断加深对于方法的理解。

同时，对于数据的预处理也尤为重要，好的预处理给分词结果带来的提升甚至要优与实验算法的差异。

### Bidirectional LSTM + CRF

中文分词即序列标注任务，主要是用 B (Begin)、M (Middle)、E (End)、S(Single) 四类标签将句子中的中文字符进行标记，然后根据标记将句子分割，达到分词的目标。

#### Bidirectional LSTM

序列模型 LSTM (Long Short-Term Memory) 可以学习序列的先前状态结合先序状态对当前输入做出判断，同句子序列一致，但是单向的 LSTM 在句子序列标注工作中，无法考虑字词在句子中逆序的编码信息，而文本标注任务在实践中往往是与上下文相关联的，所以在这里采用的 Bidirectional-LSTM 即双向 LSTM 可以结合句子的上下文状态，即对给定字在原文中的前后文信息结合进行判断，能够在理论上既包含历史信息、又能包含未来信息，可能会更有利于对当前词的标注。

![img](https://image.jiqizhixin.com/uploads/editor/df55a9f8-422e-4252-a768-9cf4f49bbb56/1540354954203.png)

> 图源：机器之心——BiLSTM 介绍及代码实现 https://www.jiqizhixin.com/articles/2018-10-24-13

#### CRF

CRF (Conditional Random Field, 条件随机场) 基于文本标注，使用特征函数 $$f(X, i, y_i, y_{i-1})$$ 来抽象表达特征，其中$X$表示输入序列，$i$表示当前位置，$y_i$表示当前的状态，而$y_{i-1}$表示上一个状态。

给定观测的序列$\textbf{X} = X_1X_2X_3...X_n$，可以构成两种标记的团，即：

1. $Y_i, X, i = 1, 2, ... ,n$
2. $Y_{i-1}, Y_i, X, i = 2, ..., n$

CRF 使用特征函数定义概率

$$
P(\pmb Y|\pmb X)=\frac1Zexp(\sum_{j=1}^{K_1}\sum_{i=2}^{n}\lambda_jt_j(Y_{i-1},Y_i,\pmb X,i)+\sum_{k=1}^{K_2}\sum_{i=1}^n\mu_ks_k(Y_i,\pmb X,i))
$$

其中：

- $t_j(Y_{i-1},Y_i,\pmb X,i)$ 为转移特征函数，表现了相邻标记变量间的相关关系。
- $s_k(Y_i,\pmb X,i)$ 为标记位置 $i$ 的状态特征函数，表现了观测序列 $\pmb X$ 对标记变量的影响。
- $\lambda_j, \mu_k$ 为参数，$Z$ 为规范化因子，$K_1$ 为转移特征函数个数，$K_2$为状态特征函数个数

CRF 求解序列标记的概率后，用 Viterbi 算法找到最佳路径，即为该序列概率最大的标注。

#### 实现过程

本次分词任务主要将 BiLSTM 与 CRF 相结合，首先要对序列进行标注，对每个字进行标签标记；然后对语料中的词进行编码，将词的下标转换为向量，之后将词输入 Bi-LSTM 网络，预测其所属标签的概率；之后将预测结果输入 CRF，根据 Viterbi 算法，计算标签预测的最佳路径，得到词序列对应的标签；最后，根据得到的 BMES 标签即可对输入句子进行分词。

```mermaid
graph LR
A[(语料)] -->
pre(预处理) -->
B(标签提取) -->
C(Bi-LSTM) -->
D(CRF) -->
E(标签预测) -->
F[(分词结果)]
%% if{a%b=0 ?}
%% if --->|yes| f1[GCD = b] --> B(结束)
%% if --->|no| f2["a, b = b, a % b "]-->if
```

##### 语料预处理

预处理阶段主要从训练数据中提取字典，然后按照词（汉语中的字）出现频率，将其映射为一个非负整数。

同时对于给定标签，也将标签映射为一个正整数。这里定义了 B M E S 四类标签，可以将标签映射为

```python
{
    'B': 0,
    'M': 1,
    'E': 2,
    'S': 3
}
```

##### 标签提取

标签提取阶段主要将训练数据中的词（字）映射为标签的序号，按照标签意义对句子中的字进行标注：

- B, Begin: 分词的开头字标为 B
- M, Middle: 分词的中间字标记 M
- E, End: 分词的末尾字标记为 E
- S, Single: 单个字作为分词的标记为 S

比如对于句子：

> 人们 常 说 生活 是 一 部 教科书 ， 而 血 与 火 的 战争 更 是 不可多得 的 教科书 ， 她 确实 是 名副其实 的 ‘ 我 的 大学 ’ 。

对应的标签就为

> 人们 常 说 生活 是 一 部 教科书 ， 而 血 与 火 的 战争 更 是 不可多得 的 教科书 ， 她 确实 是 名副其实 的 ‘ 我 的 大学 ’ 。
>
> BE S S BE S S S BME S S S S S S BE S S BMME S BME S S S S S BMME S S S S BE S S

##### Bi-LSTM

Bi-LSTM 是 在原来 LSTM 模型的改进，是一种序列处理模型，主要变化在于增加了逆序序列隐状态的转移，即由两个 LSTM 组成：一个在前向接收输入，另一个在后向接收输入，Bi-LSTM 有效地增加了网络可用的信息量，使得算法能够得知上下文信息。

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-25_at_8.54.27_PM.png" alt="img" style="zoom:30%;" />

> Modelling Radiological Language with Bidirectional Long Short-Term Memory Networks, Cornegruta et al

在本次实验中，这里使用了 PyTorch 的 LSTM 实现，设置 _bidirection=True_ 即可实现双向的 LSTM。

##### CRF

CRF （条件随机场）在给定一组输入序列条件下另一组输出序列的条件概率分布，CRF 接收到来自上层双向 LSTM 输出的对于序列中单个字的四种标签的预测概率，而后输出使得输入序列概率最大的序列的标签。本次实验中使用了 Pypi 上公开的**pytorch-crf**工具，方便实现 CRF 中评分函数及状态转移矩阵等参数的训练优化。

##### 标签预测

上述的上述的**pytorch-crf**的输出即为整个神经网络的输出，为输入序列中词对应的标签的序号。这一步骤根据标签编码方式对其进行解码，就能得到模型对输入序列预测的标签。

##### 分词结果

根据上一步预测的文本标签和标签的实际意义，可以对词语进行分词。在本次实验中，对于 B M E S 的实际意义，对其进行分词：

- B：Begin，词语的开头字，即与句子前文分隔。
- M：Middle，词语的中部，与前文和后文相连接。
- E：End，词语的尾部字，与后文相分隔。
- S：Single，单字成词，与前后文相分隔。

#### 实验结果

##### MSR Dataset

|   Method   | TOTAL TRUE WORD COUNT | TOTAL TEST WORD COUNT | Recall | Precision | F1-score | OOV Rate | OOV Recall Rate | IV Recall Rate |
| :--------: | :-------------------: | :-------------------: | ------ | :-------: | :------: | :------: | :-------------: | :------------: |
| LSTM + CRF |        106873         |        106809         | 0.938  |   0.938   |  0.938   |  0.026   |      0.690      |     0.945      |

##### PKU Dataset

|   Method   | TOTAL TRUE WORD COUNT | TOTAL TEST WORD COUNT | Recall | Precision | F1-score | OOV Rate | OOV Recall Rate | IV Recall Rate |
| :--------: | :-------------------: | :-------------------: | ------ | :-------: | :------: | :------: | :-------------: | :------------: |
| LSTM + CRF |        104372         |        103642         | 0.891  |   0.897   |  0.894   |  0.058   |      0.528      |     0.913      |

##### Case

> 天津市 民冬泳 迈入 新 世纪
>
> 乌鲁木齐 多云－ 10℃／－4℃

### Bert + CRF

#### Bert


#### 实现过程

分词任务主要将 Bert 与 CRF 相结合，首先要对序列进行标注，对每个字进行标签标记；然后对语料中的词进行编码(Tokenizer)，而后将词的下标转换为向量，将词向量输入与训练的 Bert 网络，预测其所属标签的概率；

之后将预测结果输入 CRF，根据 Viterbi 算法，计算标签预测的最佳路径，得到词序列对应的标签；最后，根据得到的 BMES 标签即可对输入句子进行分词。

```mermaid
graph LR
A[(语料)] -->
pre(预处理) -->
B(标签提取) -->
C(Bert) -->
D(CRF) -->
E(标签预测) -->
F[(分词结果)]
%% if{a%b=0 ?}
%% if --->|yes| f1[GCD = b] --> B(结束)
%% if --->|no| f2["a, b = b, a % b "]-->if
```

其余部分与其他实验一致，在此不再赘述，仅对使用 Bert 的预训练模型部分进行解释。


##### Bert


在本次实验中，这里使用了 HuggingFace 开源的 Transformers 实现，使用**bert-base-chinese**预训练模型。使用有标注的训练数据对预训练模型进行微调。

Bert 对于最大序列长度的限制为 512，所以对于长文本，需要将其截断为长度最大510的文本，然后加上开头结尾特殊标记，刚好是512。
"How to Fine-Tune BERT for Text Classification?" 中叙述了多种序列截断的方法。在此，我们选择头部截断的方法，即从句子头部选择长度为510的文本，然后加上开头结尾的特殊标记，后序序列若仍大于512，则再次采取头部截断的方法。

另外，由于采用了预训练的模型，所以在此对 Bert 与其下游的 CRF 分别选择不同的基础学习率来进行训练使模型更快收敛。


## Evaluation Scripts

```shell
perl ./seg-data/scripts/score seg-data/gold/pku_training_words.utf8 ./seg-data/gold/pku_test_gold.utf8 ./output/pku_test.utf8.seg > pku_score.txt

perl ./seg-data/scripts/score seg-data/gold/msr_training_words.utf8 ./seg-data/gold/msr_test_gold.utf8 ./output/msr_test.utf8.seg > msr_score.txt
```
