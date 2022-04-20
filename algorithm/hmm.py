import pickle
import numpy as np

class HmmModel:

    def __init__(self):
        # 分词状态
        self.stateDict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        # 状态转移矩阵
        self.A = np.zeros((4, 4))
        # 发射矩阵（输出符号的概率分布）,65536是为了保证每一个汉字都能被覆盖
        self.B = np.zeros((4, 65536))
        # 初始状态概率分布
        self.PI = np.zeros(4)

    # 加载数据 先加载模型数据，没有就读取语料库重新训练
    def load(self,train_files, model_file='./seg-data/training/model.pkl'):
        # 加载模型数据
        try:
            with open(model_file, 'rb') as f:
                self.A = pickle.load(f)
                self.B = pickle.load(f)
                self.PI = pickle.load(f)
                return
        except FileNotFoundError:
            pass

        """
        依据数据集统计PI，A,B这三个参数
        train_files: 训练文件（位置及文件名）组成的列表
        """
        for train_file in train_files:
            # 读取训练文本
            fr = open(train_file, encoding='utf-8')
            
            # 逐行操作
            for line in fr.readlines():
                # 获取每行的词语列表
                curLineWords = line.strip().split()
                # 这一行的状态链，它的存在是为了统计A
                wordsLabel = []
                # 对每一个单词进行遍历
                for i in range(len(curLineWords)):
                    # 如果单词长度为1，那么将该词标为"S"，即单个词
                    if len(curLineWords[i]) == 1:
                        label = "S"
                    # 如果单词长度大于1，开头为B，最后为E，中间添加（总长度-2）个M
                    else:
                        label = "B" + 'M' * (len(curLineWords[i]) - 2) + 'E'

                    # 如果是单行开头的第一个字，PI中对应位置加1
                    if i == 0:
                        self.PI[self.stateDict[label[0]]] += 1

                    # 对于单词中的每一个字，在生成的状态链中统计B
                    for j in range(len(label)):
                        # 遍历状态链中每一个状态，并找到对应的中文汉字（对应的ASCII码），并在B对应的位置中加1
                        self.B[self.stateDict[label[j]]][ord(curLineWords[i][j])] += 1

                    # 在整行的状态链中添加该单词的状态链
                    wordsLabel.extend(label)

                # 一行结束后，统计A矩阵
                for i in range(1, len(wordsLabel)):
                    # 统计t时刻状态和t-1时刻状态的所有状态组合的出现次数
                    self.A[self.stateDict[wordsLabel[i - 1]]][self.stateDict[wordsLabel[i]]] += 1

            fr.close()

        # 将PI，A,B转化为概率—归一化
        # 求和，获取分母
        sumPI = np.sum(self.PI)
        # 概率 = 次数/总次数
        for i in range(len(self.PI)):
            # 为了防止结果下溢出，取对数形式
            # 当出现概率为0的时候，手动赋予一个极小值
            if self.PI[i] == 0:
                self.PI[i] = -3.14e+100
            else:
                self.PI[i] = np.log(self.PI[i] / sumPI)

        # A归一化
        for i in range(len(self.A)):
            sumA = np.sum(self.A[i])
            for j in range(len(self.A[i])):
                if self.A[i][j] == 0:
                    self.A[i][j] = -3.14e+100
                else:
                    self.A[i][j] = np.log(self.A[i][j] / sumA)

        # B归一化
        for i in range(len(self.B)):
            sumB = np.sum(self.B[i])
            for j in range(len(self.B[i])):
                if self.B[i][j] == 0:
                    self.B[i][j] = -3.14e+100
                else:
                    self.B[i][j] = np.log(self.B[i][j] / sumB)

        # 保存模型
        self.saveModel(model_file)

    # 保存中间模型数据，即保存A,B,PI
    def saveModel(self, model_file):
        # 序列化
        with open(model_file, 'wb') as f:
            pickle.dump(self.A, f)
            pickle.dump(self.B, f)
            pickle.dump(self.PI, f)

    """
    加载文章
    :param fileName: 文件路径
    :return: 文章每行组成的列表
    """
    def loadArtical(self,fileName):
        # 初始化文章列表
        artical = []

        fr = open(fileName, encoding='utf-8')
        # 读取文章每一行
        for line in fr.readlines():
            curLineWords = line.strip()
            artical.append(curLineWords)
        fr.close()

        return artical

    """
    基于维特比算法实现的分词
    :param artical: 要分词的文章（每行为一项组成的列表）
    :return:分词结果（列表）
    """
    def participle(self, artical, train_files):
        self.load(train_files)

        # 存储分词结果
        participleResult = []

        for line in artical:
            # 初始化delta，大小为(文本长度*四种状态)
            delta = [[0 for i in range(4)] for i in range(len(line))]

            # 第一步：初始化
            for i in range(4):
                # 初始化delta状态链中第一个字的四种状态概率
                delta[0][i] = self.PI[i] + self.B[i][ord(line[0])]
            # 定义psi
            psi = [[0 for i in range(4)] for i in range(len(line))]

            # 第二步：递推
            for t in range(1, len(line)):
                # 对于每一个字，求四种状态概率
                for i in range(4):
                    # 初始化一个临时列表，用于存放从前面的四种状态转到当前状态的概率
                    temp = [0] * 4
                    for j in range(4):
                        temp[j] = delta[t - 1][j] + self.A[j][i]

                    # 找到最大的概率
                    maxDelta = max(temp)
                    # 记录最大值对应的状态
                    maxDeltaIndex = temp.index(maxDelta)
                    # 将找到的最大值*b放入delta中
                    delta[t][i] = maxDelta + self.B[i][ord(line[t])]
                    # 在psi中记录对应的最大状态索引
                    psi[t][i] = maxDeltaIndex
            # 初始化状态链列表，开始生成状态链
            sequence = []

            # 第三步：终止；获取最后一个状态概率对应的索引
            i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1]))
            # 在状态链中添加索引
            sequence.append(i_opt)

            # 第四步:最优路径回溯
            # 从后往前遍历整条链
            for t in range(len(line) - 1, 0, -1):
                i_opt = psi[t][i_opt]
                sequence.append(i_opt)
            # 需要翻转一下
            sequence.reverse()

            # 根据状态链开始进行分词
            curLineWords = ''
            for i in range(len(line)):
                # 在字符串中加入该字
                curLineWords += line[i]
                # 如果该字是3：S->单个词  或  2:E->结尾词 ，并且不是这句话的最后一个字，则在该字后面加上分隔符
                if (sequence[i] == 3 or sequence[i] == 2) and i != (len(line) - 1):
                    curLineWords += '  '
            # 在返回的列表中添加分词后的该行
            participleResult.append(curLineWords)
        # 返回分词结果
        return participleResult

    def store_result(self,store_file_name,partiArtical):
        fw = open(store_file_name, "w+", encoding='utf-8')
        for line in partiArtical :
            fw.write(line + "\n")
        fw.close()


if __name__ == '__main__':
    hmm = HmmModel()

    # 加载待分词文章
    artical = hmm.loadArtical('./seg-data/testing/msr_test.utf8')
    # 进行分词
    partiArtical = hmm.participle(artical,['./seg-data/training/pku_training.utf8','./seg-data/training/msr_training.utf8'])
    # 存储分词结果
    hmm.store_result('./seg-data/gold/my_msr_test.utf8',partiArtical)

    # 加载待分词文章
    artical1 = hmm.loadArtical('./seg-data/testing/pku_test.utf8')
    # 进行分词
    partiArtical1 = hmm.participle(artical1,['./seg-data/training/pku_training.utf8', './seg-data/training/msr_training.utf8'])
    # 存储分词结果
    hmm.store_result('./seg-data/gold/my_pku_test.utf8', partiArtical1)
