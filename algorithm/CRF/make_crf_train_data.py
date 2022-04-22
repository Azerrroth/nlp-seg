# 4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)
import sys
import codecs

def data_tag(input_file,output_file):
    input_data=codecs.open(input_file,'r','utf-8')
    output_data=codecs.open(output_file,'w','utf-8')
    for line in input_data.readlines():
        #逐行读取数据
        #按空格切分
        word_list=line.strip().split()
        for word in word_list:
            if len(word)==1:
                output_data.write(word+"\tS\n")
            else:# BMMME
                output_data.write(word[0]+"\tB\n")
                for w in word[1:len(word)-1]:
                    output_data.write(w+"\tM\n")
                output_data.write(word[len(word)-1]+"\tE\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("pls use: python make_crf_train_data.py input output")
        #python make_crf_train_data.py ../../../seg-data/training/pku_training.utf8 pku_training.tagging2crf.utf8
        sys.exit()
    input_file=sys.argv[1]
    output_file=sys.argv[2]
    data_tag(input_file,output_file)