# 4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)

import codecs
import sys

def character_2_word(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        char_tag_pair = line.strip().split('\t')
        if len(char_tag_pair) < 3:
            output_data.write("\n")
        else:
            char_tag_pair = line.strip().split('\t')
            char = char_tag_pair[0]
            tag = char_tag_pair[2]
            if tag == 'B':
                output_data.write(' ' + char)
            elif tag == 'M':
                output_data.write(char)
            elif tag == 'E':
                output_data.write(char + ' ')
            else: # tag == 'S'
                output_data.write(' ' + char + ' ')
    input_data.close()
    output_data.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("pls use: python crf_data_2_word.py input output")
        # python crf_data_2_word.py pku_test4crf.tag.utf8 pku_test4crf.tag2word.utf8
        sys.exit()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    character_2_word(input_file, output_file)