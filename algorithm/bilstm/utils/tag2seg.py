def tag2seg(sentence, labels):
    assert len(sentence) == len(labels), "sentence and labels length not equal"

    res = ""
    for index in range(len(sentence)):
        if labels[index] == "B":
            res += " "
            res += sentence[index]
        elif labels[index] == "M":
            res += sentence[index]
        elif labels[index] == "E":
            res += sentence[index]
            res += " "
        else:  # labels[index] == "S"
            res += " "
            res += sentence[index]
            res += " "

    return res


if __name__ == "__main__":
    sentence = "这是一个美好的早上"
    label = ["B", "E", "B", "E", "B", "M", "E", "B", "E"]
    res = tag2seg(sentence, label)
    print(res)