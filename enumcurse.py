def enumcurseing():
    with open("curse_words.csv", 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            word = lines[i].strip()
            if not word.endswith("s"):
                if i + 1 == len(lines) or lines[i + 1].strip() != word + "s":
                    lines.insert(i + 1, word + "s\n")
                    i += 1
            i += 1
    with open("curse_words_new.csv", 'w') as file:
        file.writelines(lines)
        
enumcurseing()