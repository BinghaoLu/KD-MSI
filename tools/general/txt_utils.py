


def add_txt(path, string):
    with open(path, 'a+') as f:
        f.write(string + '\n')