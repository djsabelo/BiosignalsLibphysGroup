import os
from pylab import *


def files_with_pattern(dir):
    for path, dirs, files in os.walk(dir):
        for f in files:
            yield os.path.join(path, f)


def get_files(dir, lstrings=''):
    # receive one string or a list of strings
    if is_string_like(lstrings):
        return [i for i in files_with_pattern(dir) if lstrings in i]
    else:
        list_files = [i for i in files_with_pattern(dir)]
        for include_string in lstrings[:]:
            list_files = [file for file in list_files if include_string in file]
    return list_files

dataset_path = ".//Mouse file_corr//"
error_file = get_files(dataset_path, ['.'])

for i, f in enumerate(error_file):
    count = 0
    old_file = open(f, 'r')
    length = sum(1 for _ in old_file)
    name = ".//Mouse file_final//" + old_file.name[20:]
    new_file = open(name, 'w')
    for j in range(length):
        old_file = open(f, 'r')
        line = old_file.readlines()[j]
        print(line)
        tab = []
        for ii in arange(0, len(line)):
            if line[ii] == "\t":
                tab.append(ii)
        if len(tab) > 11:
            count += 1
            new_file.write(line[:tab[10]+14] + "\n" + line[tab[10]+14:])
            print("change")
        elif len(tab) != 0:
            new_file.write(line)
    new_file.close()
    if count == 0:
        os.remove(new_file.name)
