import os
import shutil
import uuid


def lang_data_dir(lang):
    return "dataset/" + lang

def lang_data_file_dir(lang, filename):
    return "dataset/" + lang + "/" + filename

def aggregate(directory, lang):
    dataset = "dataset/" + lang + "/"

    file_lang = '.' + lang

    def walk(cur):
        for item in os.listdir(cur):
            item_full_path = os.path.join(cur, item)

            if os.path.isdir(item_full_path):
                walk(item_full_path)
            else:
                if item.endswith(file_lang):
                    shutil.copyfile(item_full_path, dataset + str(uuid.uuid4()))
                    print(item)

    walk(directory)

def main():
    with open('langs.txt') as f:
        lines = [line.rstrip('\n') for line in f]
        langs = lines
        
        for line in lines:
            if not os.path.exists("dataset/" + line):
                os.makedirs("dataset/" + line)

    with open('repositories.txt') as f:
        lines = [line.rstrip('\n') for line in f]
        for line in lines:
            _split = line.split(' ')

            if len(_split) > 1:
                directory, lang = line.split(' ')
                aggregate(directory, lang)

if __name__ == '__main__':
    main()
