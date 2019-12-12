import os
import shutil
import uuid


def train_lang_data_dir(lang):
    return "data/train/" + lang

def test_lang_data_dir(lang):
    return "data/test/" + lang

def train_lang_data_file_dir(lang, filename):
    return "data/train/" + lang + "/" + filename

def test_lang_data_file_dir(lang, filename):
    return "data/test/" + lang + "/" + filename

lang_cnt = dict()
alphabet = set()

# Reads directory and for the lang given, aggregates 
# each directory's files and aggregates them at either
# the test_path or train_path.
def aggregate(directory, lang, max_cnt=900):
    file_lang = '.' + lang

    test_threshold = (max_cnt * 9) // 10

    def walk(cur):
        for item in os.listdir(cur):
            item_full_path = os.path.join(cur, item)

            if os.path.isdir(item_full_path):
                walk(item_full_path)
            else:
                if item.endswith(file_lang):
                    with open(item_full_path) as f:
                        try:
                            for char in f.read():
                                alphabet.add(char)

                            if lang in lang_cnt:
                                lang_cnt[lang] += 1
                            else:
                                lang_cnt[lang] = 1

                            cnt = lang_cnt[lang]

                            if cnt > max_cnt:
                                return
                            elif cnt >= test_threshold:
                                test_path = test_lang_data_file_dir(lang, str(uuid.uuid4()))
                                shutil.copyfile(item_full_path, test_path)
                                print("Test: ", item)
                            else:
                                train_path = train_lang_data_file_dir(lang, str(uuid.uuid4()))
                                shutil.copyfile(item_full_path, train_path)
                                print("Train: ", item)
                        except:
                            print("Exception occurred while building alphabet.")

    walk(directory)

def main():
    with open('langs.txt') as f:
        lines = [line.rstrip('\n') for line in f]
        langs = lines
        
        for line in lines:
            if not os.path.exists("data/train/" + line):
                os.makedirs("data/train/" + line)
            if not os.path.exists("data/test/" + line):
                os.makedirs("data/test/" + line)

    with open('repositories.txt') as f:
        lines = [line.rstrip('\n') for line in f]
        for line in lines:
            _split = line.split(' ')

            if len(_split) > 1:
                directory, lang = line.split(' ')
                aggregate(directory, lang)

if __name__ == '__main__':
    main()
