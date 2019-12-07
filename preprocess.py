import os

from nltk.tokenize import word_tokenize


def lang_data_dir(lang):
    return "dataset/" + lang

def lang_data_file_dir(lang, filename):
    return "dataset/" + lang + "/" + filename

def preprocess(langs):
    def preprocess_lang(lang, paths):
        tokens = dict()

        def add_token(token):
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1

        cnt = 0

        for path in paths:
            cnt += 1

            with open(path) as f:
                try:
                    print(f'[{lang} {cnt}/{len(paths)}] Read file {path}')

                    file_tokens = word_tokenize(f.read())

                    for token in file_tokens:
                        add_token(token)
                except:
                    print('Exception occurred during reading...')

        sorted_tokens = sorted(tokens.items(), key=lambda kv: kv[1], reverse=True)
        print(sorted_tokens[:400])

    for lang in langs:
        data_dir = lang_data_dir(lang)

        items = os.listdir(data_dir)
        paths = [os.path.join(data_dir, item) for item in items]
        preprocess_lang(lang, paths)

        return

def main():
    with open('langs.txt') as f:
        lines = [line.rstrip('\n') for line in f]
        preprocess(lines) 

if __name__ == '__main__':
    main()
