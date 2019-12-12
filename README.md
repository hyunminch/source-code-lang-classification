# 2019년 가을학기 기계학습 개론 프로젝트

## 프로젝트 설정
```sh
conda env create -f environment.yml
conda activate cnnproj
```

## 학습
```sh
python train.py
```

20 epoch 동안 실행되며, 실행되는 동안엔 epoch가 끝날 때마다 한 epoch 동안의 loss가 출력됩니다. 학습이 완료가 되면 score가 standard output으로 출력되고, confusion matrix의 창이 열립니다.

## 파일별 설명
- `dataset.py`: CodeDataset을 통해 파일들을 읽고 1KB 단위 소스 코드 chunk들의 데이터셋을 구성합니다.
- `aggregate.py`: `repositories.txt`와 `langs.txt`를 읽고 `data/` 디렉토리에 소스 코드를 저장합니다. 저장된 소스 코드는 `dataset.py`에 의해 chunk들로 변환됩니다.
- `model.py`: 문자 기반 CNN을 정의한 파일입니다. alphabet_size를 입력받고, 각 언어일 확률을 출력합니다.
- `train.py`: main()이 정의된 곳으로, 20 epoch 동안 실행되며 `model.py`의 신경망을 학습합니다.
