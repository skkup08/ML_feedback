# 파워랙 하이바 백스쿼트 실험 실행 방법

## 영상과 라벨

BigThree100의 하이바 백스쿼트 첫 세트 중 추가 중량이 0으로 표시된 영상을 사용했습니다. 하나의 입력에 같은 참가자의 정면과 측면이 좌우로 배치되어 있어야 합니다. 센터 설치 기구와 1대 카메라 입력에는 별도 시험이 필요합니다. [원자료](https://digital-media-lab.uni-bremen.de/BigThree100/) · [수집 설계](https://doi.org/10.26092/elib/6434)

| 모델 라벨 | 기준 |
| --- | --- |
| `0` / `PASS` | 무릎 안쪽 붕괴, 발꿈치 들림, 과도한 전경, 깊이 미달, 등 굽음, 허리 과신전 6항목이 모두 미발생으로 기록된 세트 |
| `1` / `FAIL` | 하나 이상이 모든 반복에서 발생했고, 나머지 항목도 미발생 또는 모든 반복으로 기록된 세트 |

깊이 미달이 미발생인 세트만 양쪽 클래스에 포함했습니다. 라벨은 이 연구에서 고정한 항목에 해당합니다. 시험 영상은 학습 12명과 겹치지 않는 사람 중 클래스별 파일 크기가 작은 4편을 골랐습니다. 모두 한 출처의 유사한 촬영 환경이므로 새 체육관의 성능은 별도로 확인해야 합니다.

## 같은 추출 방식으로 영상 처리하기

기록된 실행은 Python 3.12와 FFmpeg, `numpy==2.3.5`, `scikit-learn==1.8.0`, `joblib==1.5.3`, `rtmlib==0.0.15`, `onnxruntime==1.23.2`, OpenCV 5.1.0을 사용했습니다. 추출 단계에는 아래 원배포 ZIP에서 각각 `end2end.onnx`를 꺼내 준비합니다. 코드는 가중치의 SHA-256을 검사합니다. [YOLOX-tiny](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip) · [RTMPose-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip)

```bash
python scripts/power_rack.py extract --video set.mp4 --detector detector.onnx --pose pose.onnx --output set_features.json
```

FFmpeg에서 초당 4프레임을 골라 두 화면에서 한 사람씩 검출된 경우를 사용합니다. 관절 점수 중앙값 0.4 이상, 유효 프레임 쌍 12개 이상, 유효 비율 50% 이상을 요구합니다. 조건을 통과한 세트는 관절 특징의 10·50·90백분위수를 `vector`에 기록합니다. 조건을 충족하지 못한 세트는 `vector`가 `null`입니다. 발목의 화면 내 높이를 포함한 15개 수치를 추출한 뒤 후속 학습에서는 해당 높이의 백분위수 3개를 제외합니다.

## 독립 참가자 자료로 학습·평가하기

추출한 `vector`에 검수 라벨 `label`과 참가자 그룹 `person`을 붙입니다. `train.json`과 `test.json`은 이 세 필드를 가진 객체의 배열입니다. 동일인에게서 얻은 세트는 같은 그룹으로 취급합니다. 이 스크립트는 참가자당 한 세트를 사용하고, 두 배열 간 참가자 겹침을 검사합니다. 각 분할에는 두 클래스가 각각 최소 3개 필요합니다.

```bash
python scripts/power_rack.py fit --train train.json --test test.json --model research.joblib
```

이 명령은 표준화, 로지스틱 회귀(C=0.25), 판정 임계값 0.5를 사용합니다. 저장한 모델을 다시 열어 예측이 일치하는지 확인한 다음 학습·평가 인원과 혼동행렬을 출력합니다. 원본의 세트 선택과 영상 획득은 데이터를 사용하는 사람이 수행합니다.

## 기록된 평가 지표만 확인하기

다음 명령은 Python 표준 라이브러리로 두 모델의 클래스별 재현율과 균형정확도를 혼동행렬에서 다시 계산합니다. 원본 영상과 가중치 없이 실행할 수 있습니다. [평가 기록](../results/power_rack_20261123.json)

```bash
python scripts/power_rack.py verify results/power_rack_20261123.json
```
