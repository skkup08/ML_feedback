# 26개 기구의 학습 자료

2026년 11월 24일 KST 기준으로 기구별 자료와 Colab 저장 출력, Drive 모델 파일을 대조했습니다. 정상 참조 모델 파일 20개와 현장 검증 완료 기구 0/26개를 확인했습니다. [Colab 실행 확인](docs/COLAB_RUNS.md) 기구 이름은 센터 목록에 맞췄으며, 이름만으로 동작을 확정하기 어려운 경우 후보를 표시했습니다. 신규 학습·평가용 영상에는 해당 동작과 검수된 정상·오류 라벨, 참가자 또는 분리 가능한 촬영 그룹 ID가 필요합니다.

`GOOD_ONLY`는 정상으로 추정한 예시를 기반으로 만든 참조 모델입니다. 오류 사례의 재현율은 오류 라벨이 있는 별도 참가자 영상에서 확인해야 합니다.

| 번호 | 기구 | 기존 코랩의 자료 또는 모델 | 남은 확인·자료 |
| ---: | --- | --- | --- |
| 1 | Treadmill | 보행 후보에서 관절 샘플 부족. 후속 웹 점검에 참고 영상 확보. | 보행·달리기별 검수 영상 |
| 2 | Recumbent Bike | 관절 부족. recumbent_cycling | 등받이형 페달링의 관절이 보이는 검수 영상 |
| 3 | Upright Bike | GOOD_ONLY 후보. 12개 재시도에서 유효 관절 샘플 8개로 저장. | 설치 기구의 정상·오류 검수 영상 |
| 4 | Elliptical | 관절 부족. elliptical_training | 기구 위에서 전신이 보이는 검수 영상 |
| 5 | Seated Chest Press | 관절 부족. seated_chest_press와 machine_chest_press 모두 부족. 별도 기구인지 미확정. | 좌석·등받이·손잡이가 확인되는 프레스 영상 |
| 6 | Pectoral Fly | GOOD_ONLY 후보. pec_deck_fly와 chest_fly 모두 같은 Kaggle 운동 검색어와 28 관절 샘플. 독립 2종으로 세지 않음. | 수평 모으기 동작의 오류 라벨 |
| 7 | High Fly | 매핑 미확정. high_fly 파일 존재. 랫풀다운 후보이며 가슴 플라이로 자동 연결하지 않음. | 설치 모델·수행 동작 확인 |
| 8 | Shoulder Press | GOOD_ONLY 후보. machine_shoulder_press | 설치형 프레스의 오류 라벨 |
| 9 | Long Pull | GOOD_ONLY 후보. 12개 재시도에서 유효 관절 샘플 9개로 저장. | 고정 좌석 케이블 로우의 정상·오류 검수 영상 |
| 10 | Arm Curl | 관절 부족. machine_biceps_curl; 일반 컬은 별개 | 머신 손잡이·팔꿈치가 보이는 검수 영상 |
| 11 | Abdominal | GOOD_ONLY 후보. 12개 재시도에서 이미지의 유효 관절 샘플 8개로 저장. | 머신 크런치의 정상·오류 검수 영상 |
| 12 | Chin Up | GOOD_ONLY / 변형 미확정. chin_up; 보조형 여부 미확정 | 보조형 여부와 해당 동작 라벨 |
| 13 | Rotary Torso | 후보 2개에서 관절 샘플 부족. 후속 웹 점검에서 학습용 자료 미확보. | 로터리 토르소 검수 영상 |
| 14 | Leg Extension | GOOD_ONLY 후보. leg_extension | 머신 무릎 펴기의 오류 라벨 |
| 15 | Leg Curl | GOOD_ONLY / 변형 미확정. leg_curl; 좌식/누운 자세 등 기구 변형 확인 | 좌식·누운 방식 확인과 해당 동작 라벨 |
| 16 | Seated Leg Press | 후속 웹 점검에서 좌식 참고 영상 확보. 자세 정답·독립 참가자 조건 미달. | 좌식 레그 프레스 검수 영상 |
| 17 | Cable Cross Over | GOOD_ONLY 후보. cable_crossover 보존. high_to_low_fly 및 low_to_high_fly의 별도 시도는 미학습. | 케이블 방향별 동작 라벨 |
| 18 | Smith Machine | 관절 부족. smith_squat만 시도; 자유중량 스쿼트와 다름 | 스미스 가이드에서 수행한 동작별 영상 |
| 19 | Mega Max | GOOD_ONLY / 변형 미확정. mega_max_lat_pulldown; 설치 복합기구 동작 미확정 | 설치 복합기구에서 지원할 동작 확인 |
| 20 | Chinning Dipping Hip Flexor | GOOD_ONLY / 부분. dip_station_dip 한 동작만 | 딥스·매달려 다리 들기 등 동작별 라벨 |
| 21 | Power Leg Press | GOOD_ONLY 후보. power_leg_press | 설치형 파워 레그 프레스의 오류 라벨 |
| 22 | Power Rack | 기존 GOOD_ONLY 후보. 11월 23일 하이바 백스쿼트 연구 학습 및 별도 참가자 평가. | 센터 설치 파워랙과 앱 입력으로 새 참가자 평가 |
| 23 | Sit Up | GOOD_ONLY 후보. sit_up | 설치 기구의 싯업 오류 라벨 |
| 24 | Double Twist | 자료 없음. double_twist; Rotary Torso와 구분 | 더블 트위스트 검수 영상 |
| 25 | Hyper Extension | GOOD_ONLY 후보. hyperextension | 설치 기구의 하이퍼 익스텐션 오류 라벨 |
| 26 | Super Bench Press (Incline/Decline) | GOOD_ONLY 후보. incline/decline은 한 기구 항목의 2개 동작 후보. flat_bench_press는 별도. | 벤치 경사별 정상·오류 라벨 |

## 11월 23일 Colab 재시도 22개

보고된 고유 자료 수는 영상 또는 참고 이미지의 건수입니다. 유효 관절은 관절 추출 조건을 통과한 샘플 수입니다. 최소 기준은 8개였습니다. 5개 키는 자료를 찾지 못했고, 11개 키는 유효 관절 샘플 2–5개에 그쳤습니다. 6개 키에서 정상 참조 모델을 저장했습니다.

| 동작 | 코드 키 | 고유 자료 | 유효 관절 | 결과 |
| --- | --- | ---: | ---: | --- |
| 트레드밀 걷기 | `treadmill_walk` | 0 | 0 | 자료 없음 |
| 리컴번트 바이크 | `recumbent_cycling` | 2 | 2 | 관절 샘플 부족 |
| 업라이트 바이크 | `upright_cycling` | 2 | 2 | 관절 샘플 부족 |
| 일립티컬 | `elliptical_training` | 7 | 5 | 관절 샘플 부족 |
| 시티드 체스트 프레스 | `seated_chest_press` | 4 | 4 | 관절 샘플 부족 |
| 펙덱 플라이 | `pec_deck_fly` | 28 | 28 | 정상 참조 모델 생성 |
| 롱 풀 시티드 케이블 로우 | `seated_cable_row` | 3 | 3 | 관절 샘플 부족 |
| 머신 암 컬 | `machine_biceps_curl` | 2 | 2 | 관절 샘플 부족 |
| 머신 앱도미널 크런치 | `abdominal_crunch` | 4 | 4 | 관절 샘플 부족 |
| 로터리 토르소 | `rotary_torso` | 0 | 0 | 자료 없음 |
| 레그 익스텐션 | `leg_extension` | 201 | 25 | 정상 참조 모델 생성 |
| 시티드 레그 프레스 | `seated_leg_press` | 0 | 0 | 자료 없음 |
| 스미스 스쿼트 | `smith_squat` | 4 | 4 | 관절 샘플 부족 |
| 더블 트위스트 | `double_twist` | 0 | 0 | 자료 없음 |
| 플랫 벤치프레스 | `flat_bench_press` | 212 | 35 | 정상 참조 모델 생성 |
| 인클라인 벤치프레스 | `incline_bench_press` | 216 | 34 | 정상 참조 모델 생성 |
| 디클라인 벤치프레스 | `decline_bench_press` | 193 | 12 | 정상 참조 모델 생성 |
| 머신 체스트 프레스 | `machine_chest_press` | 4 | 4 | 관절 샘플 부족 |
| 머신 체스트 플라이 | `chest_fly` | 197 | 28 | 정상 참조 모델 생성 |
| 하이 투 로우 플라이 | `high_to_low_fly` | 0 | 0 | 자료 없음 |
| 로우 투 하이 플라이 | `low_to_high_fly` | 2 | 2 | 관절 샘플 부족 |
| 인클라인 덤벨 플라이 | `incline_dumbbell_fly` | 2 | 2 | 관절 샘플 부족 |

이전에 저장된 정상 참조 모델 키는 `high_fly`, `machine_shoulder_press`, `chin_up`, `leg_curl`, `cable_crossover`, `mega_max_lat_pulldown`, `dip_station_dip`, `power_leg_press`, `power_rack_squat`, `sit_up`, `hyperextension`입니다. 이 실행 직후에는 11개 기존 키와 6개 신규 키로 합계 17개였습니다. 후속 12개 재시도에서 정상 참조 모델 3개를 추가해 20개가 됐습니다. [단계별 결과와 남은 9개](docs/COLAB_RUNS.md) 펙덱 플라이와 머신 체스트 플라이는 일부 자료를 공유했고, 각 키의 현장 적용 여부를 판단하려면 설치 기구를 확인해야 합니다.

별도 37개 세부 동작 프로토타입의 연결 이력은 당시 범위와 설정을 그대로 기록합니다. [세부 동작 연결표](results/prototype_37.csv)
