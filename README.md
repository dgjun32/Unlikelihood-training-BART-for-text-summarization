# Sequence-level unlikelihood fine-tuning KoBART

## 1. Backgroud
Current Korean text abstractive generation models suffer from ```text degeneration```, which includes text repetition issue.

Applying advanced <b>stochastic decoding strategies</b> such as ```top-k sampling``` and ```nucleaus sampling``` can mitigate text repetition issue with open-ended language generation tasks. 

Recent research([Cho et al.2019](https://github.com/facebookresearch/unlikelihood_training)) suggests unlikelihood training to resolve text degeneration issue.

I applied unlikelihood training scheme to finetune pretrained ```KoBART```(Korean BART).

## 2. Data Preparation

1. Set directory : ```$cd data/```
2. Download ```.json```file to current dir from the <b>AI hub</b> [link](https://aihub.or.kr/aidata/8054)

## 3. Training

1. Set directory : ```$cd code/```
2. Training + Validation : ```$python main.py```

## 4. Demo
``` 
    [경주=환경일보] 강광태 기자 = 한국수력원자력(사장 정재훈, 이하 한수원)이 지난 6일 경주 황룡원에서 발전소 상주협력사 및 원전건설 주설비 시공사 등 20개 협력사, 39명이 참석한 가운데 '한수원-협력사 안전문화 협의회'를 개최했다.
    
    협의회는 최근 이슈가 되고 있는 협력사 직원들의 산업안전사고를 예방하고, 안전문화에 관한 한수원의 프로그램 등을 전파함으로써 협력사의 안전의식 수준을 개선하기 위해 개최됐다.
    
    협의회에서는 세한대 김천용 교수의 안전문화 특강이 있었고, 협력사 대표들과의 안전다짐 서약식을 통해 원자력 안전에 대한 공동체 의식을 높이고 소통과 협업을 통해 지속적인 안전문화 향상에 노력할 것을 다짐했다.
    
    이어 한수원은 협력사들의 애로사항을 청취하며 현장 최일선까지 정착될 수 있는 안전문화 체계 구축을 약속했고, 안전이 보장된 양질의 일자리 창출을 위한 지역경제 지원 방안에 대해서도 열띤 토론을 펼쳤다.
    
    이재동 한수원 품질안전본부장은 "한수원은 앞으로도 협력사의 안전의식 제고는 물론, 안전에 대한 상생협력을 더욱 강화해 안전사고 예방에 힘쓸 것"이라며, "'안전 최우선 일터 구현'이라는 사회적 가치를 실현하기 위해 끊임없이 노력하겠다"고 말했다.
    --------------------------------------------------------------------------------------------------------------------
    한국수력원자력이 지난 6일 경주 황룡원에서 20개 협력사, 39명이 참석한 가운데 한수원-협력사 안전문화 협의회를 개최하여 세한대 김천용 교수의
    
    안전문화 특강과 안전다짐 서약식을 통해 안전문화 체계 구축을 약속하고 안전이 보장된 양질의 일자리 창출을 위한 지역경제 지원 방안에 대해서도 열띤 
    
    토론을 벌였다.
```

## 5. Result