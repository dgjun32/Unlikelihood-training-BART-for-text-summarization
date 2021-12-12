import easydict
from easydict import EasyDict as edict

cfg = edict()

# random seed
cfg.seed = 2021
# dataset
cfg.dataset = edict()
cfg.dataset.text_fields = ['text', 'abstractive']
cfg.dataset.news_useless_feats = ['category', 'media_type', 'media_sub_type', 'media_name', 'size', 'char_count', 'publish_date', 'title', 'annotator_id', 'document_quality_scores', 'extractive']
cfg.dataset.law_useless_feats = ['category', 'size', 'char_count', 'publish_date', 'title', 'annotator_id', 'document_quality_scores', 'extractive']
cfg.dataset.test_useless_feats = ['id', 'media', 'article_original']

# model
cfg.model = edict()
cfg.model.name = 'hyunwoongko/kobart'

# train
cfg.train = edict()
cfg.train.input_length = 512
cfg.train.output_length = 128
cfg.train.max_length = 50
cfg.train.top_p = 0
cfg.train.top_k = 1
cfg.train.max_steps = -1
cfg.train.n_steps = 10000
cfg.train.max_epochs = 100
cfg.train.batch_size = 4
cfg.train.gradient_accumulation_steps = 1
cfg.train.learning_rate = 6.25e-5
cfg.train.weight_decay = 1e-2
cfg.train.adam_eps = 1e-8
cfg.train.lm_coef = 0.9
cfg.train.optimizer = 'torch.optim.AdamW'
cfg.train.lr_schedule = 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts'
cfg.train.warmup_steps = 5
cfg.train.T_mult = 1

# ul_train
cfg.ul_train = edict()
cfg.ul_train.ngram_n = 4
cfg.ul_train.p = 0.5

# validation & inference
cfg.val = edict()
cfg.val.batch_size = 16
cfg.val.validate_every = 10000
cfg.val.example_article = '전두환 전 대통령의 사망 이틀째인 24일 빈소가 마련된 서울 신촌세브란스병원은 전날과 마찬가지로 대체로 한산했다. 정치권의 온도는 싸늘했다. 전직 대통령의 빈소치고는 현역 정치인들의 조문 행렬은 보이지 않았다. 이들이 보내는 조화도 간간이 빈소에 들어설 뿐이었다. 이날 오후 2시 기준으로 21대 국회의원 중에서는 국민의힘 주호영 의원이 유일하게 발걸음했다. 그는 윤석열 대선 후보의 선대위 조직총괄본부장에 내정됐다. 전두환 전 대통령 장례 이틀째인 24일 오전 빈소가 마련된 서울 서대문구 신촌세브란스병원 장례식장에서 이재오 비상시국국민회의 상임의장이 조문하고 있다. 한때 전씨의 사위였던 윤상현 의원이 전날 조문한 데 이어 국민의힘 의원 중에서는 두 번째다. 이날 저녁에는 김기현 원내대표 등도 조문할 예정이다. 주 의원은 "고인에 대해 어떻게 평가하느냐"는 취재진 질문에 "평가는 역사가 할 일이고, 돌아가셨으니 저는 명복을 빌 따름"이라며 "특임장관 시절에 (전씨를) 여러 번 찾아뵀다. (전씨가) 대구 오셨을 때도 여러 번 뵀다"고 말했다. 주요 대선 후보들이 조문을 안 하는 데 대한 생각을 묻자 "제가 언급한 일은 아닌 것 같다"고 말을 아꼈다. 이어 윤 후보의 조문 불참에 대해 묻자 자리를 떠났다.'

# generation
cfg.gen = edict()
cfg.gen.num_beams = 5
cfg.gen.no_repeat_ngram_size = 3
cfg.gen.max_length = 128

# path
cfg.path = edict()
cfg.path.data = '../data'
cfg.path.train_law = 'law_train.json'
cfg.path.train_megazine = 'megazine_train.json'
cfg.path.train_news = 'news_train.json'
cfg.path.val_law = 'law_val.json'
cfg.path.val_megazine = 'megazine_val.json'
cfg.path.val_news = 'news_val.json'
cfg.path.test = 'test.jsonl'
cfg.path.sub = 'submission.csv'
cfg.path.output = '../output'
cfg.path.model = '../model'
cfg.path.submission = '../submission'
