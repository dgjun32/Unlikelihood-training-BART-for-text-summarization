import easydict
from easydict import EasyDict as edict

cfg = edict()

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
cfg.train.max_epochs = 20
cfg.train.batch_size = 8
cfg.train.optimizer = 'torch.optim.AdamW'
cfg.train.lr_schedule = 'torch.optim.lr_scheduler.CosineAnnealingLR'
# validation & inference
cfg.val = edict()
cfg.val.batch_size = 16
# generation
cfg.gen = edict()
cfg.gen.num_beams = 5
cfg.gen.no_repeat_ngram_size = 3
cfg.gen.max_length = 50
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