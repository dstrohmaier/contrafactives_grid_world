python data_creation.py creation_settings.json all_data.tsv
python creation_encoding.py encoding_cfg.json
python down_sampler.py all_data.tsv sampled.tsv
python splitter.py sampled.tsv train.tsv test.tsv
python cross_validation_creation.py train.tsv cv_splits/
