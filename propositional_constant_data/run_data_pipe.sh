python data_creation.py 120 all_small_data.tsv
python creation_encoding.py all_small_data.tsv encoding_cfg.json
python splitter.py all_small_data.tsv .
python cross_validation_creation.py train.tsv cv_splits/
