from src.utils.data_util import format_gec, format_data2text


format_gec("data_orig/gec/tmu-gfm-dataset.csv", "data/gec", shot_num=8)
format_data2text("data_orig/data2text/", "data/data2text", shot_num=8)
