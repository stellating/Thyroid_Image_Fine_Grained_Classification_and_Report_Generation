from utils import create_input_files, load_feature

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='thyroid',
                       karpathy_json_path='./dic_thyroid_gyt_20200911.json',
                       feature_json_path='feature_CN.json',
                       image_folder='./images/',
                       captions_per_image=10,
                       min_word_freq=0,
                       output_folder='./caption_data_crop_1/',
                       max_len=16)

    # load_feature(word_map_file='caption_data_crop/WORDMAP_thyroid_10_cap_per_img_0_min_word_freq.json',
    #              feature_file='feature_CN.json',
    #              output_json='caption_data_crop/FEATURE_thyroid_10_cap_per_img_0_min_word_freq.json')
