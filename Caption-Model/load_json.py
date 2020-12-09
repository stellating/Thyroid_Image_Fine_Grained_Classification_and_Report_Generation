import json
from nltk.translate.bleu_score import corpus_bleu

filepath="./output_hypo_5.json"
filepath2="./output_ref_5.json"
with open(filepath,'r') as load_f:
    load_dict=json.load(load_f)
with open(filepath2,'r') as load_f2:
    load_dict2=json.load(load_f2)
hypotheses=[]
references=load_dict2
print(len(references))
f=open('results.txt','w')
for count in range(len(load_dict)):
    # print(load_dict[0])
    # print(load_dict2[0])
    # print("keys:",load_dict.keys())
    # print("images:",load_dict['images'][0])
    with open('./caption_data_crop/WORDMAP_thyroid_10_cap_per_img_0_min_word_freq.json', 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    words=[]
    words2=[]
    hypotheses_item=[]
    f.write('predict:')
    for i in load_dict[count]:
        f.write(str([rev_word_map[ind] for ind in i]))
        f.write('\n')
        words.append([rev_word_map[ind] for ind in i])
        hypotheses_item.append(ind for ind in i)
    hypotheses.append(hypotheses)
    f.write('original:')
    for i in load_dict2[count]:
        f.write(str([rev_word_map[ind] for ind in i]))
        f.write('\n')
        words2.append([rev_word_map[ind] for ind in i])
    f.write('\n\n\n')

# Calculate BLEU-4 scores
bleu4 = corpus_bleu(references, hypotheses)  # ,weights=(1.0, 0, 0, 0))
f.close()
print(bleu4)
    # print('predict:',words)
    #
    # print('original:',words2)
    # print('\n')
