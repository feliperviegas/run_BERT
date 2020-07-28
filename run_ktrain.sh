#!/usr/bin/env bash

path_bert=$1;
path_test=$2
out=$3;

#for dataset in  aisopos_ntua debate english_dailabor nikolaos_ted pang_movie \
#    sanders sarcasm sentistrength_bbc sentistrength_digg sentistrength_myspace \
#    sentistrength_rw sentistrength_twitter sentistrength_youtube stanford_tweets \
#    tweet_semevaltest vader_amazon vader_movie vader_nyt vader_twitter yelp_reviews ; do
for dataset in aisopos_ntua; do
    mkdir "${out}/${dataset}"
    for fold in 0 1 2 3 4; do
        python3 exec_ktrain.py -t ${path_bert}/${dataset}/train${fold}.csv \
        -l ${path_test}/${dataset}/d_test${fold} \
        -c ${path_test}/${dataset}/c_test${fold} \
        -o ${out}/${dataset}/results${fold}.txt \
        -d ${dataset} \
        -f ${fold};
    done
done
