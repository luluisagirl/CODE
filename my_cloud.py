import os
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from matplotlib import pyplot as plt
import random
from matplotlib.font_manager import FontProperties


if __name__ == '__main__':
    #data_path='data/pos_train.csv'
    #cloud_path='image/wordCloud.png'
    #data_path='data/neg_train.csv'
    #cloud_path='image/wordCloud.png'
    data_path='data/Amazon.csv'
    cloud_path='image/wordCloud.png'
    texts = pd.read_csv(data_path, header=None, index_col=None)[0]
    texts = [BeautifulSoup(text, 'html.parser').get_text() for text in texts]
    texts = ''.join(texts)
    background_image = np.array(Image.open('image/background.png'))

    img_colors = ImageColorGenerator(background_image)
    stopwords = set(STOPWORDS)
    stopwords.add('one')
    wc = WordCloud(margin=2,
                   mask=background_image,
                   scale=2,
                   background_color='white',
                   max_words=200,
                   min_font_size=4,
                   stopwords=stopwords,
                   random_state=42,
                   max_font_size=150
                   )
    wc.generate_from_text(texts)
    wc.recolor(color_func=img_colors)
    wc.to_file(cloud_path)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.title(cloud_path[6:-4])
    plt.savefig(cloud_path)
    process_word = WordCloud.process_text(wc, texts)
    sort_list = sorted(process_word.items(), key=lambda x: x[1], reverse=True)
