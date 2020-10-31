# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 00:11:45 2020

@author: kumarnvn
"""

import cv2
import pytesseract
import os
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction import text

from sklearn.naive_bayes import MultinomialNB

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def p_image_2_text(fullpath):
    Img = cv2.imread(fullpath)
    Img = cv2.resize(Img,None,fx=3.5,fy=3.5 ,interpolation=cv2.INTER_CUBIC)
    Img = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    text1 = pytesseract.image_to_string(Img,lang='eng')
    text1 = text1.replace("\n"," ")
    text1 = " ".join(re.findall("[a-zA-Z]*",text1))
    return text1


def image_2_text(input_path,out_path):
    print("Image to text conversion started......")
    for ImgName in os.listdir(input_path):
        fullpath = os.path.join(input_path,ImgName)
        Img = cv2.imread(fullpath)
        text = pytesseract.image_to_string(Img,lang='eng')
        text = text.replace("\n"," ")
        text = " ".join(re.findall("[a-zA-Z]*",text))
        if len(text) < 1 :
            text = p_image_2_text(fullpath)
        
        file_append = open(out_path,'a+')
        file_append.write(ImgName+'\t')
        file_append.write(text+'\n')
        file_append.close()
    print("Process Complete")

#  Real data 
neg_path = r"G:\hackerearth\Detecting_sentiments_of_quote\Data Files\Train\Negative"
neg_file = "G:\\hackerearth\\Detecting_sentiments_of_quote\\Data Files\\Textnegative.txt"

pos_path = "G:\\hackerearth\\Detecting_sentiments_of_quote\\Data Files\\Train\\Positive"
pos_file = "G:\\hackerearth\\Detecting_sentiments_of_quote\\Data Files\\Textpositive.txt"

ran_path = "G:\\hackerearth\\Detecting_sentiments_of_quote\\Data Files\\Train\\Random"
ran_file = "G:\\hackerearth\\Detecting_sentiments_of_quote\\Data Files\\Textrandom.txt"

image_2_text(neg_path,neg_file)
image_2_text(pos_path,pos_file)
image_2_text(ran_path,ran_file)

neg = pd.read_csv(neg_file,sep='\t',encoding='ISO-8859-1',names=['file_name' , 'Text'])
pos = pd.read_csv(pos_file,sep='\t',encoding='ISO-8859-1',names=['file_name' , 'Text'])
ran = pd.read_csv(ran_file,sep='\t',encoding='ISO-8859-1',names=['file_name' , 'Text'])

neg['target'] = 0
pos['target'] = 1
ran['target'] = 2

data = pd.concat([neg,pos,ran],ignore_index=True)
#data.loc[(data['Text'].isnull() == True) & (data['target'] == 2)]['Text'] = 'AA'

d = data.drop([39,53,54,55,56,58,59])

d.fillna('AA',inplace=True)

Count_vec = text.CountVectorizer(decode_error='ignore')

X = Count_vec.fit_transform(d['Text'])
Y = d['target'].as_matrix()

# Training the model
NB_clg  = MultinomialNB()
NB_clg.fit(X,Y)

Score = NB_clg.score(X,Y)



# Prediction with given dataset
test_path = r"G:\hackerearth\Detecting_sentiments_of_quote\Data Files\Dataset"
test_file = r"G:\hackerearth\Detecting_sentiments_of_quote\Data Files\prediction.txt"

image_2_text(test_path,test_file)

test = pd.read_csv(test_file,sep='\t',encoding= 'ISO-8859-1',names=['Filename','Text'])
test.fillna('AA',inplace=True)
x_text = Count_vec.transform(test['Text'])

y_pred = NB_clg.predict(x_text)
y_pred = pd.DataFrame(y_pred,columns=['Category'])
submit = pd.concat([test['Filename'],y_pred],axis=1,names=['Filename','Category'])
submit.rename(columns={'file_name': 'Filename',0: 'Category'},inplace=True)

submit['Category'] = submit['Category'].apply(lambda x: "Positive" if (x ==1) else ('Negative' if (x == 0) else 'Random'))

submit.to_csv('submit3.csv',index=False)

out = pd.concat([test,y_pred],axis=1)
out.rename(columns={'file_name': 'Filename',0: 'Category'},inplace=True)