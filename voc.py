import os

data_path = './JPEGImages'


file_list = os.listdir(data_path)

with open('data.txt','w') as f:
    for f_name in range(file_list):
        w_str = '/JPEGImages/'+f_name+ './jpg'+'/SegmentationClass/'+f_name +'.png'+'\n'
        f.write(w_str)

