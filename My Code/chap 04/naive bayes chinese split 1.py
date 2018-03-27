"""
函数说明：切分中文语句

"""
import os
import jieba
import random

def TextProcessing(folder_path,test_size = 0.2):
    #查看folder_path下的文件
    folder_list=os.listdir(folder_path)
    #训练集
    data_list=[]
    class_list=[]

    #遍历每个子文件夹
    for folder in folder_list:
        #根据子文件夹，生成新的路径
        new_folder_path=os.path.join(folder_path,folder)
        #存放子文件夹下的txt文件的列表
        files=os.listdir(new_folder_path)

        j=1
        #遍历每个txt文件
        for file in files:
            #每类txt样本数最多100个
            if j>100:
                break
            #打开txt文件
            with open(os.path.join(new_folder_path,file),'r',encoding='utf-8') as f:
                raw=f.read()
            #精简模式，返回一个可迭代的generator
            word_cut=jieba.cut(raw,cut_all=False)
            #generator转换为list
            word_list=list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)
            j+=1
        #zip压缩合并，将数据与标签对应压缩
        data_class_list=list(zip(data_list,class_list))
        #将data_class_list乱序
        random.shuffle(data_class_list)
        #训练集与测试集切分的索引值
        index=int(len(data_class_list)*test_size)+1
        #训练集
        train_list=data_class_list[index:]
        #测试集
        test_list=data_class_list[:index]
        #训练集解压缩
        train_data_list,train_class_list=zip(*train_list)
        #测试集解压缩
        test_data_list,test_class_list=zip(*test_list)
        #统计训练集词频
        all_words_dict={}
        for word_list in train_data_list:
            for word in word_list:
                if word in all_words_dict.keys():
                    all_words_dict[word]+=1
                else:
                    all_words_dict[word]=1

        #根据键值倒序排列
        all_words_tuple_list=sorted(all_words_dict.items(),key=lambda
            f:f[1],reverse=True)
        #解压缩
        all_words_list,all_words_nums=zip(*all_words_tuple_list)
        #转换成列表
        all_words_list=list(all_words_list)
        return all_words_list,train_data_list,test_data_list,train_class_list,\
               test_class_list

if __name__=='__main__':
    #文本预处理，训练集存放的地址
    folder_path='E:\python\machine learning in action\My Code\chap 04\SogouC\Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, \
    test_class_list=TextProcessing(folder_path,test_size=0.2)
    print(all_words_list)
