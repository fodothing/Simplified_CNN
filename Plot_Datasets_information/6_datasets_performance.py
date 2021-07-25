import matplotlib.pyplot as plt
##################################################
fig, ax = plt.subplots()
plt.figure(1)
##################################################
fig_path = "D:\SoftPackage\pycharm\PyCharmProject\MachineLearning\Plot_Datasets_information"
##################################################
x_label = ['SST-2','IMDB','Yelp-5','Yelp-2','Amaz-5','Amaz-2']

our_model = [0.7600,0.8694, 0.5827,0.9194,0.6591,0.9468]
char_cnn = [0,0, 0.6205,0.9512,0.5946,0.9449]
bert = [0.949,0.9549, 0.7068,0.9811,0.6583,0.9737]
xlnet = [0.97,0.9621, 0.722,0.9845,0.6774,0.9789]
sota = [0.97,0.9621, 0.722,0.9845,0.6774,0.9789]
##################################################
x = list(range(len(x_label)))
total_width, n = 0.6, 5
width = total_width/n

plt.bar(x,char_cnn,width=width,label='CharCNN')

for i in range(len(x)):
    x[i] += width
plt.bar(x,our_model,width=width,label='Simplified CNN')

for i in range(len(x)):
    x[i] += width
plt.bar(x,bert,width=width,label='BERT',tick_label=x_label)

for i in range(len(x)):
    x[i] += width
plt.bar(x,xlnet,width=width,label='XLNet')

for i in range(len(x)):
    x[i] += width
plt.bar(x,sota,width=width,label='SOTA')

##################################################
plt.legend()  # 显示数据的线条名称
plt.xlabel('Datasets name')
plt.ylabel('Accuracy')
##################################################
plt.show()
fig.savefig(fig_path+'\\6_datasets_performance.png',dpi=900,format='png')


