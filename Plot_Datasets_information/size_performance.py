import matplotlib.pyplot as plt
##################################################
fig, ax = plt.subplots()
plt.figure(1)
##################################################
fig_path = "D:\SoftPackage\pycharm\PyCharmProject\MachineLearning\Plot_Datasets_information"
##################################################
x = ['MR2.0','MR1.0','SST-5','SST-2','IMDB','Yelp-5','Yelp-2','Amaz-5','Amaz-2']

our_model = [0.795,0.7355,0.3662,0.7600,0.8694, 0.5827,0.9194,0.6591,0.9468]
char_cnn = [None,None,None,None,None, 0.6205,0.9512,0.5946,0.9449]
bert = [None,None,None,0.949,0.9549, 0.7068,0.9811,0.6583,0.9737]
xlnet = [None,None,None,0.97,0.9621, 0.722,0.9845,0.6774,0.9789]
sota = [None,0.838,0.562,0.97,0.9621, 0.722,0.9845,0.6774,0.9789]
##################################################
plt.plot(x, our_model,'x--', label='Simplified CNN')
plt.plot(x, char_cnn,'m^-', label='CharCNN')
plt.plot(x, bert,'k<-.',label="BERT")
plt.plot(x, xlnet,'p:',label="XLNet")
plt.plot(x, sota,'x:',label="SOTA")


plt.xlabel('Datasets name')
plt.ylabel('Accuracy')
##################################################
plt.legend()  # 显示数据的线条名称
plt.show()
fig.savefig(fig_path+'\\size_performance.png',dpi=900,format='png')
