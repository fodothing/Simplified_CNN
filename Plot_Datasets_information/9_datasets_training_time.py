import matplotlib.pyplot as plt
##################################################
fig, ax = plt.subplots()
plt.figure(1)
##################################################
fig_path = "D:\SoftPackage\pycharm\PyCharmProject\MachineLearning\Plot_Datasets_information"
##################################################
x = ['MR2.0','MR1.0','SST-5','SST-2','IMDB', 'Yelp-5','Yelp-2','Amaz-5','Amaz-2']

our_model = [0.06,0.03,0.02,0.03,0.9, 1.8,1.4,6.9,8.7]
##################################################
plt.plot(x, our_model,'m^-',)
# 显示数据的值
for a,b in zip(x,our_model):
    plt.text(a,b, b)
##################################################
plt.xlabel('Datasets name')
plt.ylabel('Training time(h)')
##################################################

# plt.legend()  # 显示数据的线条名称
plt.show()
fig.savefig(fig_path+'\\9_datasets_training_time.png',dpi=900,format='png')
