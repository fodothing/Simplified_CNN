import matplotlib.pyplot as plt
##################################################
fig, ax = plt.subplots()
plt.figure(1)
##################################################
fig_path = "D:\SoftPackage\pycharm\PyCharmProject\MachineLearning\Plot_Datasets_information"
##################################################
x = ['SST-5','SST-2','MR1.0','MR2.0','IMDB','Yelp-5','Yelp-2','Amaz-5','Amaz-2']
avg_length = [18,19,20,353,294,155,153,93,91]
##################################################
plt.plot(x, avg_length, c = 'b')

plt.title('Avg length')
plt.xlabel('Datasets name')
plt.ylabel('Quantity')
##################################################
# plt.legend()  # 显示数据的线条名称
plt.show()
fig.savefig(fig_path+'\\datasets_avg_length.png',dpi=900,format='png')
