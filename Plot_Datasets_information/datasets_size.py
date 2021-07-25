import matplotlib.pyplot as plt
##################################################
fig, ax = plt.subplots()
plt.figure(1)
##################################################
fig_path = "D:\SoftPackage\pycharm\PyCharmProject\MachineLearning\Plot_Datasets_information"
##################################################
x = ['SST-5','SST-2','MR1.0','MR2.0','IMDB','Yelp-5','Yelp-2','Amaz-5','Amaz-2']
size = [11855,9613,10622,2000,50000,700000,598000,3650000,4000000]
##################################################
plt.plot(x, size, c = 'r')

plt.title('Size')
plt.xlabel('Datasets name')
plt.ylabel('Quantity')
##################################################
# plt.legend()  # 显示数据的线条名称
plt.show()
fig.savefig(fig_path+'\\datasets_size.png',dpi=900,format='png')
