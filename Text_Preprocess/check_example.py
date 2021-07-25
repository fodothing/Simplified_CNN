from Text_Preprocess.TextPreprocess import *

doc = ['\" \n\n haved @ ... 🚂 a  \t \n \r amnbe   nice day', 'yeah, 💪:) () good afternoon', 'ac 123 /.// ..']
text_pro = Text_Preprocess()

doc = text_pro.preprocess(doc)

print(doc)
