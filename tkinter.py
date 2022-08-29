
# import tkinter
# top = tkinter.Tk()
# #
# #
# top.mainloop()


# import socket

# s= socket.socket()
# host= socket.gethostname()
# port= 12345

# s.connect((host.port))
# print (s.recv(1024))
# s.close()


# import numpy as np

# import matplotlib.pyplot as plt 

# x= np.arange(0,3 *np.pi, 0.1)

# y= np.sin(x)

# plt.title("sine wave form ")


# plt.plot(x,y)
# plt.show()

# import pandas as pd 
# import numpy as np 
# import panel as pn

# data={'item1': pd.DataFrame(np.random.randn(4,3)),
# 'item2': pd.DataFrame(np.random.randn(4,2))}

# p= pn.panel(data)


# import numpy as np

# a= np.array([[1,2],[3,4]])
# print(a)

# import numpy as np
# a= np.array([1,2,3,4,5], ndmin=2)
# print(a)

# import numpy as np
# a= np.array([1,2,3,4,5], dtype=complex)
# print(a)

# import pandas as pd 
# import numpy as np 
# data = np.array(['a','b','c','d'])
# s= pd.Series(data)
# print(s)

# import pandas as pd
# import numpy as np

# df= pd.DataFrame(np.random.randn(5,3), 
# index=['a','c','e','f','h'], 
# columns = ['one','two','three'])

# df= df.reindex(['a','b','c','d','e','f','g','h'])

# print(df)

# import pandas as pd
# import numpy as np

# df= pd.DataFrame(np.random.randn(5,3), 
# index=['a','c','e','f','h'], 
# columns = ['one','two','three'])

# df= df.reindex(['a','b','c','d','e','f','g','h'])

# print(df['one'].isnull())

# import pandas as pd
# import numpy as np

# df= pd.DataFrame(np.random.randn(5,3), 
# index=['a','c','e','f','h'], 
# columns = ['one','two','three'])

# df= df.reindex(['a','b','c','d','e','f','g','h'])

# print(df)

# print("NaN replaced with'0.' ")

# print(df.fillna(0))



# import tkinter as tk
# root = tk.Tk()
# root.title("My GUI")
# #
# label = tk.Label(root, text="hello world")
# #
# label.pack()

# root.mainloop()

# import pandas as pd
# import numpy as np

# df= pd.DataFrame(np.random.randn(5,3), 
# index=['a','c','e','f','h'], 
# columns = ['one','two','three'])

# df= df.reindex(['a','b','c','d','e','f','g','h'])

# print(df)

# print("replace NanN with 'o.' ")

# print(df['one'].fillna(0))


# import pandas as pd 
# import numpy as np 

# data={'salary':[4600,129128,2412,31412,214], 'Hours Worked':[4,2,5,4,3]}

# df= pd.DataFrame(data,
# index=['one','two','three','four','five'])


# print(df)


# import pandas as pd 
# import numpy as np 

# data={'salary':[4600,129128,2412,31412,214], 'Hours Worked':[4,2,5,4,3]}

# index=('one','two','three','four','five')

# df= pd.DataFrame(data, index)


# print(df)

# import pandas as pd 
# import numpy as np

# import matplotlib.pyplot as plt 

# Hours_worked= 4,2,5,4,


# data={'salary':[4600,129128,2412,31412,214], 'Hours Worked':[4,2,5,4,3]}

# names=('ope','prince','gbemi','gbenga','bisi')

# df= pd.DataFrame(data, names)

# plt.plot(names,Hours_worked)
# plt.show()

# print(df)


# import pandas as pd 
# import numpy as np

# import matplotlib.pyplot as plt 

# Hours_worked= 4,2,5,4,3

# salary= 6,7,12,3,10

# data={'salary':[4600,129128,2412,31412,214], 'Hours Worked':[4,2,5,4,3]}

# names=('ope','prince','gbemi','gbenga','bisi')

# df= pd.DataFrame(data, names)

# plt.plot(names,Hours_worked)
# plt.xlabel("names of workers")
# plt.ylabel("Hours_worked")
# plt.show()

# print(df)


# import pandas as pd 
# import numpy as np

# import matplotlib.pyplot as plt 

# Hours_worked= 4,2,5,4,3

# salary= 6,7,12,3,10

# No_of_times_absent =1,4,2,3,5

# data={'salary':[4600,129128,2412,31412,214], 'Hours Worked':[4,2,5,4,3], 'No of times absent': [1,4,2,3,5]}

# names=('ope','prince','gbemi','gbenga','bisi')

# df= pd.DataFrame(data, names)

# plt.plot(names,salary)
# plt.xlabel("names of workers")
# plt.ylabel("salary(Thousand Naira)")
# plt.show()

# print(df)

# import pandas as pd 
# import numpy as np

# import matplotlib.pyplot as plt 

# Hours_worked= 4,2,5,4,3

# salary= 6,15,7,3,10

# No_of_times_absent =1,4,2,3,5

# data={'salary':[4600,129128,2412,31412,214], 'Hours Worked':[4,2,5,4,3], 'No of times absent': [1,4,2,3,5]}

# names=('ope','prince','gbemi','gbenga','bisi')

# df= pd.DataFrame(data, names)

# plt.plot(names,salary,No_of_times_absent,Hours_worked)
# plt.xlabel("names of workers")
# plt.ylabel("salary(Thousand Naira)")
# plt.show()

# print(df)

# import pandas as pd 
# import numpy as np
# import matplotlib.pyplot as plt 

# data= {'name':['shyllon', 'dennis', 'ofe', 'fola'], 'class':['ss3a','ss3a','ss3a','ss3a']}

# score=(95, 75,89,72)
# index= ('1','2','3','4')

# df= pd.DataFrame(data, index)

# plt.plot(index, score)
# plt.xlabel("students")
# plt.ylabel("Exam scores")

# print(df)


# import pandas as pd 
# import numpy as np 
# df= pd.DataFrame(np.random.randn(5,3),
# index=['a','c','e','f','h'], columns=['one','two','three'])

# df=df.reindex(['a','b','c','d','e','f','g','h'])

# print(df.fillna(method='pad'))


import pandas as pd 
import numpy as np 
df= pd.DataFrame(np.random.randn(5,3),
index=['a','c','e','f','h'], columns=['one','two','three'])

df=df.reindex(['a','b','c','d','e','f','g','h'])

print(df.dropna())

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
df= pd.DataFrame(np.random.randn(5,3),
index=['a','c','e','f','h'], columns=['one','two','three'])

df=df.reindex(['a','b','c','d','e','f','g','h'])

print(df.dropna())

plt.plot(df)

import pandas as pd 

data=  pd.read_csv('myfirstcsvfile.txt')

data=pd.DataFrame(data)
print(data)

import pandas as pd 

data=  pd.read_json('myfirstjsonfile.json')

data=pd.DataFrame(data)
print(data)

#importing data into and from an sql database which for some reason isnt working 
# import pandas as pd 
# from sqlalchemy import create_engine

# data=  pd.read_csv('myfirstcsvfile.txt')
# data=pd.DataFrame(data)

# engine= create_engine('sqlite:///:tkinter folder:')

# data.to_sql('data_table', engine)
# res1 = pd.read_sql_query('SELECT * FROM data_table',engine)
# print('Resullt 1')