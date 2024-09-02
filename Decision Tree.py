#parameter Tunning
# by default criterion="gini", splitter="best", max_depth=None
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train)
print(y_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.9
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr

bias = classifier.score(X_train, y_train)
bias
# 0.996875
variance = classifier.score(X_train, y_train)
variance
# 0.9

variance =classifier.score(X_test,y_test)
variance
#----------------------------------****------------------------------****-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.9
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.996875
variance = classifier.score(X_test,y_test)
variance
# 0.9
#----------------------------------****------------------------------****-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="log_loss", splitter="best", max_depth=None)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.9
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.996875
variance = classifier.score(X_test,y_test)
variance
# 0.9
#----------------------------------****------------------------------****-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=None)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.8875
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.996875
variance = classifier.score(X_test,y_test)
variance
# 0.8875
#----------------------------------****------------------------------****-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=None)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.875
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.996875
variance = classifier.score(X_test,y_test)
variance
# 0.875
#----------------------------------****------------------------------****-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="log_loss", splitter="random", max_depth=None)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.875
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.996875
variance = classifier.score(X_test,y_test)
variance
# 0.875
#----------------------------------****------------------------------****-----------------------------
#Hyper parameter tunning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)

'''print(X_train.shape)
print(X_test.shape)
print(y_train)
print(y_test)'''

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=3)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.9125
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr

bias = classifier.score(X_train, y_train)
bias
# 0.815625
variance = classifier.score(X_test,y_test)
variance
# 0.9125
#--------------------------------****----------------------------------------------*****-----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.8375
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.78125
variance = classifier.score(X_test,y_test)
variance
#0.8325
#---------------------------****--------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.875
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
#  0.775
variance = classifier.score(X_test,y_test)
variance
# 0.875
#---------------------------------****-------------------------------------****----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.925
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.875
variance = classifier.score(X_test,y_test)
variance
#0.975
#-----------------------------------------***-------------------------------****-----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.8375
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.78125
variance = classifier.score(X_test,y_test)
variance
#0.8375
#-------------------------****------------------------------------------*****------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.909375
variance = classifier.score(X_test,y_test)
variance
# 0.95
#--------------------------------****----------------------------------------------*****---------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)

'''print(X_train.shape)
print(X_test.shape)
print(y_train)
print(y_test)'''

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=3)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.9125
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr

bias = classifier.score(X_train, y_train)
bias
# 0.815625
variance = classifier.score(X_test,y_test)
variance
# 0.9125
#--------------------------------****----------------------------------------------*****---------------
#-------------------------****------------------------------------------*****------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.909375
variance = classifier.score(X_test,y_test)
variance
# 0.95
#--------------------------------****----------------------------------------------*****---------------
#-------------------------****------------------------------------------*****------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.909375
variance = classifier.score(X_test,y_test)
variance
# 0.95
#--------------------------------****----------------------------------------------*****-----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="log_loss", splitter="best", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.909375
variance = classifier.scoreX_test,y_test)
variance
# 0.95
#--------------------------****---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.8625
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.80
variance = classifier.score(X_test,y_test)
variance
# 0.8625
#---------------------------------****-------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.90
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.8
variance = classifier.score(X_test,y_test)
variance
# 0.90
#---------------------------****--------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="log_loss", splitter="best", max_depth=2)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.909375
variance = classifier.score(X_test,y_test)
variance
# 0.95
#--------------------------------------****----------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=3)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.9125
variance = classifier.score(X_test,y_test)
variance
# 0.95
#---------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=3)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.909375
variance = classifier.score(X_test,y_test)
variance
# 0.95
#--------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=3)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.875
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.80625
variance = classifier.score(X_test,y_test)
variance
# 0.875
#--------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=3)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.909375
variance = classifier.score(X_test,y_test)
variance
# 0.95
#-------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="log_loss", splitter="best", max_depth=3)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
# 0.95
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr
bias = classifier.score(X_train, y_train)
bias
# 0.909375
variance = classifier.score(X_test,y_test)
variance
# 0.95
#-----------------------------------the end-------------------------------------------------
