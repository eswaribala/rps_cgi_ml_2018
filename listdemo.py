# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:14:11 2018

@author: Balasubramaniam
"""

item_list=["CGI",1374574,True,"TCS",3624563,False]

for elem in item_list:
    if type(elem) == str: 
       print(elem)
       
#nested list
profileData=["Arun",["Java","C","C++","Python"],
             "Monika",["C#","Groovy"]
             ]
#count the skillsets
for _ in profileData:
        
    if type(_) is type(list):
       #count=0
       for elem in _:
           print(elem,end="\t")
           #count+=1
       #print(count)
    if type(_) is str:
         print(_)
         
#creating dynamic list
import random
randomList=[]
for _ in range(1,5):
    randomList.append(random.randint(1,1000))
print("\n",randomList)
#sort 
randomList.sort()
print("\n",randomList)
#reverse
randomList.reverse()
print("\n",randomList)

#concatenate list
list1=[45,67,89]
list2=["James","Jacob","Joy"]
print(list1+list2)
#zip
for (x,y) in zip(list1,list2):
    print(y,'-->',x)
#join
names=["Subha","Shobha","Shilaja"]
print("-->".join(names))

#replication
print(names*4) 

#tuples
data=(45,67,78,89)
data.append(100)
    


       
       
       
       
       
       
       
       
       

