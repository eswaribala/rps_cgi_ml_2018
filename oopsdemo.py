# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:10:11 2018

@author: Balasubramaniam
"""

class Customer:
    roi=0.8
    def __init__(self,id,name):
        self.__customerId=id
        self.__custsomerName=name
        
    def getCustomerId(self):
        return self.__customerId
    def getCustomerName(self):
        return self.__customerName
    
    def setCustomerName(self,newname):
        self.__customerName=newname        
    @staticmethod
    def getStaticData():
        return Customer.roi
'''
customerObj=Customer(428568476,"Anoop")
print(customerObj.getCustomerId())
customerObj.setCustomerName("Anju")
print(customerObj.getCustomerName())
print(Customer.getStaticData())
'''    