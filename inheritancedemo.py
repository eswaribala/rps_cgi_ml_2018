# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:51:48 2018

@author: Balasubramaniam
"""
from oopsdemo import Customer
class PrivilegedCustomer(Customer):
    def __init__(self,id,name):
        Customer.__init__(self,id,name)
        self.offer=0.5
        
privCustomer=PrivilegedCustomer(428568476,"Anoop")
print(privCustomer.getCustomerId())