# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:45:09 2018

@author: Balasubramaniam
"""

customerInfo={"customerId":37473,"customerName":"HCL"}
#extract keys
print(customerInfo.keys())
#extract values
print(customerInfo.values())

for (key,value) in customerInfo.items():
     print(key,'-->',value)
     
#Array of customers
customerInfo=[{"customerId":37473,"customerName":"HCL"},
              {"customerId":37474,"customerName":"CGI"},
              {"customerId":37475,"customerName":"CTS"},
              {"customerId":37476,"customerName":"TCS"}
              ]     
     
     
for _ in customerInfo:
    
    for (key,val) in _.items():
        print(key,"-->",val)
     
     
     
     
     
     
     