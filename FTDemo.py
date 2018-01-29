# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:08:34 2018

@author: Balasubramaniam
"""
import sys
sys.path.append('./lib')
from functions import fundTransfer        
fromAccount={"accountNo":48254,"amount":5000}
toAccount={"accountNo":48254}

fundTransfer(fromAccount,toAccount)

x,y=5,6
print(x)