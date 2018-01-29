# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:46:04 2018

@author: Balasubramaniam
"""

name='CGI'
print(name.center(len(name)+20,'*'))
amount="345632"
print(amount.rjust(len(amount)+10,'*'))
print(name.ljust(len(name)+10,'*'))

#slicing
organization='CGI Technologies,  DLF, Ramapuram'
print(organization[:-1])

#print vertically
for _ in organization[:4]:
    print(_)

#print tab space
for _ in organization[4:8]:
    print(_,end='\t')
    
    
    
    
    
    
    
    
    







