# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:32:11 2017

@author: BALASUBRAMANIAM
"""

from openpyxl import Workbook
import calendar
filePath="F:/yatrabakup/SVSReports/AnnualReport2018.xlsx"

wb=Workbook()
i=0
for month in calendar.month_name:
    #print(month)
     if not(len(str(month))==0):
        print(month)
        wb.create_sheet(month+"_2018",i)
        i=i+1
        
wb.save(filePath)
        

