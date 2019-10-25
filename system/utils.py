import openpyxl
from decimal import Decimal
# from math import pow, isnan, exp, log, log10, floor
# import numpy as np
# from scripy.stats import norm
# from statistics import mean, stdev
# from oct2py import octave
from loop.models import Loop


class Reader:
    def __init__(self, system):
        self.system = system

    @staticmethod
    def format_value(value):
        if isinstance(value, str):
            return Decimal(value.replace(',', '.'))
        return Decimal(value)

    def read(self):
        path = self.system.filename
        workbook = openpyxl.load_workbook(path)
        worksheet = workbook.active
        if worksheet.max_column % 4 != 0:
            return {'ok': False, 'message': 'Wrong number of columns.'}

        loops_number = worksheet.max_column // 4
        rows_number = worksheet.max_row
        loops = []

        required_columns = {'mv', 'ma', 'spa', 'cv'}
        for i in range(0, loops_number):
            columns = []
            loop = {}
            for j in range(1, 5):
                column = worksheet.cell(row=1, column=j+i*4).value
                columns.append(column)
                if column.split(':')[1] in required_columns:
                    loop[column.split(':')[1]] = [
                        self.format_value(worksheet.cell(row=iterator, column=j+i*4).value) for iterator in
                        range(3, rows_number+1)
                    ]
            loop_name = [column.split(':')[0] for column in columns]
            if len(set(loop_name)) > 1:
                return {'ok': False, 'message': 'Incorrect column names!'}
            loop['name'] = loop_name[0]
            loops.append(loop)

        return {'ok': True, 'loops': loops}


class Assessment:
    def __init__(self, id):
        self.loop = Loop.objects.get(pk=id)
        self.err = [cv-spa for cv in self.loop.cv for spa in self.loop.spa]
        print(self.err)

    # def stat_indexes(self):
    #     octave.addpath('E:\\Documents\\Praca magisterska\\quality-assessment\\qualityassessment\\matlab\\FitFunc')
    #     octave.addpath('E:\\Documents\\Praca magisterska\\quality-assessment\\qualityassessment\\matlab\\stbl')
    #     octave.addpath('E:\\Documents\\Praca magisterska\\quality-assessment\\qualityassessment\\matlab\\matlab\\LIBRA')
    #     octave.eval('pkg load statistics')
    #
    #     nbins = 140
    #     d_std = 3.0
    #     mu = mean(self.err)
    #     self.mu = mu
    #     st = stdev(self.err)
    #     self.st = st