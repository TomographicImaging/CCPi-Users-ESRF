# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#   Authored by:    Gemma Fardell (UKRI-STFC)

from cil.framework import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

class WeightDuplicateAngles(DataProcessor):

    """
    AcquisitionData -> AcquisitionData Processor
    Can be used to weight projections of any duplicate angles in a dataset for FBP
    Can be used to weight residuals of any duplicate angles in a dataset for iterative algorithms
    
    """
    def __init__(self):
        super().__init__()

    def check_input(self, dataset):
        return True
        #must be acquisition data

    def process(self, out=None):

        #get angles 
        data_in = self.get_input()

        # TODO wrap 360 or 180, degrees or radians
        angles_mod = np.mod(data_in.geometry.angles, 360)  

        #sort
        index_sorted = np.argsort(angles_mod)
        index_unsorted  = index_sorted.argsort()
        angles_sorted= angles_mod[index_sorted]

        #return an array of denominators based on number of duplicates
        denom = np.ones_like(angles_mod)

        #todo tolerance (step size) find or be passed
        num_ang = len(angles_mod)
        j=0
        while  j < num_ang-1:
            count = 1
            start_ind = j
            dif = abs(angles_sorted[j] - angles_sorted[j+1])
            while (dif < 0.06) and (j < num_ang - 2):
                count+=1
                j+=1
                dif = abs(angles_sorted[j] - angles_sorted[j+1])
            j += 1
            denom[start_ind:j] = count

        denom_weights= 1/denom[index_unsorted]

        #plt.plot(denom_weights)
        if out is None:
            out = data_in.geometry.allocate(None)
            for i in range(num_ang):
                out.array[i] = data_in.array[i] * denom_weights[i]
            return out

        else:
            for i in range(num_ang):
                out.array[i] = data_in.array[i] * denom_weights[i]
