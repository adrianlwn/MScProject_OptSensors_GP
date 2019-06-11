import numpy as np

import sys
sys.path.append('/usr/bin/fluidity')
import vtktools
ug=vtktools.vtu('LSBU_20.vtu')
ug.GetFieldNames()

#read the values of the tracers and copy in a vector named p
p=ug.GetScalarField('Tracer')

n=len(p)





#how to write a vector named x in a vtu file

ug.AddScalarField('results', x)
ug.Write('Experimental_results.vtu')




