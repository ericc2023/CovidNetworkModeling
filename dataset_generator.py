from main import makeModel, simulate, findResults
import warnings
import csv
import numpy as np



def makeRow(a, b):
  ret = []
  ret += [a]
  for k in range(np.size(b)):
    ret.append(float(b[k]))
  return ret

name = 'data_1k_to_5k_pts'
path = open('/Users/EricChen/PycharmProjects/COVIDModeling/' + name + '.csv', 'w')
writer = csv.writer(path)
warnings.filterwarnings("ignore", category = FutureWarning)

#Making the header
header = ['Maximum Infected']
for i in range(1,101):
  header.append('Eigenvalue ' + str(i))
writer.writerow(header)

#Doing the simulations and recording
reps = 10
for k in range(1,reps + 1):
  print(k)
  for fl in range(1,6):
    for stanford in range(1,6):
      print(str(k) + '.' + str(fl) + '.' + str(stanford))
      makeModel(numpts = 5000*stanford, numblocks = 10, intra = 0.1, inter = 0.0001*fl)
      simulate()
      a,b = findResults()
      writer.writerow(makeRow(a,b))
      print('Maximum is ' + str(a))
path.close()