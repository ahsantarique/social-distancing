'''manipulate strategy vector'''

PRO = 1
ANTI = 0

def make_antivax(strategy, s):
  for i in s:
    strategy[i] = ANTI
  return strategy


def make_provax(strategy, s):
  for i in s:
    strategy[i] = PRO

  return strategy

def is_subset(lstANTI, lst2):

  for i in lstANTI:
    if( i not in lst2):
      # print(i)
      return False

  return True

def get_antivax(strategyNE):
  s = []
  for i in strategyNE.keys():
    if (strategyNE[i] == ANTI):
      # print(i)
      s.append(i)
  
  return s


def get_provax(strategyNE):
  s = []
  for i in strategyNE.keys():
    if (strategyNE[i] == PRO):
      # print(i)
      s.append(i)
  
  return s