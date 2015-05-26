#!/usr/bin/env python
import numpy as np
import sys


f = sys.stdin

while True:
	l1, l2 = f.readline()[:-1], f.readline()[:-1]
	if l1 == '' or l2 == '':
		break
	print ''.join([str(c) for c in np.array([int(i) for i in l1]) + np.array([int(i) for i in l2])])




