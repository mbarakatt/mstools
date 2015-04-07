import numpy as np
import sys
import os
from itertools import product
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt

NB_DEMES = 25
TOTAL_SAMPLE = 2000



# def migration_matrix_migrate_nbh(M):
# 	for i in range(len(M)):
# 		for j in range(len(M)):

migration_matrix = np.zeros((NB_DEMES, NB_DEMES))


# print migration_matrix_to_cmd_line(migration_matrix)


ref_matrix = np.arange(NB_DEMES).reshape((5,5))

d = {}
for i,p in enumerate(product(range(np.array(5)), range(np.array(5)))):
	d[i] = p



for i,j in product(range(np.array(NB_DEMES)), range(np.array(NB_DEMES))):
	coor1 = d[i]
	coor2 = d[j]
	if np.absolute(coor1[0] - coor2[0]) + np.absolute(coor1[1] - coor2[1]) == 1:
		migration_matrix[j,i] = 1.0
		print j,i, coor1, coor2

# print migration_matrix
# print migration_matrix_to_cmd_line(migration_matrix)


def v_color():
    return '#ffcc33'


def plot_point(ax, p):
    x, y = p
    ax.plot(x, y, 'o', color='#999999', zorder=1)

# def plot_bounds(ax, ob):
#     x, y = zip(*list((p.x, p.y) for p in ob.boundary))
#     ax.plot(x, y, 'o', color='#000000', zorder=1)


def plot_line(ax, p1, p2):
    x, y = [p1[0], p2[0]], [p1[1], p2[1]]
    ax.plot(x, y, color=v_color(), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


def draw_grid(ax, pos, migration_matrix):
	points = [ plot_point(ax, p)  for p in pos ]
	for i in range(len(migration_matrix)):
		for j in range(i, len(migration_matrix)):
			if migration_matrix[i,j] > 0 :
				plot_line(ax, d[i], d[j])



fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.5)
# draw_grid(ax, np.array([d[i] for i in range(NB_DEMES)]), migration_matrix)
# plt.show()

def arg_sampling_number_per_demes():
	output_list = [TOTAL_SAMPLE/25] * 25
	return '-I %d ' % len(output_list) +  ' '.join(map(str,output_list))


def arg_migration_matrix(M):
	output_list = []
	for i in range(len(M)):
		for j in range(len(M)):
			output_list.append(str(M[i,j]) if i != j else 'x' )

	return '-ma ' + ' '.join(map(str,output_list))


def arg_create():
	args = []
	args.append('-t 13.0')  # mutation rate
	args.append('-r 100.0 2501')  # recombination
	args.append(arg_sampling_number_per_demes())
	args.append(arg_migration_matrix(migration_matrix))

	f = open('c.ms', 'w')
	for l in args:
		f.write('%s\n' % l)
	f.close()
	f = open('markers.txt', 'w')
	markers = reduce(lambda x, y: x+y, [[i] * 80 for i in range(25)] )
	for m in markers:
		f.write('%d\n' % m )
	f.close()





arg_create()
