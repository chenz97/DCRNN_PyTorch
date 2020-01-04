import pandas as pd
import csv

# original file: a csv containing a (#station, #station) matrix

def gen_dist_file(file, out_file):
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        matrix = pd.read_csv(file, header=None)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                writer.writerow([i, j, matrix[i][j]])

if __name__ == '__main__':
    gen_dist_file('distance.csv', 'distances_d7.csv')