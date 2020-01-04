
if __name__ == '__main__':
    with open('../data/sensor_graph/graph_sensor_ids_d7.txt', 'w') as f:
        f.write('0')
        for i in range(1, 228):
            f.write(',' + str(i))
