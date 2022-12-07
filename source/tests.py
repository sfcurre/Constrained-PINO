import numpy as np
from data.dynamic_graph import DynamicGraph
from data.periodic_graph import PeriodicGraph
from data.minibatch_graph import MiniBatchGraph

ARANGE = np.arange(0, 5)
GRAPH_1D = np.stack(np.meshgrid(ARANGE, [0], [0], indexing='ij'), axis=-1)
GRAPH_2D = np.stack(np.meshgrid(ARANGE, ARANGE, [0], indexing='ij'), axis=-1)
GRAPH_3D = np.stack(np.meshgrid(ARANGE, ARANGE, ARANGE, indexing='ij'), axis=-1)

# test points
DYNAMIC_1D = {(2, 0, 0): {(1, 0, 0),
                          (3, 0, 0)},
              (0, 0, 0): {(1, 0, 0)},
              (4, 0, 0): {(3, 0, 0)}}
PERIODIC_1D = {(2, 0, 0): {(1, 0, 0),
                          (3, 0, 0)},
               (0, 0, 0): {(1, 0, 0),
                           (4, 0, 0)},
               (4, 0, 0): {(3, 0, 0),
                           (0, 0, 0)}}

DYNAMIC_2D = {(2, 3, 0): {(1, 3, 0),
                          (2, 2, 0),
                          (2, 4, 0),
                          (3, 3, 0)},
              (0, 0, 0): {(1, 0, 0),
                          (0, 1, 0)},
              (4, 4, 0): {(3, 4, 0),
                          (4, 3, 0)}}  
PERIODIC_2D = {(2, 3, 0): {(1, 3, 0),
                           (2, 2, 0),
                           (2, 4, 0),
                           (3, 3, 0)},
               (0, 0, 0): {(1, 0, 0),
                           (4, 0, 0),
                           (0, 4, 0),
                           (0, 1, 0)},
               (4, 4, 0): {(3, 4, 0),
                           (4, 0, 0),
                           (0, 4, 0),
                           (4, 3, 0)}}

DYNAMIC_3D = {(2, 3, 1): {(1, 3, 1),
                          (2, 2, 1),
                          (2, 4, 1),
                          (2, 3, 0),
                          (2, 3, 2),
                          (3, 3, 1)},
              (0, 0, 0): {(1, 0, 0),
                          (0, 1, 0),
                          (0, 0, 1)},
              (4, 4, 4): {(3, 4, 4),
                          (4, 3, 4),
                          (4, 4, 3)}}  
PERIODIC_3D = {(2, 3, 1): {(1, 3, 1),
                          (2, 2, 1),
                          (2, 4, 1),
                          (2, 3, 0),
                          (2, 3, 2),
                          (3, 3, 1)},
              (0, 0, 0): {(1, 0, 0),
                          (0, 1, 0),
                          (0, 0, 1),
                          (4, 0, 0),
                          (0, 4, 0),
                          (0, 0, 4)},
              (4, 4, 4): {(3, 4, 4),
                          (4, 3, 4),
                          (4, 4, 3),
                          (0, 4, 4),
                          (4, 0, 4),
                          (4, 4, 0)}}

MINIBATCH_SAMPLE_LENGTHS = {1: 7 * 125, 2: 7 * 27, 3: 7 * 8, 4: 7 * 8, 5: 7}

# tests
def test_graph_structure(constructor, graph, test_points):
    d = constructor(graph, np.stack([graph] * 10, axis=0))
    #index into time returns adj, locations, features, targets
    for point, true_neighbors in test_points.items():
        true_neighbors.add(point)
        location = np.argmax((d[0][1].detach().numpy() == point).all(axis=-1))
        #index order is (time), (adj or locations), (adj list), (adj list index)
        neighbors = d[0][1][d[0][0][0][0][d[0][0][0][1] == location]]
        alg_neighbors = set()
        for neighbor in neighbors:
            alg_neighbors.add(tuple(neighbor.detach().numpy().astype(int)))
        
        neighbors = d[0][2][d[0][0][0][0][d[0][0][0][1] == location]]
        feat_neighbors = set()
        for neighbor in neighbors:
            feat_neighbors.add(tuple(neighbor.detach().numpy().astype(int)))
        
        assert alg_neighbors == feat_neighbors

        if not(alg_neighbors == true_neighbors):
            print(f"Neighbors mismatch (point = {point})")
            print("True Neighbors:", true_neighbors)
            print("Alg Neighbors:", alg_neighbors)
            print("Set Difference:", true_neighbors.symmetric_difference((alg_neighbors)))
        else:
            print('Neighbors match - TEST PASSED')

def test_periodic_graph_structure(constructor, graph, test_points):
    d = constructor(graph, np.stack([graph] * 10, axis=0))
    #index into time returns adj, locations, features, targets
    for point, true_neighbors in test_points.items():
        true_neighbors.add(point)
        const_scaler = np.pi * 0.4
        locs = d[0][1].detach().numpy() / const_scaler
        location = np.argmax((locs == point).all(axis=-1))

        #index order is (time), (adj or locations), (adj list), (adj list index)
        neighbors = locs[d[0][0][0][0][d[0][0][0][1] == location]]
        alg_neighbors = set()
        for neighbor in neighbors:
            alg_neighbors.add(tuple(neighbor.astype(int)))
        
        neighbors = d[0][2][d[0][0][0][0][d[0][0][0][1] == location]]
        feat_neighbors = set()
        for neighbor in neighbors:
            feat_neighbors.add(tuple(neighbor.detach().numpy().astype(int)))
        
        assert alg_neighbors == feat_neighbors    

        if not(alg_neighbors == true_neighbors):
            print(f"Neighbors mismatch (point = {point})")
            print("True Neighbors:", true_neighbors)
            print("Alg Neighbors:", alg_neighbors)
            print("Set Difference:", true_neighbors.symmetric_difference((alg_neighbors)))
        else:
            print('Neighbors match - TEST PASSED')

def test_minibatch_graph_structure(constructor, graph, test_points, width, depth):
    d = constructor(graph, np.stack([graph] * 10, axis=0), width, depth)
    #index into time returns adj, locations, features, targets
    for point, true_neighbors in test_points.items():
        scaled_point = (point[0] // width, point[1] // width, point[2] // width)
        dim = 5 // width + bool(5 % width)
        i = np.ravel_multi_index(scaled_point, dims=(dim, dim, dim))
        true_neighbors.add(point)
        const_scaler = np.pi * 0.4
        locs = d[i][1].detach().numpy() / const_scaler
        location = np.argmax((locs == point).all(axis=-1))

        #index order is (time), (adj or locations), (adj list), (adj list index)
        neighbors = locs[d[i][0][0][0][d[i][0][0][1] == location]]
        alg_neighbors = set()
        for neighbor in neighbors:
            alg_neighbors.add(tuple(neighbor.astype(int)))
        
        neighbors = d[i][2][d[i][0][0][0][d[i][0][0][1] == location]]
        feat_neighbors = set()
        for neighbor in neighbors:
            feat_neighbors.add(tuple(neighbor.detach().numpy().astype(int)))
        
        assert alg_neighbors == feat_neighbors    

        if not(alg_neighbors == true_neighbors):
            print(f"Neighbors mismatch (point = {point})")
            print("True Neighbors:", true_neighbors)
            print("Alg Neighbors:", alg_neighbors)
            print("Set Difference:", true_neighbors.symmetric_difference((alg_neighbors)))
        else:
            print('Neighbors match - TEST PASSED')

def test_minibatch_sampling_length(constructor, graph, test_points):
    for width, true_length in test_points.items():
        d = constructor(graph, np.stack([graph] * 10, axis=0), width, 1)
        if not len(d) == true_length:
            print(f"Sampling lengths mismatch (width = {width})")
            print(f"True Length: {true_length}")
            print(f"Alg Length: {len(d)}")
        else:
            print('Lengths match - TEST PASSED')

#===========================================================
def main():
    print('---DYNAMIC---')
    test_graph_structure(DynamicGraph, GRAPH_1D, DYNAMIC_1D)
    test_graph_structure(DynamicGraph, GRAPH_2D, DYNAMIC_2D)
    test_graph_structure(DynamicGraph, GRAPH_3D, DYNAMIC_3D)
 
    print('---PERIODIC---')
    test_periodic_graph_structure(PeriodicGraph, GRAPH_1D, PERIODIC_1D)
    test_periodic_graph_structure(PeriodicGraph, GRAPH_2D, PERIODIC_2D)
    test_periodic_graph_structure(PeriodicGraph, GRAPH_3D, PERIODIC_3D)

    print('---MINIBATCH---')
    # MiniBatch only works in 3D
    test_minibatch_graph_structure(MiniBatchGraph, GRAPH_3D, PERIODIC_3D, 2, 1)
    test_minibatch_graph_structure(MiniBatchGraph, GRAPH_3D, PERIODIC_3D, 1, 2)
    test_minibatch_sampling_length(MiniBatchGraph, GRAPH_3D, MINIBATCH_SAMPLE_LENGTHS)

if __name__ == '__main__':
    main()