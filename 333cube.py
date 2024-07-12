import numpy as np
from copy import deepcopy
from scipy import spatial

def generate_cube_seeds(number_cells_x, number_cells_y, number_cells_z, standard_deviation=0, spatial_step=1):
    grid_values = ([[(i + np.random.normal(0, standard_deviation)) * spatial_step,
                    (j + np.random.normal(0, standard_deviation)) * spatial_step,
                    (k + np.random.normal(0, standard_deviation)) * spatial_step]
                  for k in range(number_cells_z)
                  for j in range(number_cells_y)
                  for i in range(number_cells_x)])
    unique_grid_values = [list(x) for x in set(tuple(x) for x in grid_values)]
    return unique_grid_values

def get_vertex_number(vertex, vertices):
    if vertex in vertices.values():
        vertex_number = list(vertices.keys())[list(vertices.values()).index(vertex)]
    else:
        if len(vertices) > 0:
            vertex_number = max(vertices.keys()) + 1
        else:
            vertex_number = 1
        vertices[vertex_number] = vertex
    return vertex_number

def get_enum(edge, edges):
    if edge in edges.values():
        enum = list(edges.keys())[list(edges.values()).index(edge)]
    elif edge[::-1] in edges.values():
        enum = - list(edges.keys())[list(edges.values()).index(edge[::-1])]
    else:
        if len(edges) > 0:
            enum = max(edges.keys()) + 1
        else:
            enum = 1
        edges[enum] = [edge[0], edge[1]]
    return enum

def get_cell_area(cell_vertices, vertices):
    x = [vertices[i][0] for i in cell_vertices]
    y = [vertices[i][1] for i in cell_vertices]
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def remove_infinite_regions(regions, tessellation, max_distance=50):
    to_delete = []
    for c in regions:
        distances = []
        if len(c) != 0 and -1 not in c:
            for ii in range(0, len(c) - 1):
                distances.append(
                    np.linalg.norm(tessellation.vertices[c[ii]] - tessellation.vertices[c[ii + 1]]))
            distances = np.array(distances)
            if np.any(np.where(distances > max_distance, True, False)):
                to_delete.append(c)
    for c in to_delete:
        regions.remove(c)
    return regions

def get_cell_area_sign(cell, all_vertices):
    return int(np.sign(get_cell_area(cell, all_vertices)))

def create_lattice_elements(tessellation):
    new_vertices = {}
    new_cells = {}
    new_edges = {}
    regions = deepcopy(tessellation.regions)  # deepcopy wont affect the original
    ridge = deepcopy(tessellation.ridge_vertices)
    big_edge = []
    cnum = 1
    regions = remove_infinite_regions(regions, tessellation)
    # print(regions)
    for c in regions:
        temp_for_cell = []
        temp_vertex_for_cell = []
        if len(c) != 0 and -1 not in c:
            if c[0] != c[-1]:
                c.append(c[0])
            for ii in range(0, len(c) - 1):
                temp_big_edge = []
                x_coordinate = np.around(np.linspace(round(tessellation.vertices[c[ii]][0], 3),
                                                    round(tessellation.vertices[c[ii + 1]][0], 3), 2), 3)
                y_coordinate = np.around(np.linspace(round(tessellation.vertices[c[ii]][1], 3),
                                                    round(tessellation.vertices[c[ii + 1]][1], 3), 2),3)
                z_coordinate = np.around(np.linspace(round(tessellation.vertices[c[ii]][2], 3),
                                                    round(tessellation.vertices[c[ii + 1]][2], 3), 2), 3)
                new_edge_vertices = list(zip(x_coordinate, y_coordinate, z_coordinate))
                for v in range(0, len(new_edge_vertices) - 1):
                    v0 = new_edge_vertices[v]
                    v1 = new_edge_vertices[v + 1]
                    vertex_number_1 = get_vertex_number(v0, new_vertices)
                    vertex_number_2 = get_vertex_number(v1, new_vertices)
                    enum = get_enum([vertex_number_1, vertex_number_2], new_edges)
                    temp_big_edge.append(enum)
                    temp_for_cell.append(enum)
                    temp_vertex_for_cell.append(vertex_number_1)
                    temp_vertex_for_cell.append(vertex_number_2)
                big_edge.append(temp_big_edge)
            area_sign = get_cell_area_sign(temp_vertex_for_cell, new_vertices)
            new_cells[-1 * cnum * area_sign] = temp_for_cell
            cnum += 1
      # for i in tessellation.ridge_points
    return new_vertices, new_edges, new_cells,regions

def generate_voronoi_tessellation(seed_values: list) -> object:
  tessellation = spatial.Voronoi(list(seed_values))
  # print(tessellation.regions)
  print(tessellation.vertices)
  print(tessellation.ridge_vertices)
  print("point region")
  print(tessellation.vertices)
  # print(tessellation.point_region)
  return tessellation

def create_example_lattice(number_cells_x, number_cells_y, number_cells_z, voronoi_seeds_std=0.15, voronoi_seeds_step=20):
    seed_values = generate_cube_seeds(number_cells_x, number_cells_y, number_cells_z, standard_deviation=0, spatial_step=voronoi_seeds_step)
    tessellation = generate_voronoi_tessellation(seed_values)
    vertices, edges, cells,regions = create_lattice_elements(tessellation)
    return vertices, edges, cells, regions, tessellation



lattice = generate_cube_seeds(3,3,3)
# 生成并打印顶点、边和单元格
vertices, edges, cells, regions, tessellation = create_example_lattice(3,3,3,0,20)
print("Regions in main:", regions)
print("Vertices in main:", vertices)
print("Edges in main:", edges)
print("Cells in main:", cells)


faces = deepcopy(tessellation.ridge_vertices)
for i in tessellation.ridge_vertices: 
  if -1 in i:
    faces.remove(i)

# print("Faces in main:",faces)
print("Faces in main:",faces)


# def generate_edges_from_faces(faces):
#   edges = {}
#   edge_num = 1
#   for i in faces:
#     for ii in range(len(i)):
#       if ii == len(i) - 1:
#         v0 = i[ii]
#         v1 = i[0]
#       else:
#         v0 = i[ii]
#         v1 = i[ii + 1]
#       # v0和v1是顶点编号
#       if (v0, v1) not in edges.values() and (v1, v0) not in edges.values():
#         edges[edge_num] = (v0, v1)
#         edge_num += 1

#   return edges

# def generate_edges_from_faces(faces):
#   edges = {}
#   edge_num = 1
#   for i in faces:
#     for ii in range(len(i)-1):
#       v0 = i[ii]
#       v1 = i[ii + 1]
#       # v0和v1是顶点编号
#       if (v0, v1) not in edges.values() and (v1, v0) not in edges.values():
#         edges[edge_num] = (v0, v1)
#         edge_num += 1
#   return edges

def adjust_faces_indices(faces):
    adjusted_faces = []
    for face in faces:
        adjusted_face = [vertex + 1 for vertex in face]
        adjusted_faces.append(adjusted_face)
    return adjusted_faces

adjust_faces = adjust_faces_indices(faces)
print(adjust_faces)


def generate_edges_and_faces_from_vertices(faces):
    edges = {}
    edge_map = {}
    edge_num = 1
    edge_faces = []

    for face in faces:
        edge_face = []
        for ii in range(len(face)):
            v0 = face[ii]
            v1 = face[(ii + 1) % len(face)]  # 确保形成闭环

            # 确保 (v0, v1) 和 (v1, v0) 视为同一条边，且反向边用负号表示
            edge = (v0, v1) if v0 < v1 else (v1, v0)
            if edge not in edge_map:
                edges[edge_num] = edge
                edge_map[edge] = edge_num
                edge_map[(edge[1], edge[0])] = -edge_num  # 反向边
                edge_num += 1

            edge_face.append(edge_map[(v0, v1)])
        edge_faces.append(edge_face)

    return edges, edge_faces


edges, edge_faces = generate_edges_and_faces_from_vertices(adjust_faces)
print("edges: ", edges)   
print("\n", edge_faces)

# def are_coplanar(vertices, edge_indices, edges):
#     points = np.array([vertices[edge_indices[0]], vertices[edge_indices[1]], vertices[edge_indices[2]], vertices[edge_indices[3]]])
#     normal = np.cross(points[1] - points[0], points[2] - points[0])
#     return np.dot(normal, points[3] - points[0]) == 0
# def create_faces_from_edges(cells, vertices, edges):
#     faces = []
#     for cell_id in cells:
#         cell_vertices = cells[cell_id]
#         cell_edges = []
#         # Find all edges that connect the vertices of the cell
#         for vnum in range(len(cell_vertices)):
#             for edge_id, (v1, v2) in edges.items():
#                 if (v1 == cell_vertices[vnum] and v2 in cell_vertices) or (v2 == cell_vertices[vnum] and v1 in cell_vertices):
#                     cell_edges.append(edge_id)       
#         # Generate faces from edges
#         num_edges = len(cell_edges)
#         for i in range(num_edges):
#             for j in range(i + 1, num_edges):
#                 for k in range(j + 1, num_edges):
#                     for l in range(k + 1, num_edges):
#                         edge_indices = [edges[cell_edges[i]][0], edges[cell_edges[i]][1], edges[cell_edges[j]][0], edges[cell_edges[j]][1]]
#                         if len(set(edge_indices)) == 4:  # Check if they are 4 unique vertices
#                             if are_coplanar(vertices, edge_indices, edges):
#                                 face_edges = [cell_edges[i], cell_edges[j], cell_edges[k], cell_edges[l]]
#                                 face_edges = sorted(face_edges, key=lambda e: (edges[e][0], edges[e][1]))
#                                 faces.append(face_edges)
#     # Remove duplicate faces
#     unique_faces = []
#     seen_faces = set()
#     for face in faces:
#         sorted_face = tuple(sorted(face))
#         if sorted_face not in seen_faces:
#             seen_faces.add(sorted_face)
#             unique_faces.append(face)

#     return unique_faces
# faces = create_faces_from_edges(cells, vertices, edges)
# print("Faces in main: ",faces)
# 输入数据
# faces = [[3, 1, 0, 2], [3, 5, 4, 1], [6, 0, 2, 7], [6, 4, 5, 7], [6, 4, 1, 0], [5, 3, 2, 7]]
# edges = {1: (3, 1), 2: (1, 0), 3: (0, 2), 4: (2, 3), 5: (3, 5), 6: (5, 4), 7: (4, 1), 8: (6, 0), 9: (2, 7), 10: (7, 6), 11: (6, 4), 12: (5, 7)}



# print(edges.items())
# faces_edges = []
def faces_by_edges(edges, faces):
    faces_by_edges = {}
    # c = 1
    # 这里定义的是根据边组成的面
    for face in faces:
      for i in range(len(face)):
        faces_by_edges[i] = i
    return faces_by_edges


