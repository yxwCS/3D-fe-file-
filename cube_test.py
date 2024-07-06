import numpy as np
import pyvoro
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Lattice:
    def __init__(self, number_cells_x, number_cells_y, number_cells_z):
        self.number_cells_x = number_cells_x
        self.number_cells_y = number_cells_y
        self.number_cells_z = number_cells_z
        self.tessellation = None

    def generate_cube_seeds(self, standard_deviation=0, spatial_step=1):
        grid_values = [[(i + np.random.normal(0, standard_deviation)) * spatial_step,
                        (j + np.random.normal(0, standard_deviation)) * spatial_step,
                        (k + np.random.normal(0, standard_deviation)) * spatial_step]
                       for k in range(self.number_cells_z)
                       for j in range(self.number_cells_y)
                       for i in range(self.number_cells_x)]
        unique_grid_values = [list(x) for x in set(tuple(x) for x in grid_values)]
        self.unique_grid_values = unique_grid_values
        min_seed = np.min(unique_grid_values, axis=0)
        max_seed = np.max(unique_grid_values, axis=0)
        limits = [[min_seed[0] - 10, max_seed[0] + 10],
                  [min_seed[1] - 10, max_seed[1] + 10],
                  [min_seed[2] - 10, max_seed[2] + 10]]
        return unique_grid_values, limits

    def generate_voronoi_tessellation(self, seed_values, limits):
        voronoi = pyvoro.compute_voronoi(seed_values, limits, 2)
        self.tessellation = voronoi
        return self.tessellation

    def plot_voronoi(self, limits):
        if self.tessellation is None:
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        rng = np.random.default_rng(11)

        for vnoicell in self.tessellation:
            faces = []
            vertices = np.array(vnoicell['vertices'])
            for face in vnoicell['faces']:
                faces.append(vertices[np.array(face['vertices'])])

            polygon = Poly3DCollection(faces, alpha=0.5, facecolors=rng.uniform(0, 1, 3), linewidths=0.5,
                                       edgecolors='black')
            ax.add_collection3d(polygon)

        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def get_vertex_number(self, vertex, vertices):
        vertex_key = tuple(vertex)
        if vertex_key not in vertices:
            vertices[vertex_key] = len(vertices) + 1
        return vertices[vertex_key]

    def get_enum(self, edge, edges):
        edge_key = tuple(sorted(edge))
        if edge_key not in edges:
            edges[edge_key] = len(edges) + 1
        return edges[edge_key]

    def extract_elements(self):
        vertices = {}
        edges = {}
        cells = {}
        vertex_count = 1
        edge_count = 1

        for cell in self.tessellation:
            cell_index = tuple(cell['original'])
            cell_vertices = cell['vertices']
            cell_faces = cell['faces']

            vertex_map = {}
            for i, vertex in enumerate(cell_vertices):
                vertex_key = tuple(np.round(vertex, 3))
                if vertex_key not in vertices:
                    vertices[vertex_key] = vertex_count
                    vertex_map[i] = vertex_count
                    vertex_count += 1
                else:
                    vertex_map[i] = vertices[vertex_key]

            cell_edges = []
            for face in cell_faces:
                face_vertices = face['vertices']
                sorted_face_vertices = sort_face_vertices([cell_vertices[idx] for idx in face_vertices])
                for i in range(len(sorted_face_vertices)):
                    v0 = tuple(np.round(sorted_face_vertices[i], 3))
                    v1 = tuple(np.round(sorted_face_vertices[(i + 1) % len(sorted_face_vertices)], 3))
                    edge_key = tuple(sorted((v0, v1)))
                    if edge_key not in edges:
                        edges[edge_key] = edge_count
                        edge_count += 1
                    cell_edges.append((v0, v1, edges[edge_key]))

            cells[cell_index] = cell_edges

        return vertices, edges, cells


class SurfaceEvolver:
    def __init__(self, vertices, edges, cells, density_values, volume_values, polygonal=True):
        self.vertices = vertices
        self.edges = edges
        self.cells = cells
        self.density_values = {tuple(map(tuple, np.round(np.array(key), 3))): round(value, 3) for key, value in density_values.items()}
        self.volume_values = volume_values
        self.polygonal = polygonal
        self.fe_file = io.StringIO()

    def generate_fe_file(self) -> io.StringIO:
        self.fe_file.write("SPACE_DIMENSION 3 \n")
        self.fe_file.write("SCALE 0.005 FIXED\n")
        self.fe_file.write("STRING \n")
        self.fe_file.write("\n")

        # Write vertices
        self.fe_file.write("vertices \n")
        print("Vertices content:")
        for k, v in self.vertices.items():
            self.fe_file.write(f"{v}   {k[0]} {k[1]} {k[2]}\n")
            print(f"Vertex {v}: Coordinates {k}")
        self.fe_file.write("\n")

        # Write edges
        self.fe_file.write("edges \n")
        print("Edges content:")
        for k, v in self.edges.items():
            if tuple(k) in self.density_values:
                lambda_val = self.density_values[tuple(k)]
                self.fe_file.write(f"{v}   {self.vertices[k[0]]}   {self.vertices[k[1]]}   density {lambda_val}\n")
                print(f"Edge {k}: Index {v}, Density {lambda_val}")
            else:
                print(f"Edge {tuple(k)} not found in density_values")
        self.fe_file.write("\n")

        # Write faces
        self.fe_file.write("faces \n")
        face_num = 1
        for cell, edges in self.cells.items():
            face_edges = self.order_edges(edges)
            if face_edges:
                for edge_group in face_edges:
                    if len(edge_group) == 4:  # 确保每个面有4条边
                        self.fe_file.write(f"{face_num}   {' '.join(map(str, [edge[2] for edge in edge_group]))}\n")
                        face_num += 1
        self.fe_file.write("\n")

        # Write bodies
        self.fe_file.write("bodies \n")
        body_num = 1
        for cell in self.cells.keys():
            self.fe_file.write(f"{body_num}   {body_num}    VOLUME {self.volume_values[body_num]} \n")
            body_num += 1
        self.fe_file.write("\n \n")

        # Additional commands
        self.fe_file.write("read \n \n")
        self.fe_file.write("show_all_edges off \n")
        self.fe_file.write("metric_conversion off \n")
        self.fe_file.write("autorecalc on \n")
        self.fe_file.write("gv_binary off \n")
        self.fe_file.write("gravity off \n")
        self.fe_file.write("ii := 0; \n")
        if not self.polygonal:
            self.add_refining_triangulation(3)
        return self.fe_file

    def order_edges(self, cell_edges):
        # Group edges by faces
        faces = {}
        for edge in cell_edges:
            for vertex in edge[:2]:
                if vertex not in faces:
                    faces[vertex] = []
                faces[vertex].append(edge)

        face_groups = []
        for vertex, edges in faces.items():
            if len(edges) < 3:
                continue
            ordered_edges = [edges[0]]
            while len(ordered_edges) < len(edges):
                last_edge = ordered_edges[-1]
                next_edge = next((e for e in edges if (e[0] == last_edge[1] or e[1] == last_edge[1]) and e not in ordered_edges), None)
                if not next_edge:
                    break
                if next_edge[0] == last_edge[1]:
                    ordered_edges.append(next_edge)
                else:
                    ordered_edges.append((next_edge[1], next_edge[0], next_edge[2]))
            face_groups.append(ordered_edges)
        return face_groups

    def save_fe_file(self, file_name: str) -> bool:
        self.fe_file.write('q; \n')
        with open(f'{file_name}', mode='w') as f:
            print(self.fe_file.getvalue(), file=f)
        return True

def sort_face_vertices(vertices):
    vertices = np.array(vertices)
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)].tolist()

# 创建3D晶格
lattice = Lattice(3, 3, 3)
seeds, limits = lattice.generate_cube_seeds(0, 20)
tessellation = lattice.generate_voronoi_tessellation(seeds, limits)
lattice.plot_voronoi(limits)

# 提取元素
vertices, edges, cells = lattice.extract_elements()

# 为每个单元分配初始体积和边张力
volume_values = {k: 1.0 for k in range(1, len(cells) + 1)}
density_values = {tuple(k): 1.0 for k in edges.keys()}

# 生成 Surface Evolver 对象并保存文件
evolver = SurfaceEvolver(vertices, edges, cells, density_values, volume_values)
evolver.generate_fe_file()
input_file_path = "3D_testv20.fe"
evolver.save_fe_file(input_file_path)

print(f"FE file saved to {input_file_path}")
