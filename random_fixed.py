import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import io


def generate_cube_seeds(number_cells_x, number_cells_y, number_cells_z, standard_deviation=0, spatial_step=20):
    """
    Generate seeds for tessellation in a cubic grid.
    """
    grid_values = [
        [
            (i + np.random.normal(0, standard_deviation)) * spatial_step,
            (j + np.random.normal(0, standard_deviation)) * spatial_step,
            (k + np.random.normal(0, standard_deviation)) * spatial_step
        ]
        for k in range(number_cells_z) for j in range(number_cells_y) for i in range(number_cells_x)
    ]
    return grid_values


def remove_large_distance_ridges(vor, max_distance):
    """
    Remove Voronoi ridges with edges longer than max_distance.
    """
    filtered_faces = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            # Calculate the distance between each pair of consecutive vertices in the ridge
            distances = [np.linalg.norm(vor.vertices[ridge[i]] - vor.vertices[ridge[i + 1]]) for i in range(len(ridge) - 1)]
            # Check if all distances are within the max_distance
            if all(d <= max_distance for d in distances):
                filtered_faces.append(ridge)
    return filtered_faces


def remove_invalid_bodies(regions, faces):
    # Extract all valid vertices
    valid_vertices = set()
    for face in faces:
        valid_vertices.update(face)

    # Store valid regions
    valid_regions = []

    # Check if all vertices in a region are valid
    for region in regions:
        if all(vertex in valid_vertices for vertex in region):
            valid_regions.append(region)

    return valid_regions


def generate_edges_and_faces_from_vertices(faces):
    edges = {}
    edge_map = {}
    edge_num = 1
    edge_faces = []

    for face in faces:
        edge_face = []
        for ii in range(len(face)):
            v0 = face[ii]
            v1 = face[(ii + 1) % len(face)]
            edge = (v0, v1) if v0 < v1 else (v1, v0)
            if edge not in edge_map:
                edges[edge_num] = edge
                edge_map[edge] = edge_num
                edge_map[(edge[1], edge[0])] = -edge_num
                edge_num += 1
            edge_face.append(edge_map[(v0, v1)])
        edge_faces.append(edge_face)

    return edges, edge_faces


def generate_bodies_by_faces(vertices, faces_by_vertices, face_index_map):
    vertices_set = set(vertices)
    bodies = []
    for face in faces_by_vertices:
        face_set = set(face)
        if face_set.issubset(vertices_set):
            bodies.append(face_index_map[tuple(sorted(face))])
    return bodies


def update_vertices_faces_bodies(vertices, faces, bodies):
    # Create a mapping for all vertices
    vertex_dict = {i: vertices[i] for i in range(len(vertices))}

    # Find all used vertices in faces and bodies
    used_vertices = set()
    for face in faces:
        used_vertices.update(face)
    for body in bodies:
        used_vertices.update(body)

    # Create a mapping from old to new indices for used vertices
    new_index = {}
    new_vertices = []
    for old_idx in used_vertices:
        new_index[old_idx] = len(new_vertices) + 1
        new_vertices.append(vertex_dict[old_idx - 1])  # Adjust for 1-based index

    # Update faces and bodies to use new indices
    new_faces = [[new_index[idx] for idx in face] for face in faces]
    new_bodies = [[new_index[idx] for idx in body] for body in bodies]

    return new_vertices, new_faces, new_bodies


def remove_unreferenced_faces(faces, bodies):
    # Create a face index set
    face_set = set(map(tuple, faces))

    # Record each face referenced by a body
    referenced_faces = set()
    for body in bodies:
        for face in faces:
            # If all vertices of a face are in the body, consider it referenced by the body
            if set(face).issubset(body):
                referenced_faces.add(tuple(face))

    # Keep only referenced faces
    valid_faces = [list(face) for face in referenced_faces if face in face_set]

    return valid_faces

def lloyd_relax(points, steps =1):
    for _ in range(steps):
        vor = Voronoi(points)
        new_points = []
        for region in vor.regions:
            if not region:
                continue
            if -1 in region:
                continue
            vertices = vor.vertices[region]
            centroid = np.mean(vertices, axis=0)  # calculate the centroid of each polygon
            new_points.append(centroid)
        if len(new_points) >= 5:  # Check if enough points exist
            points = np.array(new_points)  # Only update if there are enough points
        else:
            break  # Exit the loop if not enough points
    return points




class SurfaceEvolver:
    def __init__(self, vertices, edges, adjust_faces, cells, density_values, volume_values, polygonal=True):
        self.vertices = vertices
        self.edges = edges
        self.faces = adjust_faces
        self.cells = cells
        self.density_values = {tuple(map(tuple, np.round(np.array(key), 3))): round(value, 3) for key, value in density_values.items()}
        self.volume_values = volume_values
        self.polygonal = polygonal
        self.fe_file = io.StringIO()

    def generate_fe_file(self):
        self.fe_file.write("SPACE_DIMENSION 3 \n")
        self.fe_file.write("SCALE 0.005 FIXED\n")
        self.fe_file.write("STRING \n")
        self.fe_file.write("\n")

        # Write vertices
        self.fe_file.write("vertices \n")
        for k, v in enumerate(self.vertices):
            self.fe_file.write(f"{k + 1} {v[0]} {v[1]} {v[2]} \n")
        self.fe_file.write("\n")

        # Write edges
        self.fe_file.write("edges \n")
        for k, v in self.edges.items():
            self.fe_file.write(f"{k} {v[0]} {v[1]} \n")
        self.fe_file.write("\n")

        # Write faces
        self.fe_file.write("faces \n")
        for i, face in enumerate(self.faces):
            self.fe_file.write(f"{i + 1} {' '.join(map(str, face))} \n")
        self.fe_file.write("\n")

        # Write bodies
        self.fe_file.write("bodies \n")
        i = 1
        for k, v in enumerate(self.cells):
            if v:
                str_value = " ".join(str(vv) for vv in v)
                self.fe_file.write(f"{abs(i)} {str_value} \n")
                i += 1
        self.fe_file.write("\n")

        # Additional commands
        self.fe_file.write("read \n \n")
        self.fe_file.write("show_all_edges off \n")
        self.fe_file.write("metric_conversion off \n")
        self.fe_file.write("autorecalc on \n")
        self.fe_file.write("gv_binary off \n")
        self.fe_file.write("gravity off \n")
        self.fe_file.write("ii := 0; \n")

        return self.fe_file

    def save_fe_file(self, file_name: str):
        with open(f'{file_name}', mode='w') as f:
            print(self.fe_file.getvalue(), file=f)
        return True


# Generate seeds and Voronoi diagram
seeds = generate_cube_seeds(3, 3, 4, 0.15, 20)
vor = Voronoi(list(seeds))

# Remove ridges with large distances
filtered_faces = remove_large_distance_ridges(vor, max_distance=30)
adjust_faces = [[vertex + 1 for vertex in face] for face in filtered_faces]
bodies = [[vertex + 1 for vertex in region] for region in vor.regions if -1 not in region]
bodies_remove_faces = remove_invalid_bodies(bodies, adjust_faces)

# Filter faces used by bodies
ref_faces = remove_unreferenced_faces(adjust_faces, bodies_remove_faces)

# Update vertices, faces, and bodies
new_vertices, new_faces, new_bodies = update_vertices_faces_bodies(vor.vertices, ref_faces, bodies_remove_faces)

face_index_map = {tuple(sorted(face)): i + 1 for i, face in enumerate(new_faces)}  # faces to numbers
edges, faces_by_edges = generate_edges_and_faces_from_vertices(new_faces)

regions_by_faces = []
for region in new_bodies:
    print("Processing region:", region)  # Debug: print the current region
    faces = generate_bodies_by_faces(region, new_faces, face_index_map)
    print("Faces found for region:", faces)  # Debug: print the faces found for the current region
    regions_by_faces.append(faces)

se = SurfaceEvolver(new_vertices, edges, faces_by_edges, regions_by_faces, density_values={}, volume_values={})
input_path = r"C:\Users\takeoff\Desktop\GRINDING\random_test_v3.8(334).fe"
se.generate_fe_file()
se.save_fe_file(input_path)



#  normal, circular, x and y furrow



