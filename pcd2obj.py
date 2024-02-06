import numpy as np
import trimesh
import open3d as o3d

dataname='example_data/lidar/main.pcd'

pcd = o3d.io.read_point_cloud(dataname)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)

o3d.io.write_triangle_mesh("example_data/anno_lang/point_cloud_top_view.obj", p_mesh_crop)

o3d.visualization.draw_geometries([p_mesh_crop])