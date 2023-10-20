import numpy as np
import open3d as o3d


def transform_extrinsic_matrices(extrinsics):
    A = np.array([[-1, 0, 0],
                  [ 0, 0, 1],
                  [ 0,-1, 0]])

    extrinsics[:, :3, :3] = np.einsum('ij,kjl->kil',
                                      A,
                                      extrinsics[:, :3, :3])

    extrinsics[:, [0, 2], :] = extrinsics[:, [2, 0], :]


def visualize_cameras(extrinsics: np.array):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    #draw ground plane
    mesh_box = o3d.geometry.TriangleMesh.create_box(
                    width=0.001,
                    height=5.0,
                    depth=5.0
                )

    mesh_box.paint_uniform_color([0.7, 0.7, 0.7])
    mesh_box.translate([0, -3, 0])

    vis.add_geometry(mesh_box)


    #transform extrinsic matrices
    transform_extrinsic_matrices(extrinsics)


    #draw camera path
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(extrinsics[:, :3, 3])
    line_segments = []
    for i in range(len(extrinsics)-1):
        line_segments.append([i, i+1])

    lines.lines = o3d.utility.Vector2iVector(line_segments)

    vis.add_geometry(lines)


    #draw camera directions
    for pose in extrinsics:
        #pose[:3, :3] = A @ pose[:3, :3]
        #pose[[0, 2], :] = pose[[2, 0], :]

        camera = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.0125, cylinder_height=0.1,
                    cone_radius=0.025, cone_height=0.05
                )
        camera.paint_uniform_color([0.5, 0.1, 0.1])
        camera.transform(pose)

        vis.add_geometry(camera)


    #set view control, adjust orientation
    vis.get_view_control().set_front([1, 1, 1])

    vis.run()

    vis.destroy_window()


if __name__ == "__main__":
    positions = np.load(open("camera_poses.npy", "rb"))

    visualize_cameras(positions)
