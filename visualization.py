import numpy as np
import open3d as o3d


def transform_extrinsic_matrices(extrinsics: np.array):
    A = np.array([[ 1, 0, 0],
                  [ 0, 0,-1],
                  [ 0,-1, 0]])

    extrinsics[:, :3, :3] = np.einsum('ij,kjl->kil',
                                      A,
                                      extrinsics[:, :3, :3])

    extrinsics[:, [0, 2], :] = extrinsics[:, [2, 0], :]


def visualize_cameras(extrinsics: np.array,
                      extra: int = 0,
                      scale: int = 0.1) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    #draw ground plane
    mesh_box = o3d.geometry.TriangleMesh.create_box(
                    width=0.001,
                    height=10.0,
                    depth=10.0
                )

    mesh_box.paint_uniform_color([0.4, 0.4, 0.4])
    mesh_box.translate([-1,-7,-5])

    vis.add_geometry(mesh_box)

    #transform extrinsic matrices
    transform_extrinsic_matrices(extrinsics)

    #draw camera path
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(extrinsics[:, :3, 3])
    line_segments = [ [i, i+1] for i in range(len(extrinsics)-1-extra) ]
    lines.lines = o3d.utility.Vector2iVector(line_segments)
    lines.paint_uniform_color([0.1, 0.7, 0.5])

    vis.add_geometry(lines)

    #draw camera directions
    for ind, pose in enumerate(extrinsics):
        camera = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=scale/8,
                    cylinder_height=scale,
                    cone_radius=scale/4,
                    cone_height=scale/2
                )
        if ind == len(extrinsics) - extra:
            camera.paint_uniform_color([0.2, 0.1, 0.7])
        else:
            camera.paint_uniform_color([0.7, 0.1, 0.2])
        camera.transform(pose)
        vis.add_geometry(camera)

    #set view control, adjust orientation
    vis.get_view_control().set_front([1, 1, 1])

    vis.run()

    vis.destroy_window()


if __name__ == "__main__":
    import os

    positions = np.load(open("camera_poses.npy", "rb"))

    if os.path.exists("new_camera_poses.npy"):
        new_pos = np.load(open("new_camera_poses.npy", "rb"))
        positions = np.concatenate((positions, new_pos))
        extra = len(new_pos)

    visualize_cameras(positions, extra=extra, scale=0.2)
