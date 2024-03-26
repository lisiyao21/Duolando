import bpy
import numpy as np
import os
import os.path as osp
import sys
from math import atan2, pi
from mathutils import Matrix, Quaternion, Vector

import argparse


smplx_blender_path = '/mnt/sfs-common/syli/duet_final/software/smplx_blender_addon.zip'


class ArgumentParserForBlender(argparse.ArgumentParser):
    """This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because it
    will try to process the script's -a and -b flags:

    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns an
        empty list)."""
        try:
            idx = sys.argv.index('--')
            return sys.argv[idx + 1:]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            print(f'Get argv error: {e}')
            return []

    # overrides superclass
    def parse_args(self):
        """This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before.

        See the docstring of the class for usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


sys.path.append(os.getcwd())

SMPLX_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2',
    'left_index3', 'left_middle1', 'left_middle2', 'left_middle3',
    'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2',
    'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1',
    'right_index2', 'right_index3', 'right_middle1', 'right_middle2',
    'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3',
    'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1',
    'right_thumb2', 'right_thumb3'
]


def get_last_keyframe(scene):
    last_keyframe = 0
    for obj in scene.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe_point in fcurve.keyframe_points:
                    frame = keyframe_point.co.x
                    if frame > last_keyframe:
                        last_keyframe = frame
    return last_keyframe


# def add_smplx(file_path):
def add_smplx(data, handpose='flat'):
    # if not osp.exists(file_path):
    #     return
    # data = np.load(file_path, allow_pickle=True)['smplx'].item()
    # handpose = 'flat'
    gender = data['meta']['gender']
    print(f'gender: {gender}')
    if bpy.context.mode != 'OBJECT':
        # Set the Blender model to pose mode
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.window_manager.smplx_tool.smplx_handpose = handpose
    bpy.context.window_manager.smplx_tool.smplx_gender = gender
    bpy.ops.scene.smplx_add_gender()
    smplx = bpy.data.objects[f'SMPLX-mesh-{gender}']

    if 'betas' in data:
        new_betas = data['betas'][0, :10]

        for i in range(len(new_betas)):
            name = f'Shape{i:03d}'
            key_block = smplx.data.shape_keys.key_blocks[name]
            value = new_betas[i]
            # Adjust key block min/max range to value
            if value < key_block.slider_min:
                key_block.slider_min = value
            elif value > key_block.slider_max:
                key_block.slider_max = value
            key_block.value = value

    bpy.ops.object.smplx_update_joint_locations()

    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = bpy.data.objects[f'SMPLX-{gender}']

    if bpy.context.mode != 'POSE':
        # Set the Blender model to pose mode
        bpy.ops.object.mode_set(mode='POSE')

    # Apply the SMPL-X model parameters to the Blender model
    obj = bpy.data.objects[f'SMPLX-{gender}']
    armature = obj
    transl = data['transl'].reshape(-1, 3)
    # global_orient = data['global_orient']
    pose = data['poses'].reshape(-1, 55, 3)
    num_frames = pose.shape[0]
    num_joints = pose.shape[1]

    # for debug
    # num_frames = 100

    for frame_idx in range(num_frames):
        # transl
        translation = transl[frame_idx]
        armature.pose.bones['root'].location = (translation[0], translation[1],
                                                translation[2])
        armature.pose.bones['root'].keyframe_insert(
            data_path='location', index=-1)
        # joint rotation
        for joint_idx in range(num_joints):
            rod = Vector(
                (pose[frame_idx][joint_idx][0], pose[frame_idx][joint_idx][1],
                 pose[frame_idx][joint_idx][2]))
            angle_rad = rod.length
            axis = rod.normalized()
            bone_name = SMPLX_JOINT_NAMES[joint_idx]
            if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
                armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
            # note: quat order is quat_xyzw
            quat = Quaternion(axis, angle_rad)
            # joint = armature.pose.bones[bone_name]
            armature.pose.bones[bone_name].rotation_quaternion = quat
            armature.pose.bones[bone_name].keyframe_insert(
                data_path='rotation_quaternion', index=-1)
        bpy.context.scene.frame_set(frame_idx)

        # bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
    bpy.data.objects[f'SMPLX-{gender}'].rotation_euler[0] = 1.5707963705062866
    bpy.data.objects[f'SMPLX-{gender}'].rotation_euler[1] = 3.1415927410125732

    # Set color
    # bpy.data.materials["SMPLX-male"].diffuse_color = [0.5, 0.8, 0.46, 1]
    # bpy.data.materials["SMPLX-female"].diffuse_color = [0.8, 0.5, 0.61, 1]

    # bpy.data.materials[f'SMPLX-{gender}'].diffuse_color = \
    #     (0.5, 0.8, 0.46, 1) if gender == 'male' else (0.8, 0.5, 0.61, 1)

    # Subsurface Metallic Alpha
    bpy.data.materials[f'SMPLX-{gender}'].node_tree.\
        nodes['Principled BSDF'].inputs[1].default_value = 0.5
    bpy.data.materials[f'SMPLX-{gender}'].node_tree.\
        nodes['Principled BSDF'].inputs[6].default_value = 0.5
    bpy.data.materials[f'SMPLX-{gender}'].node_tree.\
        nodes['Principled BSDF'].inputs[21].default_value = 0.5

    # base color
    bpy.data.materials[f'SMPLX-{gender}'].node_tree.\
        nodes['Principled BSDF'].inputs[0].default_value = \
        (0.5, 0.8, 0.46, 1) if gender == 'male' else (0.8, 0.5, 0.61, 1)

    # Subsurface color
    bpy.data.materials[f'SMPLX-{gender}'].node_tree.\
        nodes['Principled BSDF'].inputs[3].default_value = \
        (0., 0., 0., 1)


def setup_parser():
    parser = ArgumentParserForBlender(
        description='Visualize a list of SMPL-X in one scene')
    # parser.add_argument(
    #     '--filelist', nargs='+', help='SMPL-X filelist', required=True)
    parser.add_argument(
        '--npy0_path',
        type=str,
        help='A npz file that containing all smplx data '
        'to be visualized in one scene',
        required=True)
    parser.add_argument(
        '--npy1_path',
        type=str,
        help='A npz file that containing all smplx data '
        'to be visualized in one scene',
        required=True)
    parser.add_argument(
        '--output_video_path',
        type=str,
        help='Rendered video path.',
        required=True)
    parser.add_argument(
        '--input_blend_path', type=str, help='Input blend path', default=None)
    parser.add_argument(
        '--export_mainfile_path',
        type=str,
        help='Output blend path (middle file)',
        default='')
    parser.add_argument(
        '--engine_type',
        type=str,
        help='Engine types for blender renderer',
        default='EEVEE')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    bpy.ops.object.select_all(action='DESELECT')

    if 'Cube' in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)
    if 'Empty' in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects['Empty'], do_unlink=True)
    if 'Light' not in bpy.data.objects.keys():
        bpy.ops.object.light_add(type='POINT')
        light = bpy.context.object
        light.name = 'Light'
    if 'Camera' not in bpy.data.objects.keys():
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW')

    bpy.ops.preferences.addon_install(overwrite = False, filepath = str(smplx_blender_path))
    bpy.ops.preferences.addon_enable(module="smplx_blender_addon")
    bpy.ops.script.reload()

    # bpy.ops.preferences.addon_enable(module='smplx_blender_addon')
    # bpy.ops.script.reload()

    if not args.input_blend_path:
        # for fn in args.filelist:
        #     add_smplx(fn)
        if not osp.exists(args.npy0_path):
            raise FileNotFoundError(args.npy0_path)
        if not osp.exists(args.npy1_path):
            raise FileNotFoundError(args.npy1_path)

        data0 = np.load(args.npy0_path, allow_pickle=True, encoding='bytes').item()
        data1 = np.load(args.npy1_path, allow_pickle=True, encoding='bytes').item()

        # data_list = np.load(
        #     args.npz_path, allow_pickle=True)['data_list'].item()
        # data_list = np.load(args.npz_path, allow_pickle=True)['data_list']
        for data in [data0, data1]:
            add_smplx(data)
    # if not args.input_blend_path:
    #     if not args.file1:
    #         exit(1)
    #     add_smplx(args.file1)
    #     if args.file2:
    #         add_smplx(args.file2)
    # else:
    #     raise NotImplementedError

    # bpy.context.scene.display.shading.light = 'STUDIO'
    # bpy.context.scene.display.shading.studio_light = 'paint.sl'
    bpy.data.scenes['Scene'].view_settings.view_transform = 'Standard'

    # bpy.data.worlds['World'].node_tree.\
    #     nodes['Background'].inputs[0].default_value = (0, 0, 0, 1)
    bpy.data.worlds['World'].node_tree.\
        nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
    # bpy.data.worlds['World'].color = (1, 1, 1)
    # Get the last keyframe in the current scene
    last_keyframe = get_last_keyframe(bpy.context.scene)

    # Set the render engine
    # bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    # bpy.context.scene.render.engine = 'CYCLES'

    # Set the output file format to FFmpeg video
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'

    # Set the output format to MPEG-4
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0

    # b_camera = bpy.data.objects['Camera']
    # b_camera.location = (0, 8., 0)
    # b_camera.rotation_euler = (3.14 / 2, 0, 3.14)
    # b_light = bpy.data.objects['Light']
    # b_light.location = (0, 9., 0)
    # b_light.rotation_euler = (3.14 / 2, 0, 3.14)
    # b_light.data.type = 'AREA'
    # b_light.data.size = 10
    # b_light.data.size_y = 10

    bpy.data.objects["SMPLX-female"].rotation_euler[0] = 1.5707963705062866
    bpy.data.objects["SMPLX-female"].rotation_euler[1] = 3.1415927410125732
    bpy.data.objects["SMPLX-male"].rotation_euler[0] = 1.5707963705062866
    bpy.data.objects["SMPLX-male"].rotation_euler[1] = 3.1415927410125732

    # bpy.data.objects["SMPLX-female"].hide_render= True
    # bpy.data.objects["SMPLX-mesh-female"].hide_render = True
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'

    bpy.data.objects["Camera"].location[0] = 0
    bpy.data.objects["Camera"].location[1] = -7.38579 
    bpy.data.objects["Camera"].location[1] = -7.38579 
    # bpy.data.objects["Camera"].location[2] = 0.84
    bpy.data.objects["Camera"].location[2] = 0

    bpy.data.objects["Camera"].rotation_euler[0] = 1.5707963705062866
    bpy.data.objects["Camera"].rotation_euler[1] = 0
    bpy.data.objects["Camera"].rotation_euler[2] = 0

    bpy.data.cameras["Camera"].lens = 38.0

    # b_empty = bpy.data.objects.new('Empty', None)
    # b_camera.parent = b_empty  # setup parenting
    # b_light.parent = b_empty
    # scn = bpy.context.scene
    # scn.collection.objects.link(b_empty)

    # cam_constraint = b_camera.constraints.new(type='TRACK_TO')
    # cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    # cam_constraint.up_axis = 'UP_Y'
    # cam_constraint.target = b_empty

    bpy.data.scenes["Scene"].view_settings.view_transform = 'Standard'
    bpy.data.materials["SMPLX-male"].diffuse_color = [0.5, 0.8, 0.46, 1]
    bpy.data.materials["SMPLX-female"].diffuse_color = [0.8, 0.5, 0.61, 1]
    bpy.data.worlds["World"].color = (1, 1, 1)

    # light_constraint = b_light.constraints.new(type='TRACK_TO')
    # light_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    # light_constraint.up_axis = 'UP_Y'
    # light_constraint.target = b_empty

    # matrix_world = Matrix(
    #     ((-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
    # if 'SMPLX-male' in bpy.data.objects.keys() and \
    #    'SMPLX-female' in bpy.data.objects.keys():
    #     m = bpy.data.objects['SMPLX-male']
    #     f = bpy.data.objects['SMPLX-female']
    #     midpoint = (matrix_world @ f.pose.bones['pelvis'].center +
    #                 matrix_world @ m.pose.bones['pelvis'].center) / 2.
    #     v = matrix_world @ f.pose.bones['pelvis'].center - midpoint

    # elif 'SMPLX-male' in bpy.data.objects.keys():
    #     m = bpy.data.objects['SMPLX-male']
    #     midpoint = (matrix_world @ m.pose.bones['left_wrist'].center +
    #                 matrix_world @ m.pose.bones['right_wrist'].center) / 2.
    #     v = matrix_world @ m.pose.bones['left_wrist'].center - midpoint
    # elif 'SMPLX-female' in bpy.data.objects.keys():
    #     f = bpy.data.objects['SMPLX-female']
    #     midpoint = (matrix_world @ f.pose.bones['left_wrist'].center +
    #                 matrix_world @ f.pose.bones['right_wrist'].center) / 2.
    #     v = matrix_world @ f.pose.bones['left_wrist'].center - midpoint

    # bpy.data.objects['Empty'].location = midpoint
    # angle = atan2(v.x, v.y) + pi / 2.
    # bpy.data.objects['Empty'].rotation_euler = (0, 0, -angle)

    # args.export_mainfile_path = args.npz_path.replace('.npz', '.blend')
    if len(args.export_mainfile_path) > 0:
        abs_path = osp.abspath(args.export_mainfile_path)
        print(f'Exporting mainfile to {abs_path}.')
        bpy.ops.wm.save_as_mainfile(filepath=abs_path)

    # render to video output path
    if args.output_video_path:
        bpy.context.scene.render.filepath = args.output_video_path

        # Set the start and end frames for the animation
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = int(last_keyframe)

        # Render the animation
        bpy.ops.render.render(animation=True)
