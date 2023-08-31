import os
import json
import csv
import random
import numpy as np
import math
import transforms3d

from pyrep.objects.shape import Shape
from pyrep.backend import sim


class ShapenetSampler:
    def __init__(self, shapenet_to_generate, num_unique, scaling_factor=0.2):
        self.shapenet_objs = read_shapenet(shapenet_to_generate) 
        self.scaling_factor = scaling_factor
        self.num_unique = num_unique

    def sample_shapenet_objects(self, task_base, num_samples, mass=0.1):
        # todo: deal with mass?
        created = []
        meta = []
        for i in range(num_samples):
            shapenet_class = random.choice(self.shapenet_objs)
            if isinstance(shapenet_class.base_names, list):
                meta.append(shapenet_class.base_names[0])
            else:
                meta.append(shapenet_class.base_names)
            shapenet_obj = shapenet_class.get_random_shape(self.num_unique)
            resp = import_convex_shape(shapenet_obj.convex_dir, scaling_factor=self.scaling_factor)
            vis = Shape.import_shape(shapenet_obj.path, scaling_factor=self.scaling_factor)
            resp.set_renderable(False)
            vis.set_renderable(True)
            vis.set_parent(resp)
            vis.set_dynamic(False)
            vis.set_respondable(False)
            resp.set_dynamic(True)
            # resp.set_mass(mass)

            euler = euler_world_to_shape(vis, [math.pi/2, 0, 0])
            resp.rotate(euler)
            
            resp.set_respondable(True)
            resp.set_model(True)
            resp.set_parent(task_base)
            created.append(resp)
        return created, meta

def rotate_shape(shape):
    euler = euler_world_to_shape(shape, [math.pi/2, 0, 0])
    shape.rotate(euler)

def euler_world_to_shape(shape, euler):
    m = sim.simGetObjectMatrix(shape._handle, -1)
    x_axis = np.array([m[0], m[4], m[8]])
    y_axis = np.array([m[1], m[5], m[9]])
    z_axis = np.array([m[2], m[6], m[10]])
    euler = np.array([euler[0], euler[1], euler[2]])
    R = transforms3d.euler.euler2mat(*euler, axes='rxyz')
    T = np.array([x_axis, y_axis, z_axis]).T
    new_R = np.linalg.inv(T)@R@T
    new_euler = transforms3d.euler.mat2euler(new_R, axes='rxyz')
    return new_euler


def import_convex_shape(convex_dir, scaling_factor=0.2):
    convex_parts = []
    for k, convex_part in enumerate(os.listdir(convex_dir)):
        if convex_part.endswith('.obj'):
            path = f'{convex_dir}/{convex_part}'
            convex_parts.append(Shape.import_shape(path, scaling_factor=scaling_factor))
            # rotate_shape(convex_parts[-1])

    if len(convex_parts) > 1:
        convex_parts_handles = [o.get_handle() for o in convex_parts]
        complete_obj_handle = sim.simGroupShapes(convex_parts_handles)
        complete_obj = Shape(complete_obj_handle)
    else:
        complete_obj = convex_parts[0]
    return complete_obj

class ShapenetObj:
    def __init__(self, path, convex_dir, name, parent_name):
        self.path = path
        self.convex_dir = convex_dir
        self.name = name
        self.parent_name = parent_name

    def __repr__(self):
        return f"{self.name}: {self.path}"


class ShapenetClass:
    def __init__(self, base_names, synset_id, num_instances):
        self.base_names = base_names
        self.synset_id = synset_id
        self.num_instances = num_instances
        self.paths, self.names = get_shapenet_data(synset_id)

    def __getitem__(self, idx):
        if idx >= self.num_instances:
            raise IndexError()
        path = f"/PATH/TO/shapenetcore_v2/{self.synset_id}/{self.paths[idx]}/models/model_normalized.obj"
        convex_dir = f"/PATH/TO/shapenet/{self.synset_id}/{self.paths[idx]}"
        name = self.names[idx][np.random.randint(len(self.names[idx]))]
        return ShapenetObj(path, convex_dir, name, self.base_names[0])

    def get_random_shape(self, num_unique):
        if num_unique <= 0:
            idx = np.random.randint(self.num_instances)
        else:
            idx = np.random.randint(num_unique)
        return self[idx]


def get_shapenet_data(synset_id):
    csv_base_dir = "/PATH/TO/shapenet/metadata"
    csv_file = f"{csv_base_dir}/{synset_id}.csv"
    paths, names = [], []
    with open(csv_file, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # first row of csv
            if row[0] == 'fullId':
                continue
            paths.append(row[0].split(".")[-1])
            names.append(row[2].split(','))
    return paths, names


def read_shapenet(shapenet_to_generate):
    json_file = "/PATH/TO/shapenetcore_v2/taxonomy.json"

    with open(json_file) as json_data:
        data = json.load(json_data)

        all_objs = []
        all_children = []
        # keep track of children to process them last
        for obj in data:
            for child in obj['children']:
                all_children.append(child)

        # this object doesn't exist in the shapenet folder, not sure why
        all_children.append("02834778")  # bicycle

        # process objects recursively
        for obj in data:
            if obj['synsetId'] not in all_children:
                name_lst = obj['name'].split(',')
                if len(shapenet_to_generate) > 0 and name_lst[0] not in shapenet_to_generate:
                    continue
                parent_obj = ShapenetClass(
                    name_lst, obj['synsetId'], obj['numInstances'])
                all_objs.append(parent_obj)

                # assert path exists
                # path, _ = parent_obj[np.random.randint(parent_obj.num_instances)]
                # assert os.path.exists(path), path
        return all_objs
