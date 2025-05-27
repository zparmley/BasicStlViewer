import abc
import colorsys
import dataclasses
import functools
import itertools
import math
import concurrent.futures
from pathlib import Path
import random
import time
import typing

import numpy as np
import trimesh
import dearpygui.dearpygui as dpg
from dpgcontainers.containers import Window
from dpgcontainers.containers import SliderFloat
from dpgcontainers.containers import SliderInt
from dpgcontainers.containers import Drawlist
from dpgcontainers.containers import DrawLayer
from dpgcontainers.containers import DrawNode
from dpgcontainers.containers import dTriangle
from dpgcontainers.containers import Group
from dpgcontainers.containers import Text
from dpgcontainers.containers import InputInt
from dpgcontainers.containers import Button
from dpgcontainers.containers import TableCell
from dpgcontainers.containers import Table
from dpgcontainers.containers import TableRow
from dpgcontainers.containers import TableColumn
from dpgmagictag.magictag import MagicTag
from stlviewer.config import Config


config = Config.factory('gui').viewer


def printing_timer(f):
    @functools.wraps(f)
    def _inner(*args, **kwargs):
        start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        ret = f(*args, **kwargs)
        duration_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start
        duration_ms = duration_ns / 1000
        duration_s = duration_ns / 1e+9
        print(f'=== {f.__name__} ran in {duration_s} seconds / {duration_ms} ms')
        return ret
    return _inner


def centroid(triangle):
    return np.mean(triangle, axis=0)

def angle_from_point(triangle, point):
    A = np.subtract(triangle[1], triangle[0])
    B = np.subtract(triangle[2], triangle[0])
    n = np.cross(A, B)
    d = np.subtract(centroid(triangle), point)
    angle = math.degrees(math.asin(
        abs(np.dot(d, n))
        /
        (np.linalg.norm(d)*np.linalg.norm(n))
    ))
    return angle

@printing_timer
def vectorized_angle_from_point(mesh, point):
    # A = np.subtract(mesh.triangles[:, 1], mesh.triangles[:, 0])
    # B = np.subtract(mesh.triangles[:, 2], mesh.triangles[:, 0])
    # n = np.cross(A, B)
    n = mesh.face_normals
    d = np.subtract(point, mesh.triangles_center)
    angle = np.degrees(np.arcsin(
        np.sum(np.multiply(d, n), axis=1)
        /
        (np.linalg.norm(d, axis=1)*np.linalg.norm(n, axis=1))
    ))
    return angle


def get_normal_1(triangle):
    A = np.subtract(triangle[1], triangle[0])
    B = np.subtract(triangle[2], triangle[0])
    return np.cross(A, B)

def get_normal_2(triangle):
    A = np.subtract(triangle[1], triangle[0])
    B = np.subtract(triangle[2], triangle[0])
    return np.cross(B, A)


def get_random_color():
    hue = random.random()
    lume = random.choice((.25, .5, .75))
    saturation = 1
    rgb = colorsys.hls_to_rgb(hue, lume, saturation)
    return list(c * 255 for c in rgb)


@dataclasses.dataclass
class SortedMesh(abc.ABC):
    triangles: np.array
    triangles_center: np.array
    triangles_cross: np.array
    face_normals: np.array
    bounding_box: np.array

@dataclasses.dataclass
class TrimeshSortedMesh(SortedMesh):
    _mesh: trimesh.Trimesh
    _sort_key: np.array

    @classmethod
    @printing_timer
    def factory(cls, path: Path) -> typing.Self:
        _mesh = trimesh.load_mesh(path)

        # _mesh.vertices -= _mesh.center_mass
        _mesh = _mesh.apply_transform(_mesh.principal_inertia_transform)

        # _mesh.fill_holes
        _mesh.fix_normals()

        _sort_key = np.argsort(_mesh.triangles_center[:, -1])[::-1]

        triangles = _mesh.triangles[_sort_key]
        triangles_center = _mesh.triangles_center[_sort_key]
        triangles_cross = _mesh.triangles_cross[_sort_key]
        face_normals = _mesh.face_normals[_sort_key]

        return cls(
            triangles,
            triangles_center,
            triangles_cross,
            face_normals,
            _mesh.bounding_box,
            _mesh,
            _sort_key,
        )


@dataclasses.dataclass
class TrianglePainter(abc.ABC):
    light_position: np.array

    @abc.abstractmethod
    def get_colors(self, mesh: SortedMesh) -> np.array:
        ...


class RandomPainter(TrianglePainter):
    def get_colors(self, mesh: SortedMesh):
        for _ in range(len(mesh.triangles)):
            yield get_random_color()

@dataclasses.dataclass
class DistanceAnglePainter(TrianglePainter):
    hue: float = config.default_hue

    def get_colors(self, mesh: SortedMesh):
        for triangle in mesh.triangles:
            angle_to_light = angle_from_point(triangle, self.light_position)
            rgb = colorsys.hls_to_rgb(self.hue / 360, angle_to_light / 90, 1)
            rgb = (c if 0 <= c <= 1 else 0 for c in rgb)
            yield [*(round(255 * c) for c in rgb), 255]


@dataclasses.dataclass
class VectorizedDistanceAnglePainter(TrianglePainter):
    hue: float = config.default_hue

    def get_colors(self, mesh: SortedMesh):
        angle_to_light = vectorized_angle_from_point(mesh, self.light_position)
        for angle in angle_to_light:
            normalized_angle = (angle + 90) / 180
            rgb = colorsys.hls_to_rgb(self.hue / 360, normalized_angle, 1)
            rgb = (c if 0 <= c <= 1 else 0 for c in rgb)
            yield [*(round(255 * c) for c in rgb), 255]


@dataclasses.dataclass
class Viewer:
    path: Path
    tag: MagicTag = dataclasses.field(default_factory=MagicTag.random_factory)
    width: int = config.width
    height: int = config.height
    canvas_width: int = config.canvas.width
    canvas_height: int = config.canvas.height
    zoom: float = config.zoom
    offset_x: float = config.offset_x
    offset_y: float = config.offset_y
    x_rotation: float = config.x_rotation
    y_rotation: float = config.y_rotation
    z_rotation: float = config.z_rotation

    def __post_init__(self):
        self.mesh = TrimeshSortedMesh.factory(self.path)


    @functools.cached_property
    def window(self):
        return Window(label=self.path.name, width=self.width, height=self.height)(
                Table(
                    header_row=False,
                    resizable=True,
                    policy=dpg.mvTable_SizingStretchProp,
                )(
                    TableColumn(),
                    TableColumn(),
                    TableRow()(
                        Text('Controls'),
                        Button('-', tag='controls_visibility', callback=self.handle_controls_visibility),
                    ),
                    TableRow(tag='controls_row')(
                        self.controls(),
                        Group(),
                    ),
                ),
                self.canvas,
            )

    @functools.cached_property
    def controls(self):
        return Group(width=-1)(
            Text('Zoom'),
            SliderFloat(tag='zoom', width=-1, default_value=self.zoom, min_value=-1.0, max_value=0.0, callback=self.handle_transform_property, user_data='zoom'),
            Text('View Offset'),
            Group(horizontal=True, indent=10, width=-1)(
                Text('x'),
                SliderFloat(tag='offset_x', min_value=-1.0, max_value=1.0, default_value=self.offset_x, callback=self.handle_transform_property, user_data='offset_x'),
            ),
            Group(horizontal=True, indent=10, width=-1)(
                Text('y'),
                SliderFloat(tag='offset_y', min_value=-1.0, max_value=1.0, default_value=self.offset_y, callback=self.handle_transform_property, user_data='offset_y'),
            ),
            Text('Rotation'),
            Group(horizontal=True, indent=10, width=-1)(
                Text('x'),
                SliderInt(tag='x_rotation', default_value=self.x_rotation, min_value=0, max_value=360, callback=self.handle_transform_property, user_data='x_rotation'),
            ),
            Group(horizontal=True, indent=10, width=-1)(
                Text('y'),
                SliderInt(tag='y_rotation', default_value=self.y_rotation, min_value=0, max_value=360, callback=self.handle_transform_property, user_data='y_rotation'),
            ),
            Group(horizontal=True, indent=10, width=-1)(
                Text('z'),
                SliderInt(tag='z_rotation', default_value=self.z_rotation, min_value=0, max_value=360, callback=self.handle_transform_property, user_data='z_rotation'),
            ),
        )

    @functools.cached_property
    def canvas(self):
        return Drawlist(width=self.canvas_width, height=self.canvas_height)(
            DrawLayer(
                tag='canvas_layer',
                depth_clipping=True,
                perspective_divide=True,
                cull_mode=dpg.mvCullMode_Back,
            )(
                DrawNode(tag='canvas_node')(
                    # dTriangle([0, 0, 0], [0, 10, 0], [10, 0, 0], color=get_random_color(), fill=get_random_color())
                ),
            ),
        )

    @functools.cached_property
    def view_matrix(self):
        return dpg.create_lookat_matrix([0, 0, 1], [0, 0, 0], [0, 1, 0])

    @property
    def projection_matrix(self):
        extents = self.mesh.bounding_box.primitive.extents
        max_extent = max(extents)
        normalized_zoom = abs(self.zoom) * 10 * max_extent
        normalized_offset_x = self.offset_x * 10 * max_extent
        normalized_offset_y = self.offset_y * 10 * max_extent

        view_left = -((max_extent/2) + normalized_zoom) + normalized_offset_x
        view_right = (max_extent/2) + normalized_zoom + normalized_offset_x
        view_bottom = -((max_extent/2) + normalized_zoom) + normalized_offset_y
        view_top = (max_extent/2) + normalized_zoom + normalized_offset_y

        return dpg.create_orthographic_matrix(view_left, view_right, view_bottom, view_top, 0.1, 100)

    @property
    def rotation_matrix(self):
        return dpg.create_rotation_matrix(math.pi*self.x_rotation/180.0 , [1, 0, 0])*\
            dpg.create_rotation_matrix(math.pi*self.y_rotation/180.0 , [0, 1, 0])*\
            dpg.create_rotation_matrix(math.pi*self.z_rotation/180.0 , [0, 0, 1])

    @printing_timer
    def render(self):
        self.window.render(tag_prefix=self.tag)
        self.render_mesh()
        dpg.set_clip_space(self.tag / 'canvas_layer', 0, 0, self.canvas_width, self.canvas_height, -100.0, 100.0)
        self.apply_transform()


    @printing_timer
    def render_mesh(self):
        transparent = [0, 0, 0, 0]

        bounding_box = self.mesh.bounding_box
        light_position = np.max(bounding_box.vertices, axis=0) * 2
        painter = VectorizedDistanceAnglePainter(light_position)

        with self.canvas.tagged_entities['canvas_node']:
            for triangle, fill in zip(self.mesh.triangles, painter.get_colors(self.mesh)):
                dpg.draw_triangle(*triangle, fill=fill, color=transparent)


    def apply_transform(self):
        dpg.apply_transform(self.tag /'canvas_node', self.projection_matrix*self.view_matrix*self.rotation_matrix)

    ## Callbacks
    def handle_transform_property(self, sender, app_data, user_data):
        setattr(self, user_data, app_data)
        self.apply_transform()

    def handle_zoom(self, sender, app_data, user_data):
        self.zoom = app_data
        self.apply_transform()

    #                               eye, origin, up
    # view = dpg.create_lookat_matrix([0, 0, 1], [0, 0, 0], [0, 1, 0])
    def handle_transform(self, sender, app_data, user_data):
        pass

    def handle_controls_visibility(self, sender, app_data, user_data):
        row = self.window.tagged_entities['controls_row']
        button = self.window.tagged_entities['controls_visibility']
        visible = not row.show
        row.show = not row.show
        button.label = '-' if visible else '...'
