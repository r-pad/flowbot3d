import copy
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)
from xml.etree import ElementTree as ET

import numpy as np
import numpy.typing as npt
import trimesh
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from scipy.spatial.transform import Rotation

try:
    pass
except:
    print("NOTE: SAPIEN is not installed, so parital pointcloud sampling won't work.")

JOINT_TYPES = {"slider", "free", "hinge", "heavy", "static", "slider+"}
JointType = Literal["slider", "free", "hinge", "heavy", "static", "slider+"]


@dataclass
class JointSemantic:
    name: str
    type: JointType
    label: str


@dataclass
class Semantics:
    """TODO: Describe the semantics file."""

    sems: List[JointSemantic]

    def by_name(self, name: str) -> JointSemantic:
        return {semantic.name: semantic for semantic in self.sems}[name]

    def by_type(
        self, joint_type: Union[JointType, Sequence[JointType]]
    ) -> List[JointSemantic]:
        if isinstance(joint_type, str):
            joint_types = {joint_type}
        else:
            joint_types = set(joint_type)
        return [sem for sem in self.sems if sem.type in joint_types]

    def by_label(self, label: str) -> List[JointSemantic]:
        return [sem for sem in self.sems if sem.label == label]

    @staticmethod
    def from_file(fn: Union[str, Path]) -> "Semantics":
        path = Path(fn)
        with path.open("r") as f:
            lines = f.read().split("\n")

        # Remove all empty lines.
        lines = [line for line in lines if line.strip()]
        semantics = []
        for line in lines:
            name, jt, sem = line.split(" ")
            if jt not in JOINT_TYPES:
                raise ValueError("bad file for parsing semantics...")
            jt = cast(JointType, jt)  # it passes parsing
            semantics.append(JointSemantic(name, jt, sem))

        # assert no duplicates.
        names = {semantic.name for semantic in semantics}
        assert len(names) == len(semantics)

        return Semantics(semantics)


@dataclass
class Metadata:
    """This represents the metadata file"""

    model_cat: str

    @staticmethod
    def from_file(fn: Union[str, Path]) -> "Metadata":
        path = Path(fn)
        with path.open("r") as f:
            raw_metadata = json.load(f)
        return Metadata(model_cat=raw_metadata["model_cat"])


class PartialPC(TypedDict):
    pos: npt.NDArray[np.float32]
    seg: npt.NDArray[np.uint]
    frame: Literal["world", "camera"]
    T_world_cam: npt.NDArray[np.float32]
    T_world_base: npt.NDArray[np.float32]
    # proj_matrix: npt.NDArray[np.float32]
    labelmap: Dict[str, int]
    angles: Dict[str, float]


class PMRawData:
    """This class describes the grouping of files for each object."""

    def __init__(self, obj_dir: Union[str, Path]):
        self.obj_dir = Path(obj_dir)

        # Load the data in.
        self.semantics = Semantics.from_file(self.semantics_fn)
        self.metadata = Metadata.from_file(self.meta_fn)
        self.obj = parse_urdf(self.urdf_fn)

        self.__issubset: Optional[bool] = None
        self.__issame: Optional[bool] = None

        # Render environment. Only create it if needed.
        self.__render_env: Optional[PMRenderEnv] = None

    @property
    def obj_id(self) -> str:
        return self.obj_dir.name

    @property
    def category(self) -> str:
        return self.metadata.model_cat

    @property
    def semantics_fn(self) -> Path:
        """The semantics file"""
        return self.obj_dir / "semantics.txt"

    @property
    def urdf_fn(self) -> Path:
        return self.obj_dir / "mobility.urdf"

    @property
    def meta_fn(self) -> Path:
        return self.obj_dir / "meta.json"

    @property
    def original_dataset(self) -> Literal["partnet", "shapenet"]:
        raise NotImplementedError("not implemented yet")

    def _evaluate_meshes(self) -> Tuple[bool, bool]:
        """Get information about whether or not the URDF agrees with the textured_objs directory."""
        meshes = self.obj.meshes

        objs = list((self.obj_dir / "textured_objs").glob("*.obj"))
        obj_bases = set([f"textured_objs/{path.name}" for path in objs])

        # Is same == are the meshes claimed in the urdf the same as are included
        # in the dataset?
        issame = meshes == obj_bases

        # issubset == are teh meshes claimed in the urdf at least a subset of the dataset?
        issubset = meshes.issubset(obj_bases)

        return issubset, issame

    @property
    def has_extra_objs(self) -> bool:
        """Some of the items in the dataset have extra obj files in their textured_objs directory."""
        if self.__issubset is None or self.__issame is None:
            self.__issubset, self.__issame = self._evaluate_meshes()

        return self.__issubset and not self.__issame  # type: ignore

    @property
    def well_formed(self) -> bool:
        """Does the set of objs in the textured_objs folder exactly match those described in the URDF?"""
        if self.__issubset is None or self.__issame is None:
            self.__issubset, self.__issame = self._evaluate_meshes()
        return self.__issame  # type: ignore

    @property
    def usable(self) -> bool:
        """Are all the objects specified in the URDF contianed in the textured_objs folder?"""
        if self.__issubset is None or self.__issame is None:
            self.__issubset, self.__issame = self._evaluate_meshes()
        return self.__issubset  # type: ignore

    def __repr__(self) -> str:
        return f'PMRawData(id="{self.obj_id}")'

    def sample_full_pointcloud(
        self, n_points: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:

        # First, load all the meshes for the object.
        # link_name, link_mesh (there may be many)
        # Also, they might be scenes, so we should flatten.
        link_meshes_unflattened = []
        for link in self.obj.links:
            obj_files = [os.path.join(self.obj_dir, f) for f in link.mesh_names]
            link_meshes_unflattened.extend(
                [(link.name, trimesh.load(obj_file)) for obj_file in obj_files]
            )

        # We need to flatten.
        link_meshes: List[Tuple[str, trimesh.Trimesh]] = []
        for link_name, link_mesh_or_scene in link_meshes_unflattened:
            if isinstance(link_mesh_or_scene, trimesh.Scene):
                scene_meshes = list(link_mesh_or_scene.geometry.values())
                link_meshes.extend([(link_name, mesh) for mesh in scene_meshes])
            elif isinstance(link_mesh_or_scene, trimesh.Trimesh):
                link_meshes.append((link_name, link_mesh_or_scene))
            else:
                raise ValueError("we are getting a mesh type we don't understand")

        # Next, compute the relative areas, and buckets to know how many points
        # to sample per mesh.
        mesh_areas = np.asarray([mtup[1].area for mtup in link_meshes])
        ratios = mesh_areas / mesh_areas.sum()
        buckets = np.floor((n_points * ratios)).astype(int)
        while buckets.sum() < n_points:
            buckets[-1] += 1

        # Finally, sample each mesh recursively.
        points = []
        # colors = []
        normals = []
        ins = []
        sem = []
        art: List[str] = []

        for bucket, (link_name, mesh) in zip(buckets, link_meshes):
            mesh_points, face_indices = trimesh.sample.sample_surface(mesh, bucket)
            face_normals = mesh.face_normals[face_indices]

            # Extract colors. Doesn't seem to work right now.
            # visual = mesh.visual
            # if isinstance(visual, trimesh.visual.TextureVisuals):
            #     color_visual: trimesh.visual.ColorVisuals = visual.to_color()
            # else:
            #     color_visual = visual
            # breakpoint()
            # face_colors = color_visual.face_colors[face_indices]

            normals.append(face_normals)
            points.append(mesh_points)
            # colors.append(face_colors)
            ins.extend([link_name] * len(mesh_points))
            sem.extend([self.semantics.by_name(link_name).label] * len(mesh_points))
            art.extend([self.semantics.by_name(link_name).type] * len(mesh_points))

        pos: np.ndarray = np.concatenate(points, axis=0)
        norm: np.ndarray = np.concatenate(normals, axis=0)

        # Lastly, apply the base transformation.
        # All the points are in the same frame, except that there's a base transform.
        bj = [j for j in self.obj.joints if j.parent == "base"]
        assert len(bj) == 1
        base_joint = bj[0]

        # Extract a transform.
        assert base_joint.origin is not None
        xyz, rpy = base_joint.origin
        T_base_obj = np.eye(4)
        T_base_obj[:3, 3] = xyz
        T_base_obj[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()

        # Apply.
        pos = pos @ T_base_obj[:3, :3].T + xyz.reshape((1, 3))
        norm = norm @ T_base_obj[:3, :3].T

        return pos, norm, ins, sem, art

    def sample_partial_pointcloud(
        self,
        randomize_joints: Union[Literal[False], Literal["all"], Sequence[str]] = False,
        randomize_camera: bool = False,
        set_joints: Union[Literal[False], Sequence[Tuple[str, float]]] = False,
    ) -> PartialPC:
        """Sample a partial pointcloud using the Pybullet GL renderer. Currently only supports
        randomized parameters.

        Args:
            randomize_joints: Decide whether and how to randomize joints. Defaults to False.
                False -> no randomization
                "all" -> Randomize all joints on the object
                [list] -> Randomize just these joints
            randomize_camera (bool, optional): Randomize the camera position. Defaults to False.
                Only occurs in a window.
            set_joints: Decide whether and how to set the joints. Can't also randomize the joints.

        Returns:
            PartialPC: A big dictionary of things. See PartialPC above for what you get.
        """
        if self.__render_env is None:
            self.__render_env = PMRenderEnv(self.obj_dir.name, str(self.obj_dir.parent))

        if randomize_joints and set_joints:
            raise ValueError("unable to randomize and set joints")
        if randomize_joints:
            if randomize_joints == "all":
                self.__render_env.randomize_joints()
            else:
                self.__render_env.randomize_joints(randomize_joints)
        if set_joints:
            self.__render_env.set_joint_angles({jn: ja for jn, ja in set_joints})
        if randomize_camera:
            self.__render_env.randomize_camera()

        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = self.__render_env.render()

        # Reindex the segmentation.
        pc_seg_obj = np.ones_like(pc_seg) * -1
        for k, (body, link) in segmap.items():
            if body == self.__render_env.obj_id:
                ixs = pc_seg == k
                pc_seg_obj[ixs] = link

        return {
            "pos": P_world,
            "seg": pc_seg_obj,
            "frame": "world",
            "T_world_cam": self.__render_env.camera.T_world2cam,
            "T_world_base": np.copy(self.__render_env.T_world_base),
            # "proj_matrix": None,
            "labelmap": self.__render_env.link_name_to_index,
            "angles": self.__render_env.get_joint_angles(),
        }


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: Optional[Tuple[np.ndarray, np.ndarray]] = None
    axis: Optional[np.ndarray] = None
    limit: Optional[Tuple[float, float]] = None


@dataclass
class Link:
    name: str

    # Since the collision geometries and the visual geometries are the same
    # for all the data, we just call it meshes. They're also always at 0 0 0.
    mesh_names: Set[str]


class PMObject:
    ROOT_LINK = "base"  # specific to partnet-mobility, all objs have a base obj

    def __init__(self, links: List[Link], joints: List[Joint]):

        self.__links = links
        self.__joints = joints
        self.__linkmap = {link.name: ix for ix, link in enumerate(self.__links)}
        self.__childmap = {joint.child: ix for ix, joint in enumerate(self.__joints)}
        self.__jointmap = {joint.name: ix for ix, joint in enumerate(self.__joints)}

    def get_chain(self, link_name: str) -> List[Joint]:
        parent_dict: Dict[str, Union[Tuple[str, Joint], Tuple[None, None]]]
        parent_dict = {joint.child: (joint.parent, joint) for joint in self.joints}
        parent_dict["base"] = None, None

        parent_name, parent_link = parent_dict[link_name]
        parents: List[Joint] = []
        while parent_name is not None and parent_link is not None:
            parents = [parent_link] + parents
            parent_name, parent_link = parent_dict[parent_name]

        return parents

    @property
    def links(self) -> List[Link]:
        return copy.deepcopy(self.__links)

    @property
    def joints(self) -> List[Joint]:
        return copy.deepcopy(self.__joints)

    def get_joint(self, name: str) -> Joint:
        if name not in self.__jointmap:
            raise ValueError(f"invalid joint name: {name}")
        return copy.deepcopy(self.__joints[self.__jointmap[name]])

    def get_joint_by_child(self, child_name: str) -> Joint:
        if child_name not in self.__childmap:
            raise ValueError(f"invalid child name: {child_name}")
        return copy.deepcopy(self.__joints[self.__childmap[child_name]])

    def get_link(self, name: str) -> Link:
        if name not in self.__linkmap:
            raise ValueError(f"invalid link name: {name}")
        return copy.deepcopy(self.__links[self.__linkmap[name]])

    @property
    def children(self) -> Dict[str, List[str]]:
        children: Dict[str, List[str]] = defaultdict(list)
        for link in self.__links:
            children[link.name] = []
        for joint in self.__joints:
            children[joint.parent].append(joint.child)
        return dict(children)

    @property
    def descendents(self) -> Dict[str, List[str]]:
        descendents = defaultdict(list)
        children = self.children
        link_dict = {link.name: link for link in self.__links}

        def dfs(link, ancestor_keys):
            for key in ancestor_keys:
                descendents[key].append(link.name)

            descendents[link.name] = []
            for child in children[link.name]:
                dfs(link_dict[child], ancestor_keys + [link.name])

        dfs(link_dict[self.ROOT_LINK], [])
        return dict(descendents)

    def __str__(self) -> str:
        return f"PMObject(links={len(self.__links)}, joints={len(self.__joints)})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def meshes(self) -> Set[str]:
        return reduce(lambda a, b: a.union(b.mesh_names), self.links, set())


def parse_urdf_from_string(urdf_string: str) -> PMObject:
    robot = ET.fromstring(urdf_string)

    def parse_pose(element: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
        xyz = (
            np.asarray(element.attrib["xyz"].split(" "), dtype=float)
            if "xyz" in element.attrib
            else np.asarray([0.0, 0.0, 0.0])
        )
        rpy = (
            np.asarray(element.attrib["rpy"].split(" "), dtype=float)
            if "rpy" in element.attrib
            else np.asarray([0.0, 0.0, 0.0])
        )
        return xyz, rpy

    def parse_link(link_et: ET.Element) -> Link:
        link_name = link_et.attrib["name"]
        # Recursively (via iter()) grab the meshes.
        meshes = {it.attrib["filename"] for it in link_et.iter() if it.tag == "mesh"}
        return Link(name=link_name, mesh_names=meshes)

    def parse_joint(joint_et: ET.Element) -> Joint:
        joint_name = joint_et.attrib["name"]
        joint_type = joint_et.attrib["type"]
        child = joint_et.find("child").attrib["link"]  # type: ignore
        parent = joint_et.find("parent").attrib["link"]  # type: ignore

        # Parse the optional fields.
        origin_et = joint_et.find("origin")
        origin = parse_pose(origin_et) if origin_et is not None else None
        axis_et = joint_et.find("axis")

        # There are a number of malformed entries (i.e. joint_0 in 103252/mobility.urdf)
        # where we need to replace None -> 0
        axis: Optional[np.ndarray]
        if axis_et is not None:
            xyzstrs = axis_et.attrib["xyz"].split(" ")
            xyzstrs = [xyzstr if xyzstr != "None" else "0" for xyzstr in xyzstrs]
            axis = np.asarray(xyzstrs, dtype=float)
        else:
            axis = None

        limit_et = joint_et.find("limit")
        limit = (
            (float(limit_et.attrib["lower"]), float(limit_et.attrib["upper"]))
            if limit_et is not None
            else None
        )

        return Joint(
            name=joint_name,
            type=joint_type,
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,
            limit=limit,
        )

    link_ets = robot.findall("link")
    joint_ets = robot.findall("joint")

    links = [parse_link(link_et) for link_et in link_ets]
    joints = [parse_joint(joint_et) for joint_et in joint_ets]

    return PMObject(links, joints)


def parse_urdf(urdf_fn: Union[str, Path]) -> PMObject:
    urdf_path = Path(urdf_fn)
    if not (urdf_path.exists() and urdf_path.suffix == ".urdf"):
        raise ValueError(f"{urdf_path} is not a URDF file")

    with urdf_path.open("r") as f:
        contents = f.read()
    return parse_urdf_from_string(contents)
