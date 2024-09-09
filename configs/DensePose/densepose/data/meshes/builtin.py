# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# pyre-unsafe

from .catalog import MeshInfo, register_meshes

DENSEPOSE_MESHES_DIR = "https://dl.fbaipublicfiles.com/densepose/meshes/"

MESHES = [
    MeshInfo(
        name="smpl_27554",
        data="smpl_27554.pkl",
        geodists="geodists/geodists_smpl_27554.pkl",
        symmetry="symmetry/symmetry_smpl_27554.pkl",
        texcoords="texcoords/texcoords_smpl_27554.pkl",
    ),
    MeshInfo(
        name="chimp_5029",
        data="chimp_5029.pkl",
        geodists="geodists/geodists_chimp_5029.pkl",
        symmetry="symmetry/symmetry_chimp_5029.pkl",
        texcoords="texcoords/texcoords_chimp_5029.pkl",
    ),
    MeshInfo(
        name="cat_5001",
        data="cat_5001.pkl",
        geodists="geodists/geodists_cat_5001.pkl",
        symmetry="symmetry/symmetry_cat_5001.pkl",
        texcoords="texcoords/texcoords_cat_5001.pkl",
    ),
    MeshInfo(
        name="cat_7466",
        data="cat_7466.pkl",
        geodists="geodists/geodists_cat_7466.pkl",
        symmetry="symmetry/symmetry_cat_7466.pkl",
        texcoords="texcoords/texcoords_cat_7466.pkl",
    ),
    MeshInfo(
        name="sheep_5004",
        data="sheep_5004.pkl",
        geodists="geodists/geodists_sheep_5004.pkl",
        symmetry="symmetry/symmetry_sheep_5004.pkl",
        texcoords="texcoords/texcoords_sheep_5004.pkl",
    ),
    MeshInfo(
        name="zebra_5002",
        data="zebra_5002.pkl",
        geodists="geodists/geodists_zebra_5002.pkl",
        symmetry="symmetry/symmetry_zebra_5002.pkl",
        texcoords="texcoords/texcoords_zebra_5002.pkl",
    ),
    MeshInfo(
        name="horse_5004",
        data="horse_5004.pkl",
        geodists="geodists/geodists_horse_5004.pkl",
        symmetry="symmetry/symmetry_horse_5004.pkl",
        texcoords="texcoords/texcoords_zebra_5002.pkl",
    ),
    MeshInfo(
        name="giraffe_5002",
        data="giraffe_5002.pkl",
        geodists="geodists/geodists_giraffe_5002.pkl",
        symmetry="symmetry/symmetry_giraffe_5002.pkl",
        texcoords="texcoords/texcoords_giraffe_5002.pkl",
    ),
    MeshInfo(
        name="elephant_5002",
        data="elephant_5002.pkl",
        geodists="geodists/geodists_elephant_5002.pkl",
        symmetry="symmetry/symmetry_elephant_5002.pkl",
        texcoords="texcoords/texcoords_elephant_5002.pkl",
    ),
    MeshInfo(
        name="dog_5002",
        data="dog_5002.pkl",
        geodists="geodists/geodists_dog_5002.pkl",
        symmetry="symmetry/symmetry_dog_5002.pkl",
        texcoords="texcoords/texcoords_dog_5002.pkl",
    ),
    MeshInfo(
        name="dog_7466",
        data="dog_7466.pkl",
        geodists="geodists/geodists_dog_7466.pkl",
        symmetry="symmetry/symmetry_dog_7466.pkl",
        texcoords="texcoords/texcoords_dog_7466.pkl",
    ),
    MeshInfo(
        name="cow_5002",
        data="cow_5002.pkl",
        geodists="geodists/geodists_cow_5002.pkl",
        symmetry="symmetry/symmetry_cow_5002.pkl",
        texcoords="texcoords/texcoords_cow_5002.pkl",
    ),
    MeshInfo(
        name="bear_4936",
        data="bear_4936.pkl",
        geodists="geodists/geodists_bear_4936.pkl",
        symmetry="symmetry/symmetry_bear_4936.pkl",
        texcoords="texcoords/texcoords_bear_4936.pkl",
    ),
]

register_meshes(MESHES, DENSEPOSE_MESHES_DIR)
