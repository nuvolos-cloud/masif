#!/usr/bin/python
import numpy as np
import os
import shutil
import logging

from Bio.PDB import *
import sys

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)

if len(sys.argv) <= 1:
    logger.error("Usage: {config} " + sys.argv[0] + " PDBID_A")
    logger.error("A or AB are the chains to include in this surface.")
    sys.exit(1)


# Save the chains as separate files.
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

if (len(sys.argv) > 2) and (sys.argv[2] == "masif_ligand"):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"], pdb_id + ".pdb")
else:
    pdb_filename = masif_opts["raw_pdb_dir"] + pdb_id + ".pdb"
tmp_dir = masif_opts["tmp_dir"]
protonated_file = tmp_dir + "/" + pdb_id + ".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

logger.info(f"Protonated file: {pdb_filename}")

# Extract chains of interest.
out_filename1 = tmp_dir + "/" + pdb_id + "_" + chain_ids1
extractPDB(pdb_filename, out_filename1 + ".pdb", chain_ids1)

logger.info(f"Extracted PDB file: {out_filename1+'.pdb'}")

# Compute MSMS of surface w/hydrogens,
vertices1, faces1, normals1, names1, areas1 = computeMSMS(
    out_filename1 + ".pdb", protonate=True
)
logger.info("MSMS computed")

# Compute "charged" vertices
if masif_opts["use_hbond"]:
    logger.info("Computing charges")
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)
    logger.info("Charges computed")

# For each surface residue, assign the hydrophobicity of its amino acid.
if masif_opts["use_hphob"]:
    logger.info("Computing hydrophobicity")
    vertex_hphobicity = computeHydrophobicity(names1)
    logger.info("Computing hydrophobicity done")

# If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
vertices2 = vertices1
faces2 = faces1

# Fix the mesh.
logger.info("Fixing mesh")
mesh = pymesh.form_mesh(vertices2, faces2)
regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])
logger.info("Fixing mesh done")

# Compute the normals
logger.info("Computing normals")
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
# Assign charges on new vertices based on charges of old vertices (nearest
# neighbor)
logger.info("Computing normals done")

if masif_opts["use_hbond"]:
    logger.info("Assigning charges")
    vertex_hbond = assignChargesToNewMesh(
        regular_mesh.vertices, vertices1, vertex_hbond, masif_opts
    )
    logger.info("Assigning charges done")

if masif_opts["use_hphob"]:
    logger.info("Assigning charges hidro")
    vertex_hphobicity = assignChargesToNewMesh(
        regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts
    )
    logger.info("Assigning charges hidro done")

if masif_opts["use_apbs"]:
    logger.info(f"Computing APBS for {out_filename1+'.pdb'}")
    vertex_charges = computeAPBS(
        regular_mesh.vertices, out_filename1 + ".pdb", out_filename1
    )
    logger.info("Computing APBS done")

iface = np.zeros(len(regular_mesh.vertices))
if "compute_iface" in masif_opts and masif_opts["compute_iface"]:
    # Compute the surface of the entire complex and from that compute the interface.
    logger.info("Computing MSMS2")
    v3, f3, _, _, _ = computeMSMS(pdb_filename, protonate=True)
    logger.info("Computing MSMS2 done")
    # Regularize the mesh
    logger.info("Regularizing mesh")
    mesh = pymesh.form_mesh(v3, f3)
    logger.info("Regularizing mesh done")
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d)  # Square d, because this is how it was in the pyflann version.
    assert len(d) == len(regular_mesh.vertices)
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    logger.info("Saving ply1")
    # Convert to ply and save.
    save_ply(
        out_filename1 + ".ply",
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        hbond=vertex_hbond,
        hphob=vertex_hphobicity,
        iface=iface,
    )
    logger.info("Saving ply1 done")

else:
    logger.info("Saving ply2")
    # Convert to ply and save.
    save_ply(
        out_filename1 + ".ply",
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        hbond=vertex_hbond,
        hphob=vertex_hphobicity,
    )
    logger.info("Saving ply2 done")
if not os.path.exists(masif_opts["ply_chain_dir"]):
    os.makedirs(masif_opts["ply_chain_dir"])
if not os.path.exists(masif_opts["pdb_chain_dir"]):
    os.makedirs(masif_opts["pdb_chain_dir"])
shutil.copy(out_filename1 + ".ply", masif_opts["ply_chain_dir"])
shutil.copy(out_filename1 + ".pdb", masif_opts["pdb_chain_dir"])
