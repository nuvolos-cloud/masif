import os
import numpy
import logging
from subprocess import Popen, PIPE


from default_config.global_vars import apbs_bin, pdb2pqr_bin, multivalue_bin

"""
computeAPBS.py: Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

logger = logging.getLogger(__name__)


def computeAPBS(vertices, pdb_file, tmp_file_base):
    """
    Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    """
    fields = tmp_file_base.split("/")[0:-1]
    directory = "/".join(fields) + "/"
    filename_base = tmp_file_base.split("/")[-1]
    pdbname = pdb_file.split("/")[-1]
    args = [
        pdb2pqr_bin,
        "--ff=PARSE",
        "--whitespace",
        "--noopt",
        "--apbs-input",
        filename_base + ".in",
        pdbname,
        filename_base + ".pqr",
    ]
    logger.debug("Running pdb2pqr_bin: {args}")
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()
    logger.debug("Running pdb2pqr_bin done")

    args = [apbs_bin, filename_base + ".in"]
    logger.debug("Running apbs: {args}")
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()
    logger.debug("Running apbs done")

    vertfile = open(directory + "/" + filename_base + ".csv", "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
    vertfile.close()

    args = [
        multivalue_bin,
        filename_base + ".csv",
        filename_base + ".pqr.dx",
        filename_base + "_out.csv",
    ]
    logger.debug("Running multivalue: {args}")
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()
    logger.debug("Running multivalue done")

    # Read the charge file
    logger.debug("Reading the charge file: {args}")
    chargefile = open(tmp_file_base + "_out.csv")
    charges = numpy.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])
    chargefile.close()
    logger.debug("Chargefile closed.")

    remove_fn = os.path.join(directory, filename_base)
    # os.remove(remove_fn, )
    os.remove(remove_fn + ".csv")
    os.remove(remove_fn + ".pqr.dx")
    os.remove(remove_fn + ".in")
    # os.remove(remove_fn+'-input.p')
    os.remove(remove_fn + "_out.csv")

    return charges
