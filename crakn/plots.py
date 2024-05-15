import pandas as pd
from amd import CifReader
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from matplotlib import pyplot as plt
from pymatgen.io.cif import CifParser
import os
import pickle
import amd
import numpy as np
from scipy.spatial.distance import pdist
from matplotlib.ticker import StrMethodFormatter
from tqdm import tqdm
import seaborn

def find_line_function(p1, p2, verbose=False, return_slope_and_intercept=False):
    slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
    intercept = p1[1] - slope * p1[0]
    fn = lambda x: slope * x + intercept
    if verbose:
        print(f"y = {slope} x + {intercept}")
    if return_slope_and_intercept:
        return (slope, intercept)
    return fn

def find_intersection(corners, line_fn, current_point, return_index=False):
    i = (0, 0)
    int_indx = -1
    for indx, corner in enumerate(corners):
        y = line_fn(corner[0])
        if corner[0] >= current_point[0]:
            break
        if y < corner[1]:
            i = (corner[0], y)
            int_indx = indx
            print(f"Intersection found at {int_indx}: {i}")
    if return_index:
        return int_indx
    return i

def get_data(dataset, target, id_tag="jid"):
    d = data(dataset)
    structures = []
    targets = []
    ids = []
    current_id = 0
    for datum in tqdm(d, desc="Retrieving data.."):
        val = datum[target]
        if isinstance(val, float):
            targets.append(val)
        else:
            continue

        if id_tag in datum:
            ids.append(datum[id_tag])
        else:
            ids.append(current_id)
            current_id += 1

        atoms = (Atoms.from_dict(datum["atoms"]) if isinstance(datum["atoms"], dict) else datum["atoms"])
        structure = atoms.pymatgen_converter()
        ps = amd.periodicset_from_pymatgen_structure(structure)
        structures.append(ps)

    print(len(structures))
    print(len(targets))
    return structures, targets, ids


def read_cif(path: str):
    crystals = CifParser(path).get_structures()


def plot_SPR_jarvis(zoomed=False, closest=10000):
    properties = ["formation_energy_peratom", "optb88vdw_bandgap", "mbj_bandgap", "exfoliation_energy"]
    label_names = {
        "formation_energy_peratom": "Formation Energy  (eV / atom)",
        "optb88vdw_bandgap": "Band Gap Energy (OPT) (eV)",
        "mbj_bandgap": "Band Gap Energy (MBJ) (eV)",
        "exfoliation_energy": "Exfoliation Energy (meV/cell)"
    }

    color = {
        "formation_energy_peratom": "red",
        "optb88vdw_bandgap": "green",
        "mbj_bandgap": "black",
        "exfoliation_energy": "purple"
    }

    for prop in properties:
        ps, targets, ids = get_data('dft_2d', prop)
        le = targets

        #to_keep = [i for i in range(len(target_values)) if target_values[i] != "na"]
        #print(f"{prop} Data size: {len(to_keep)}")
        #if len(to_keep) > 10000:
        #    continue
        #ps = [periodic_sets[i] for i in to_keep]
        #le = [target_values[i] for i in to_keep]

        cutoff = 0.01
        amds = [amd.AMD(p, k=100).astype(np.float32) for p in ps]

        distances = amd.AMD_pdist(amds, low_memory=True)
        inds = np.where(distances < cutoff)[0]
        distances = distances[inds]
        le = np.array(le)
        fe_diffs = pdist(le.reshape((-1, 1)))[inds]

        plt.figure(figsize=(30, 20))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        pl = seaborn.scatterplot(x=distances, y=fe_diffs, color=color[prop])
        pl.set_xlabel("L-inf on AMD at k = 100 (Angstroms)", fontsize=46)
        pl.set_ylabel(f"Absolute Difference in {label_names[prop]}", fontsize=46)
        plt.xlim([0, cutoff])
        pl.set_yticklabels(np.round(pl.get_yticks(), 2), size=30)
        pl.set_xticklabels(np.round(pl.get_xticks(), 2), size=30)
        plt.savefig(f"./figures/{prop}_spr.png")

        x = distances
        y = fe_diffs

        corner_points = []
        internal_corners = []
        for a, b in zip(x, y):
            point_qualifies = not np.any((x < a) & (y > b))
            if point_qualifies:
                corner_points.append((a, b))

        corner_points.sort(key=lambda x: x[0])
        for i in range(1, len(corner_points)):
            b = corner_points[i - 1][1]
            a = corner_points[i][0]
            internal_corners.append((a, b))

        internal_corners = corner_points.copy()
        internal_corner_slopes = [i[1] / i[0] for i in internal_corners]
        max_slope = np.max(internal_corner_slopes)
        expr = []
        log_convexity = []
        for i in range(1, len(internal_corner_slopes) - 1):
            expr.append(internal_corner_slopes[i - 1] + internal_corner_slopes[i + 1] - 2 * internal_corner_slopes[i])
        expr = [internal_corner_slopes[1] - 2 * internal_corner_slopes[0]] + expr + [
            internal_corner_slopes[-2] - 2 * internal_corner_slopes[-1]]

        df = pd.DataFrame({"x": [i[0] for i in internal_corners], "y": [i[1] for i in internal_corners],
                           "slope=y/x": internal_corner_slopes, "convexity": expr})
        df["log(slope)"] = np.log(df["slope=y/x"])
        b_n = list(np.log(df["slope=y/x"]))
        jumps = [internal_corners[0][1]]
        for i in range(1, len(internal_corners)):
            jumps.append(internal_corners[i][1] - internal_corners[i - 1][1])

        for i in range(1, len(b_n) - 1):
            log_convexity.append(b_n[i - 1] + b_n[i + 1] - 2 * b_n[i])
        log_convexity = [b_n[1] - 2 * b_n[0]] + list(log_convexity) + [b_n[len(b_n) - 2] - 2 * b_n[len(b_n) - 3]]
        df["log-convexity"] = log_convexity
        df["normalized angle"] = np.arctan(df["slope=y/x"] / max_slope) * 180 / np.pi
        a_n = df["normalized angle"].to_numpy()
        a_n = np.concatenate([a_n, [0]])
        df["angular jump"] = a_n[1:] - a_n[:-1]
        df.sort_values("x").to_csv(f"./figures/{prop}_external_points.csv", index=False)
        original_corners = corner_points.copy()
        # Plot staircase
        for i, corner_point in enumerate(corner_points):
            if i == 0:
                plt.plot([0, corner_point[0]], [0, 0], color="blue")
                plt.plot([corner_point[0], corner_point[0]], [0, corner_point[1]], color="blue")
            else:
                plt.plot([corner_points[i - 1][0], corner_point[0]], [corner_points[i - 1][1], corner_points[i - 1][1]],
                         color="blue")
                plt.plot([corner_point[0], corner_point[0]], [corner_points[i - 1][1], corner_points[i][1]],
                         color="blue")

        internal_corner_slopes = np.array(internal_corner_slopes)
        internal_corners.sort(key=lambda x: x[0])
        arg_for_SPF = np.argmax(list(df["angular jump"])[1:])
        internal_corner_for_SPF = internal_corners[1:][arg_for_SPF]
        adj_corner = [c for c in corner_points if c[1] == internal_corner_for_SPF[1]][0]
        print(f"next_internal_corner: {adj_corner}")
        print(f"internal_corner: {internal_corners[1:][arg_for_SPF]}")
        if adj_corner == (0, 0):
            print(f"Skipping {prop}")
            continue
        SPF = list(df["slope=y/x"])[1:][arg_for_SPF]
        initial_SPF = SPF
        SPD = adj_corner[1] - SPF * adj_corner[0]
        SPD = max(SPD, 0)
        print(f"y = {np.round(SPF, 3)} x + {np.round(SPD, 3)}")
        line_function = lambda x: SPF * x + SPD
        # corner_points = stage1(corner_points)
        # corner_points = stage2(corner_points)
        SPB = [corner for corner in corner_points if
               line_function(corner[0]) < corner[1] and corner[0] >= adj_corner[0]]
        if len(SPB) == 0:
            SPB = 0
        else:
            SPB = SPB[0][0]
        point_before_SPB = [corner for corner in corner_points if corner[0] < SPB][-1]
        print(f"Using an SPB of : {np.round(SPB, 4)}")
        potential_SPF = SPF

        if SPD > 0:
            while True:
                intersection_index = find_intersection(original_corners, line_function, (SPB, 0), return_index=True)
                print(
                    f"previous line equation: y = {np.round(line_function(1) - line_function(0), 3)} x + {np.round(line_function(0), 3)}")
                if intersection_index == -1:
                    break
                c, d = original_corners[intersection_index]
                amount_to_shift = (d - line_function(c)) + line_function(0)
                line_function = lambda x: potential_SPF * x + amount_to_shift
                print(f"y = {np.round(potential_SPF, 3)} x + {np.round(amount_to_shift, 3)}")
                print(f"Intersects at {np.round((c, d), 3)}")
                SPD = line_function(0)
                SPF = potential_SPF

        highlighted_points = [point for point in corner_points if
                              (point[0] == SPB and point[1] >= line_function(SPB)) or line_function(point[0]) == point[
                                  1]]
        p1 = adj_corner
        points_to_consider_for_p2 = [point for point in corner_points if point[0] < p1[0]]
        potential_slope_and_intercepts = [find_line_function(p1, point, return_slope_and_intercept=True) for point in
                                          points_to_consider_for_p2]
        min_slope = np.argmin([p[0] for p in potential_slope_and_intercepts])
        SPF, SPD = potential_slope_and_intercepts[min_slope]
        p2 = points_to_consider_for_p2[min_slope]
        print("-------------------------------------")
        print("Optimization:")
        print(f"Using points: {p1} and {p2}")
        line_function = find_line_function(p1, p2, verbose=True)
        print(f"Checking line function: {line_function(p1[0]) - p1[1]} and {line_function(p2[0]) - p2[1]}")
        print("-------------------------------------")
        SPD = line_function(0)
        if SPD < 0:
            SPD = 0
            line_function = find_line_function(p1, (0, 0))
        SPB = [corner[0] for corner in corner_points if
               line_function(corner[0]) < corner[1] and abs(line_function(corner[0]) - corner[1]) > 1e-15][0]
        highlighted_points = [point for point in corner_points if
                              (point[0] == SPB and point[1] >= line_function(SPB)) or abs(
                                  line_function(point[0]) - point[1]) < 1e-10]

        print("-----------------------------------")
        print(f"SPD: {SPD} kJ/mol")
        print(f"SPB: {SPB}")
        print(f"SPF: {SPF}")
        # xran = [0, 0.4]
        if zoomed:
            xran = [0, 2 * SPB]
            yran = [0, np.max(fe_diffs[distances < 2 * SPB])]
        else:
            d = np.max([i[1] for i in highlighted_points])
            xran = [0, min(4 * SPB, np.max(distances))]
            yran = [0, min(np.max(fe_diffs[distances < 4 * SPB]), 4 * d)]
        #  Original line
        # plt.plot([0, SPB], [0, initial_SPF * SPB], color="red")
        #  Shifted line
        plt.plot([0, SPB], [SPD, line_function(SPB)], color='springgreen')
        plt.plot([SPB, SPB], [line_function(SPB), np.max(fe_diffs)], color='springgreen')
        plt.scatter([i[0] for i in highlighted_points], [i[1] for i in highlighted_points], c="red", s=110)

        plt.xlim(xran)
        plt.ylim(yran)
        pl.set_yticklabels(np.round(pl.get_yticks(), 2), size=30)
        pl.set_xticklabels(np.round(pl.get_xticks(), 3), size=30)
        z_suffix = "2SPB" if zoomed else "4SPB"
        plt.savefig(f"./figures/jarvis_{prop}-vs-EMD_PDD100_{z_suffix}_angular_jump.png")
        plt.show()
        plt.close()


def plot_T2(path="/home/jon/go/src/opencrystal/data/T2_Predicted_Structures.cif"):
    reader = CifReader(path)
    amds, le = list(zip(*[(amd.AMD(c, k=100), float(c.name.split("_")[0])) for c in tqdm(reader, desc="Converting..")]))
    distances = amd.AMD_pdist(amds, low_memory=True)
    diffs = pdist(np.array(le)[:, None])

    cutoff = 0.1
    inds = np.where(distances < cutoff)[0]
    distances = distances[inds]
    diffs = diffs[inds]

    plt.figure(figsize=(30, 20))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    pl = seaborn.scatterplot(x=distances, y=diffs, color="blue")
    pl.set_xlabel("L-inf on AMD at k = 100 (Angstroms)", fontsize=46)
    pl.set_ylabel(f"Absolute Difference in Lattice Energy (kJ/mol)", fontsize=46)
    plt.xlim([0, cutoff])
    pl.set_yticklabels(np.round(pl.get_yticks(), 0), size=30)
    pl.set_xticklabels(np.round(pl.get_xticks(), 2), size=30)
    plt.savefig(f"./figures/lattice_energy_spr.png")