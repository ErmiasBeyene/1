"""
Helper functions for data trasformation from Gate root files
with 'Coincidences' tree to csv format.
"""
import numpy as np
import pandas
try:
  import uproot
except ModuleNotFoundError:
  import uproot3 as uproot

BRANCHES_COINCIDENCES = [
    'eventID1', 'globalPosX1', 'globalPosY1', 'globalPosZ1', 'time1', 'energy1', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1', 'comptonPhantom1', 'comptonCrystal1', 
    'eventID2', 'globalPosX2', 'globalPosY2', 'globalPosZ2','time2', 'energy2', 'sourcePosX2', 'sourcePosY2', 'sourcePosZ2','comptonPhantom2', 'comptonCrystal2', 
    'eventID3', 'globalPosX3', 'globalPosY3', 'globalPosZ3', 'time3', 'energy3', 'sourcePosX3', 'sourcePosY3', 'sourcePosZ3', 'comptonPhantom3', 'comptonCrystal3'
]
ALIASES_COINCIDENCES = [
    'eventID1', 'x1', 'y1', 'z1', 't1', 'e1', 'sX1', 'sY1', 'sZ1','comptonPhantom1', 'comptonCrystal1', 
    'eventID2', 'x2', 'y2', 'z2', 't2', 'e2', 'sX2', 'sY2', 'sZ2', 'comptonPhantom2', 'comptonCrystal2', 
    'eventID3', 'x3', 'y3', 'z3', 't3', 'e3', 'sX3', 'sY3', 'sZ3', 'comptonPhantom3', 'comptonCrystal3'
]
TREE_COINCIDENCES = 'Coincidences'

#Column names according to GOJA format
GOJA_FEATURES = [
    "x1",  # 1 gamma detected x position [cm]
    "y1",  # 1 gamma detected y position [cm]
    "z1",  # 1 gamma detected z position [cm]
    "t1",  # 1 gamma detection time [ps]
    "x2",  # 2 gamma detected x position [cm]
    "y2",  # 2 gamma detected y position [cm]
    "z2",  # 2 gamma detected z position [cm]
    "t2",  # 2 gamma detection time [ps]
    "vol1",  # 1 gamma volume ID
    "vol2",  # 2 gamma volume ID
    "e1",  # 1 gamma energy loss during detection [keV]
    "e2",  # 2 gamma energy loss during detection [keV]
    # Type of coincidence(1-true, 2-phantom-scattered, 3-detector-scattered, 4-accidental)
    "class",
    "sX1",  # 1 gamma emission x position [cm]
    "sY1",  # 1 gamma emission y position [cm]
    "sZ1",  # 1 gamma emission z position [cm]
    "sX2",  # 2 gamma emission x position [cm]
    "sY2",  # 2 gamma emission y position [cm]
    "sZ2"  # 2 gamma emission z position [cm]
]


def get_class_triple(row):
    try:
        if not (row['eventID1'] == row['eventID2'] == row['eventID3']):
            return 4
    except KeyError:
        return 4

    if (row.get('comptonPhantom1', 1) == 0 and row.get('comptonPhantom2', 1) == 0 and
        row.get('comptonPhantom3', 1) == 0 and
        row.get('comptonCrystal1', 0) == 1 and row.get('comptonCrystal2', 0) == 1 and
        row.get('comptonCrystal3', 0) == 1):
        return 1

    if (row.get('comptonPhantom1', 0) != 0 or row.get('comptonPhantom2', 0) != 0 or
        row.get('comptonPhantom3', 0) != 0):
        return 2

    if (row.get('comptonCrystal1', 0) > 1 or row.get('comptonCrystal2', 0) > 1 or
        row.get('comptonCrystal3', 0) > 1):
        return 3

    return -1


def add_features_triple(df):
    """Add features for triple coincidences. Strategy:
       - if x3/y3/z3 present, compute pairwise reconstructions (12,13,23)
       - choose the pair whose detector-hit vectors are most opposite (angle nearest 180°)
         and use that pair to fill rX/rY/rZ, rError, lorL, deg3D, deg2D (so downstream code can reuse same columns).
       - also compute eSum as sum of three energies, eDiff as max-min, etc.
    """
    # standard energy/time sums
    df['eSum'] = df.get('e1', 0) + df.get('e2', 0) + df.get('e3', 0)
    df['eDiff'] = (df[['e1', 'e2', 'e3']].max(axis=1) - df[['e1', 'e2', 'e3']].min(axis=1)) \
                   if {'e1','e2','e3'}.issubset(df.columns) else np.nan

    # pairwise dt
    if {'t1', 't2', 't3'}.issubset(df.columns):
        df['dt12'] = df['t1'] - df['t2']
        df['dt13'] = df['t1'] - df['t3']
        df['dt23'] = df['t2'] - df['t3']
    else:
        df['dt12'] = df['dt13'] = df['dt23'] = np.nan

    # if coordinates available compute pairwise reconstructions and angles
    if {'x1','y1','z1','x2','y2','z2','x3','y3','z3'}.issubset(df.columns):
        # pairwise deg3D
        df['deg12'] = get_angle_3D(df['x1'], df['y1'], df['z1'], df['x2'], df['y2'], df['z2'])
        df['deg13'] = get_angle_3D(df['x1'], df['y1'], df['z1'], df['x3'], df['y3'], df['z3'])
        df['deg23'] = get_angle_3D(df['x2'], df['y2'], df['z2'], df['x3'], df['y3'], df['z3'])

        # reconstruct vertices for each pair using existing function
        r12 = df.apply(lambda r: reconstruct_vertex_with_delta_time(
                          r.x1, r.y1, r.z1, r.x2, r.y2, r.z2, r.dt12) if not np.isnan(r.dt12) else (np.nan, np.nan, np.nan),
                      axis=1, result_type='expand')
        r12.columns = ['rX12', 'rY12', 'rZ12']

        r13 = df.apply(lambda r: reconstruct_vertex_with_delta_time(
                          r.x1, r.y1, r.z1, r.x3, r.y3, r.z3, r.dt13) if not np.isnan(r.dt13) else (np.nan, np.nan, np.nan),
                      axis=1, result_type='expand')
        r13.columns = ['rX13', 'rY13', 'rZ13']

        r23 = df.apply(lambda r: reconstruct_vertex_with_delta_time(
                          r.x2, r.y2, r.z2, r.x3, r.y3, r.z3, r.dt23) if not np.isnan(r.dt23) else (np.nan, np.nan, np.nan),
                      axis=1, result_type='expand')
        r23.columns = ['rX23', 'rY23', 'rZ23']

        # join back
        df = df.join(r12).join(r13).join(r23)

        # choose the pair with angle closest to 180 (i.e., largest degree)
        def pick_best_pair(row):
            degs = {'12': row['deg12'], '13': row['deg13'], '23': row['deg23']}
            best = max(degs, key=lambda k: degs[k] if not np.isnan(degs[k]) else -999)
            return best

        best_pairs = df.apply(pick_best_pair, axis=1)

        # set standard columns (rX,rY,rZ,lorL,deg3D,deg2D,rError) using chosen pair
        rX = []
        rY = []
        rZ = []
        lorL = []
        deg3D = []
        deg2D = []
        rError = []
        for idx, row in df.iterrows():
            bp = best_pairs.loc[idx]
            if bp == '12':
                xA, yA, zA = row['x1'], row['y1'], row['z1']
                xB, yB, zB = row['x2'], row['y2'], row['z2']
                rvec = (row['rX12'], row['rY12'], row['rZ12'])
            elif bp == '13':
                xA, yA, zA = row['x1'], row['y1'], row['z1']
                xB, yB, zB = row['x3'], row['y3'], row['z3']
                rvec = (row['rX13'], row['rY13'], row['rZ13'])
            else:  # '23'
                xA, yA, zA = row['x2'], row['y2'], row['z2']
                xB, yB, zB = row['x3'], row['y3'], row['z3']
                rvec = (row['rX23'], row['rY23'], row['rZ23'])

            # append features (some may be NaN)
            rX.append(rvec[0])
            rY.append(rvec[1])
            rZ.append(rvec[2])
            lorL.append(get_length(xA, yA, zA, xB, yB, zB))
            deg3D.append(get_angle_3D(xA, yA, zA, xB, yB, zB))
            deg2D.append(get_angle_2D(xA, yA, xB, yB))
            # rError: compare chosen reconstructed vertex against known source for the annihilation gamma(s)
            if {'sX1','sY1','sZ1'}.issubset(df.columns):
                # try to use sX of one of the pair (use sX1 if it belongs, else sX2/sX3)
                if bp == '12':
                    sX, sY, sZ = row['sX1'], row['sY1'], row['sZ1']
                elif bp == '13':
                    sX, sY, sZ = row['sX1'], row['sY1'], row['sZ1']
                else:
                    sX, sY, sZ = row['sX2'], row['sY2'], row['sZ2']
                rError.append(np.sqrt((sX - rvec[0])**2 + (sY - rvec[1])**2 + (sZ - rvec[2])**2))
            else:
                rError.append(np.nan)

        df['rX'] = rX
        df['rY'] = rY
        df['rZ'] = rZ
        df['lorL'] = lorL
        df['deg3D'] = deg3D
        df['deg2D'] = deg2D
        df['rError'] = rError

    else:
        df['rX'] = df['rY'] = df['rZ'] = np.nan
        df['lorL'] = df['deg3D'] = df['deg2D'] = df['rError'] = np.nan

    return df
