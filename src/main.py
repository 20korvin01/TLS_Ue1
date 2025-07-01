import os
import numpy as np
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
import json


def project_points_onto_plane(points_3d, plane_model):
    """
    Projiziert 3D-Punkte orthogonal auf eine Ebene und liefert 2D-Koordinaten im lokalen Koordinatensystem.
    Optional: Rückgabe von Basisvektoren und Ursprung zur Rücktransformation.

    Parameter:
        points_3d: (N, 3)
        plane_model: (4,)

    Rückgabe:
        points_2d: (N, 2)
        u: (3,) Basisvektor in X-Richtung
        v: (3,) Basisvektor in Y-Richtung
        origin: (3,) Ursprung des lokalen Koordinatensystems
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal_unit = normal / np.linalg.norm(normal)

    # Projektion der Punkte
    projected_points_3d = points_3d - ((points_3d @ normal + d)[:, np.newaxis]) * normal / np.dot(normal, normal)

    # Lokale Basisvektoren
    arbitrary = np.array([1, 0, 0]) if abs(normal_unit[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal_unit, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal_unit, u)

    # Ursprung des lokalen Systems
    origin = projected_points_3d.mean(axis=0)
    relative = projected_points_3d - origin

    # 2D Koordinaten
    u_coords = relative @ u
    v_coords = relative @ v
    points_2d = np.vstack([u_coords, v_coords]).T
    
    return  points_2d, u, v, origin


if __name__ == "__main__":
    ###TODO 0. SCHRITT: Daten einlesen und vorbereiten -------------------------------------------------------------------
    # * Verzeichnisse für die Daten
    data_SP1 = "SP1/"
    data_SP2 = "SP2/"

    # * Targetzentren, die später für die Auswertung der Ergebnisse verwendet werden
    target_centers_SP1 = {
        "T1": [
            [],
            [],
            [],
        ],
        "T2": [
            [],
            [],
            [],
        ],
        "T3": [  # T3.3 ist leider nicht brauchbar, da zu wenige Punkte gescannt wurden
            [],
            [],
        ],
        "T4": [
            [],
            [],
            [],
        ],
    }
    target_centers_SP2 = {
        "T1": [
            [],
            [],
            [],
        ],
        "T2": [
            [],
            [],
            [],
        ],
        "T3": [
            [],
            [],
            [],
        ],
        "T4": [
            [],
            [],
            [],
        ],
    }

    # for dir in [data_SP1, data_SP2]:
    #     dir_path = f"data/targets/{dir}"
    #     for filename in os.listdir(dir_path):

    for dir in tqdm([data_SP1, data_SP2], desc="Verarbeite Verzeichnisse"):
        dir_path = f"data/Targets/{dir}"
        for filename in tqdm(os.listdir(dir_path), desc=f"Dateien in {dir}"):

            file_path = os.path.join(dir_path, filename)

            ## Punktwolke + Zusatzdaten laden
            xyz = np.loadtxt(file_path, usecols=(0, 1, 2), skiprows=1)
            rgb = np.loadtxt(file_path, usecols=(3, 4, 5), skiprows=1)
            intensity = np.loadtxt(file_path, usecols=(6,), skiprows=1)

            ## Open3D Punktwolke erstellen
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            ###TODO 1. SCHRITT: Ebenenfitting ----------------------------------------------------------------------------
            ##? RANSAC und Bestimmung bestanpassenster Ebene mit Open3D
            # * RANSAC-Parameter
            distance_threshold = 0.003  # max. Abstand
            ransac_n = 3  # Anzahl Punkte für Ebenenfit
            num_iterations = 1000  # Anzahl der RANSAC Iterationen
            # * Ebene segmentieren
            plane_model, inlier_indices = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )
            # * Inlier aus der Punktwolke extrahieren
            inlier_xyz = xyz[inlier_indices]
            inlier_rgb = rgb[inlier_indices]
            inlier_intensity = intensity[inlier_indices]

            # * Outlier extrahieren
            outlier_indices = np.setdiff1d(np.arange(len(xyz)), inlier_indices)
            outlier_xyz = xyz[outlier_indices]
            outlier_rgb = rgb[outlier_indices]
            outlier_intensity = intensity[outlier_indices]

            ##? (Orthogonale) Projektion der Inlier-Punkte auf die Ebene -> Ergebnis: 2D-Punktwolke
            projected_inlier_uv, u, v, origin = project_points_onto_plane(inlier_xyz, plane_model)

            ###TODO 2. SCHRITT: Kantendetektion --------------------------------------------------------------------------
            """ Annahme: Punkte auf der Kante zwischen schwarzen und weißen Flächen haben 'mittlere Intensitäten' """
            # * Histogramm der Intensitäten erstellen
            hist, bin_edges = np.histogram(inlier_intensity, bins=200)

            # * Oberes und unteres Maximum des Histogramms finden
            upper_max_idx = np.argmax(hist[len(hist) // 2 :]) + len(hist) // 2
            lower_max_idx = np.argmax(hist[: len(hist) // 2])
            upper_max = (
                bin_edges[upper_max_idx] + (bin_edges[1] - bin_edges[0]) / 2
            )  # Mitte des Bins
            lower_max = (
                bin_edges[lower_max_idx] + (bin_edges[1] - bin_edges[0]) / 2
            )  # Mitte des Bins

            # * Schwellwerte für 'mittlere Intensitäten' definieren
            percentage = 20  # Prozentuale Abweichung von der Mitte der Intensitätswerte
            range_ = upper_max_idx - lower_max_idx
            center_idx = (upper_max_idx + lower_max_idx) // 2
            threshold_upper_idx = center_idx + int(range_ // 2 * percentage / 100)
            threshold_lower_idx = center_idx - int(range_ // 2 * percentage / 100)

            # * Mittlere Intensitäten extrahieren basierend auf den Schwellwerten
            middle_inlier_mask = (
                inlier_intensity >= bin_edges[threshold_lower_idx]
            ) & (inlier_intensity <= bin_edges[threshold_upper_idx])
            middle_inlier_uv = projected_inlier_uv[middle_inlier_mask]
            middle_inlier_rgb = inlier_rgb[middle_inlier_mask]
            middle_inlier_intensity = inlier_intensity[middle_inlier_mask]

            ###TODO 3. SCHRITT: Geradenschätzung -------------------------------------------------------------------------
            ##? RANSAC für die Schätzung der beiden Geraden in den 2D-Punkten
            # * Initialisierung der Listen für die Inlier und Outlier, sowie Parameter der Geraden
            line_outlier_uv = (
                middle_inlier_uv.copy()
            )  # Kopie der mittleren Inlier-Punkte, da zu Beginn alle Punkte als Outlier betrachtet werden
            line_outlier_rgb = middle_inlier_rgb.copy()
            line_outlier_intensity = middle_inlier_intensity.copy()
            # --
            line_inlier_uv = []
            line_inlier_rgb = []
            line_inlier_intensity = []
            # --
            line_params = []

            # * Schleife für die Schätzung der beiden Geraden
            for _ in range(2):
                # * RANSAC initialisieren für die Schätzung der Geraden in den 2D-Punkten
                line_ransac = RANSACRegressor(
                    LinearRegression(),
                    min_samples=2,
                    residual_threshold=0.003,
                    max_trials=1000,
                )

                # * Fit der Geraden auf die 2D-Punkte mit mittleren Intensitäten
                line_ransac.fit(
                    line_outlier_uv[:, 0].reshape(-1, 1), line_outlier_uv[:, 1]
                )
                line_inlier_mask = line_ransac.inlier_mask_
                line_model = [
                    line_ransac.estimator_.coef_[0],  # Steigung
                    line_ransac.estimator_.intercept_,  # Achsenabschnitt
                ]

                # * Speichern der Parameter der Geraden
                line_params.append(line_model)

                # * Extraktion der Inlier-Punkte für die Gerade
                line_inlier_uv_i = line_outlier_uv[line_inlier_mask]
                line_inlier_rgb_i = line_outlier_rgb[line_inlier_mask]
                line_inlier_intensity_i = line_outlier_intensity[line_inlier_mask]

                # * Extraktion der Outlier-Punkte für die Gerade -> Überschreiben der Outlier-Listen
                line_outlier_uv = line_outlier_uv[~line_inlier_mask]
                line_outlier_rgb = line_outlier_rgb[~line_inlier_mask]
                line_outlier_intensity = line_outlier_intensity[~line_inlier_mask]

                # * Speichern der Inlier-Punkte der Geraden
                line_inlier_uv.append(line_inlier_uv_i)
                line_inlier_rgb.append(line_inlier_rgb_i)
                line_inlier_intensity.append(line_inlier_intensity_i)

            # * Konvertieren der Listen in NumPy-Arrays
            line_inlier_uv = np.vstack(line_inlier_uv)
            line_inlier_rgb = np.vstack(line_inlier_rgb)
            line_inlier_intensity = np.hstack(line_inlier_intensity)

            ###TODO 4. SCHRITT: Koordinatenberechnung --------------------------------------------------------------------
            ## ? Schnittpunkt der beiden Geraden in den 2D-Punkten
            target_center_2d = np.linalg.solve(
                np.array([[line_params[0][0], -1], [line_params[1][0], -1]]),
                np.array([-line_params[0][1], -line_params[1][1]]),
            )
            
            ## ? Umwandlung der 2D-Koordinaten in 3D-Koordinaten
            target_center_3d = origin + target_center_2d[0] * u + target_center_2d[1] * v

            ###TODO PLOTTEN der Zwischenergebnisse -----------------------------------------------------------------------
            # * Visualisierung der originalen 3D-Punkte, Inlier in grün, Outlier in rot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                outlier_xyz[:, 0],
                outlier_xyz[:, 1],
                outlier_xyz[:, 2],
                c="r",
                s=1,
                label="Outlier",
            )
            ax.scatter(
                inlier_xyz[:, 0],
                inlier_xyz[:, 1],
                inlier_xyz[:, 2],
                c="g",
                s=1,
                label="Inlier",
            )
            ax.set_title(f"3D Points - {filename.strip('.txt')}")
            ax.set_xlabel("X-Koordinate")
            ax.set_ylabel("Y-Koordinate")
            ax.set_zlabel("Z-Koordinate")
            ax.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"./plots/targets/{dir}/{filename.strip('.txt')}_3d_points.png")

            # * Visualisierung der 2D-Punkte mit Intensitätswerte als Farbskala farbig
            plt.figure(figsize=(10, 10))
            plt.scatter(
                projected_inlier_uv[:, 0],
                projected_inlier_uv[:, 1],
                c=inlier_intensity,
                cmap="viridis",
                s=1,
            )
            plt.colorbar(label="Intensität")
            plt.title(f"2D Projection - {filename.strip('.txt')}")
            plt.xlabel("U-Koordinate")
            plt.ylabel("V-Koordinate")
            plt.axis("equal")
            plt.grid()
            plt.tight_layout()
            plt.savefig(
                f"./plots/projected/{dir}/{filename.strip('.txt')}_2d_projection.png"
            )

            # * Visualisierung der Intensitäten-Histogramms
            plt.figure(figsize=(10, 5))
            plt.bar(
                bin_edges[:-1],
                hist,
                width=np.diff(bin_edges),
                edgecolor="black",
                align="edge",
                alpha=0.7,
            )
            plt.axvline(
                x=upper_max, color="r", linestyle="-", label="Oberes/Unteres Maximum"
            )
            plt.axvline(x=lower_max, color="r", linestyle="-")
            plt.axvline(
                x=bin_edges[threshold_upper_idx],
                color="g",
                linestyle="--",
                label=f"Oberer/Unterer Schwellwert | ±{percentage}%",
            )
            plt.axvline(x=bin_edges[threshold_lower_idx], color="g", linestyle="--")
            plt.xlabel("Intensität")
            plt.ylabel("Häufigkeit")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(
                f"./plots/histograms/{dir}/{filename.strip('.txt')}_histogram.png"
            )
            plt.close("all")

            # * Visualisierung der mittleren-Intensitäten-Punkte
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(
                middle_inlier_uv[:, 0],
                middle_inlier_uv[:, 1],
                c=middle_inlier_intensity,
                cmap="viridis",
                s=5,
            )
            ax.set_xlabel("X-Koordinate")
            ax.set_ylabel("Y-Koordinate")
            ax.axis("equal")
            plt.grid()
            plt.tight_layout()
            plt.savefig(
                f"./plots/mean_intensities/{dir}/{filename.strip('.txt')}_mean_intensities.png"
            )

            # * Visualisierung der Geraden in den 2D-Punkten
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(
                line_outlier_uv[:, 0],
                line_outlier_uv[:, 1],
                c="r",
                s=5,
                label="Outlier",
            )
            ax.scatter(
                line_inlier_uv[:, 0],
                line_inlier_uv[:, 1],
                c="g",
                s=5,
                label="Inlier",
            )
            x_vals = np.linspace(
                np.concatenate([line_inlier_uv[:, 0], line_outlier_uv[:, 0]]).min(),
                np.concatenate([line_inlier_uv[:, 0], line_outlier_uv[:, 0]]).max(),
                num=100,
            )
            y_vals_1 = line_params[0][0] * x_vals + line_params[0][1]
            y_vals_2 = line_params[1][0] * x_vals + line_params[1][1]
            ax.plot(
                x_vals,
                y_vals_1,
                label="geschätzte Geraden",
                color="blue",
                linewidth=1,
                alpha=0.5,
            )
            ax.plot(x_vals, y_vals_2, color="blue", linewidth=1, alpha=0.5)
            ax.scatter(
                target_center_2d[0],
                target_center_2d[1],
                c="orange",
                s=2,
                label="Targetzentrum",
            )
            plt.axis("equal")
            ax.set_xlabel("U-Koordinate")
            ax.set_ylabel("V-Koordinate")
            ax.set_title(f"2D Points with Lines - {filename.strip('.txt')}")
            ax.legend()
            ax.grid()
            plt.tight_layout()
            plt.savefig(f"./plots/lines/{dir}/{filename.strip('.txt')}_lines.png")

            ###TODO SPEICHERN der Zwischenergebnisse ----------------------------------------------------
            # * Inlier speichern -> zunächst mit RGB und Intensität kombinieren
            inliers_combined = np.hstack(
                [inlier_xyz, inlier_rgb.reshape(-1, 3), inlier_intensity.reshape(-1, 1)]
            )
            np.savetxt(
                f"./data/inliers/{dir}/{filename.strip('.txt')}_inliers.txt",
                inliers_combined,
                fmt=["%.8f", "%.8f", "%.8f", "%.0f", "%.0f", "%.0f", "%.0f"],
                header="//X Y Z R G B Intensity",
                comments="",
            )
            # * Projektion speichern -> zunächst mit RGB und Intensität kombinieren
            points_2d_comnined = np.hstack(
                [
                    projected_inlier_uv,
                    inlier_rgb.reshape(-1, 3),
                    inlier_intensity.reshape(-1, 1),
                ]
            )
            np.savetxt(
                f"./data/projected/{dir}/{filename.strip('.txt')}_projected_2d.txt",
                points_2d_comnined,
                fmt=["%.8f", "%.8f", "%.0f", "%.0f", "%.0f", "%.0f"],
                header="//U V R G B Intensity",
                comments="",
            )
            # * Targetzentren in die Dictionaries eintragen
            target_name = filename.strip(".txt").split(".")[0]
            if dir == data_SP1:
                target_centers_SP1[target_name][0] = target_center_3d[0]
                target_centers_SP1[target_name][1] = target_center_3d[1]
                try:
                    target_centers_SP1[target_name][2] = target_center_3d[2]
                except:
                    pass
            else:
                target_centers_SP2[target_name][0] = target_center_3d[0]
                target_centers_SP2[target_name][1] = target_center_3d[1]
                target_centers_SP2[target_name][2] = target_center_3d[2]

        #     break
        # break

    # * Speichern der Targetzentren in eine Textdatei als json
    with open("./data/target_centers_SP1.json", "w") as f:
        json.dump(target_centers_SP1, f, indent=4)
    with open("./data/target_centers_SP2.json", "w") as f:
        json.dump(target_centers_SP2, f, indent=4)
    print("Fertig!")
