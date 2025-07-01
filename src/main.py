import os
import numpy as np
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def project_points_onto_plane(points_3d, plane_model):
    """
    Projiziert 3D-Punkte orthogonal auf eine Ebene und liefert 2D-Koordinaten auf dieser Ebene.

    Parameter:
        points_3d: (N, 3)
            3D-Punkte im Raum
        plane_model: (4,)
            Ebenenparameter [a, b, c, d] der Ebene: ax + by + cz + d = 0

    Rückgabe:
        points_2d: (N, 2) ndarray
            2D-Koordinaten der Punkte im lokalen Koordinatensystem der Ebene
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal_unit = normal / np.linalg.norm(normal)

    # 1. Projektion jedes Inlier-Punktes orthogonal auf die Ebene
    projected_points_3d = np.zeros_like(points_3d)  # (N, 3)
    for i, point in enumerate(points_3d):
        projected_point_3d = point - (np.dot(point, normal) + d) / np.linalg.norm(normal)**2 * normal
        projected_points_3d[i] = projected_point_3d

    # 2. Lokales Koordinatensystem (zwei Basisvektoren u, v)
    arbitrary = np.array([1, 0, 0])      # Beliebiger, nicht-paralleler Vektor zur Normalen
    u = np.cross(normal_unit, arbitrary) # Kreuzprodukt, um einen Vektor zu erhalten, der nicht parallel zur Normalen ist
    u /= np.linalg.norm(u)               # Normierung des Vektors u
    v = np.cross(normal_unit, u)         # Kreuzprodukt, um den zweiten Basisvektor v zu erhalten

    # 3. Ursprung des lokalen Systems: Mittelpunkt/Schwerpunkt der Ebene
    origin = projected_points_3d.mean(axis=0)
    relative = projected_points_3d - origin

    # 4. 2D-Koordinaten berechnen
    u_coords = relative @ u
    v_coords = relative @ v
    points_2d = np.vstack([u_coords, v_coords]).T

    return points_2d


if __name__ == "__main__":
    ###TODO 0. SCHRITT: Daten einlesen und vorbereiten -------------------------------------------------------------------
    data_SP1 = "SP1/"
    data_SP2 = "SP2/"

    # for dir in tqdm([data_SP1, data_SP2], desc="Verarbeite Verzeichnisse"):
    #     dir_path = f"data/Targets/{dir}"
    #     for filename in tqdm(os.listdir(dir_path), desc=f"Dateien in {dir}"):
    for dir in [data_SP1, data_SP2]:
        dir_path = f"data/Targets/{dir}"
        for filename in os.listdir(dir_path):
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
            #* RANSAC-Parameter
            distance_threshold = 0.001      # max. Abstand
            ransac_n = 3                    # Anzahl Punkte für Ebenenfit
            num_iterations = 1000           # Anzahl der RANSAC Iterationen
            #* Ebene segmentieren
            plane_model, inlier_indices = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            #* Inlier aus der Punktwolke extrahieren
            inlier_xyz = xyz[inlier_indices]
            inlier_rgb = rgb[inlier_indices]
            inlier_intensity = intensity[inlier_indices]
            
            ##? (Orthogonale) Projektion der Inlier-Punkte auf die Ebene -> Ergebnis: 2D-Punktwolke
            projected_points = project_points_onto_plane(inlier_xyz, plane_model)
            
            
            ###TODO 2. SCHRITT: Kantendetektion --------------------------------------------------------------------------
            """ Annahme: Punkte auf der Kante zwischen schwarzen und weißen Flächen haben 'mittlere Intensitäten' """
            #* Histogramm der Intensitäten erstellen
            hist, bin_edges = np.histogram(inlier_intensity, bins=200)
            
            #* Oberes und unteres Maximum des Histogramms finden
            upper_max_idx = np.argmax(hist[len(hist)//2:]) + len(hist)//2
            lower_max_idx = np.argmax(hist[:len(hist)//2])
            upper_max = bin_edges[upper_max_idx] + (bin_edges[1] - bin_edges[0]) / 2  # Mitte des Bins
            lower_max = bin_edges[lower_max_idx] + (bin_edges[1] - bin_edges[0]) / 2  # Mitte des Bins
                        
            
            #* Schwellwerte für 'mittlere Intensitäten' definieren
            percentage = 20  # Prozentuale Abweichung von der Mitte der Intensitätswerte
            range = upper_max_idx - lower_max_idx
            center_idx = (upper_max_idx + lower_max_idx) // 2
            threshold_upper_idx = center_idx + int(range//2 * percentage / 100)
            threshold_lower_idx = center_idx - int(range//2 * percentage / 100)
            
            #* Plotten des Histogramms
            plt.figure(figsize=(10, 5))
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7)
            plt.axvline(x=upper_max, color='r', linestyle='-', label='Oberes Maximum')
            plt.axvline(x=lower_max, color='r', linestyle='-', label='Unteres Maximum')
            plt.axvline(x=bin_edges[threshold_upper_idx], color='g', linestyle='--', label='Oberer Schwellwert')
            plt.axvline(x=bin_edges[threshold_lower_idx], color='g', linestyle='--', label='Unterer Schwellwert')
            plt.xlabel('Intensität')
            plt.ylabel('Häufigkeit')
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"./plots/Histograms/{dir}/{filename.strip('.txt')}_histogram.png")
            plt.close('all')
            
            
            
            ###TODO LETZTER SCHRITT: Speichern der Zwischenergebnisse ----------------------------------------------------
            #* Inlier speichern -> zunächst mit RGB und Intensität kombinieren
            inliers_combined = np.hstack([
                inlier_xyz,
                inlier_rgb.reshape(-1, 3),
                inlier_intensity.reshape(-1, 1)
            ])
            np.savetxt(
                f"./data/Inliers/{dir}/{filename.strip('.txt')}_inliers.txt",
                inliers_combined,
                fmt=['%.8f', '%.8f', '%.8f', '%.0f', '%.0f', '%.0f', '%.0f'],
                header="//X Y Z R G B Intensity",
                comments="",
            )
            #* Projektion speichern -> zunächst mit RGB und Intensität kombinieren
            points_2d_comnined = np.hstack([
                projected_points,
                inlier_rgb.reshape(-1, 3),
                inlier_intensity.reshape(-1, 1)
            ])
            np.savetxt(
                f"./data/Projected/{dir}/{filename.strip('.txt')}_projected_2d.txt",
                points_2d_comnined,
                fmt=['%.8f', '%.8f', '%.0f', '%.0f', '%.0f', '%.0f'],
                header="//U V R G B Intensity",
                comments=""
            )
            
            
            
        #     break
        # break
