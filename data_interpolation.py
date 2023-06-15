import numpy as np
import data_utils
import os
from tqdm import tqdm
from scipy.interpolate import CubicSpline

def interpolate_sequence(sequence, desired_frames):
    frames = sequence.shape[0]
    t = np.linspace(0, 1, frames)  # Paramètre temporel normalisé entre 0 et 1
    t_interpolated = np.linspace(0, 1, desired_frames)  # Paramètre temporel interpolé
    joints = sequence.shape[1]  # Nombre de joints dans la séquence

    interpolated_sequence = np.zeros((desired_frames, joints))

    for joint in range(72):
        spline = CubicSpline(t, sequence[:, joint])
        interpolated_sequence[:, joint] = spline(t_interpolated)

    return interpolated_sequence



def main():

    desired_frames = 1000
    # Define the folder path where the style data is stored
    data_folder = 'Style_data'

    # Define the styles and motion labels
    styles = ['ES', 'Neurotic', 'Normal']
    motions = ['waving', 'stopping', 'showingphone', 'showing', 'hiding']  # Add more motion labels as needed

    # Initialize empty lists to store motion data and labels
    motion_data = []
    motion_labels = []

    # Appliquez l'interpolation cubique par morceaux à chaque séquence du dataset
    interpolated_dataset = []
    for style_idx, style in enumerate(styles):
            print('Processing : ',style)
            for motion_idx, motion in enumerate(motions):
                print('Processing : ',motion)
                # Define the path to the folder containing the motion files
                motion_folder = os.path.join(data_folder, style, motion)
                
                # Iterate over the files in the motion folder
                for file_name in tqdm(os.listdir(motion_folder)):
                    # Read the file content
                    file_path = os.path.join(motion_folder, file_name)
                    
                    #file_data = np.loadtxt(file_path)
                    sequence= data_utils.readCSVasFloat(file_path)
                    interpolated_sequence = interpolate_sequence(sequence, desired_frames)
                    interpolated_dataset.append(interpolated_sequence)

                    
if __name__ == '__main__':
  main()
