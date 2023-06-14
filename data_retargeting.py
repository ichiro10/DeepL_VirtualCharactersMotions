import numpy as np
import os 
import data_utils
import copy
from tqdm import tqdm



def main():
    # Define the folder path where the style data is stored
    data_folder = 'Style_data'

    # Define the styles and motion labels
    styles = ['ES', 'Neurotic', 'Normal']
    motions = ['waving', 'stopping', 'showingphone', 'showing', 'hiding']  # Add more motion labels as needed

    # Initialize empty lists to store motion data and labels
    motion_data = []
    motion_labels = []

    # Iterate over the styles and motion labels to extract the data
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
                    action_sequence= data_utils.readCSVasFloat(file_path)

                    # Append the motion data and labels
                    if len(motion_data) == 0:
                        motion_data = copy.deepcopy(action_sequence)
                    else:
                        motion_data = np.append(motion_data, action_sequence,axis=0)
                        motion_labels.append([motion_idx, style_idx])

    # Convert the motion data and labels to numpy arrays
    clips = np.array(motion_data)
    classes = np.array(motion_labels)

    # Save the data and labels into an npz file
    np.savez('data_styletransfer.npz', clips=clips, classes=classes)


if __name__ == '__main__':
  main()
