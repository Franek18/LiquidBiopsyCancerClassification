import os
import csv

def annotateCancer(dir, cancer_dir, noncancer_dir):
    '''
        This function creates a csv file in directory annotations which contains
        values grouped into two columns: Path to the sample and Class of this sample.
        @param dir: path to the directory with samples.
        @param cancer_dir: name of a subfolder in @param dir which contains
        cancer samples as a matrices - .txt files not images.
        @param noncancer_dir: name of a subfolder in @param dir which contains
        non-cancer samples as a matrices - .txt files not images.
    '''
    with open('annotations/Cancer_annotations_mts.csv', mode='w') as cancer_file:
        cancer_writer = csv.writer(cancer_file, delimiter=',')

        cancer_writer.writerow(["Path", "Class"])
        # Annotate positive samples
        for img_name in os.listdir(dir + "/" + cancer_dir):
            cancer_writer.writerow([cancer_dir + "/" + img_name, 1])

        # Annotate negative (HC + CTRL) samples
        for img_name in os.listdir(dir + "/" + noncancer_dir):
            cancer_writer.writerow([noncancer_dir + "/" + img_name, 0])
