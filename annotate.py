import os
import csv

def annotateMultiClass(sampleInfofile="TEPS_multiclass_14_10_22/TEPS_Data_preparation_data_sample_info/SampleInfo_short_multiclass_2022-10-14.tsv", matrices_dir="TEPS_multiclass_14_10_22/matricesMulticlassNew"):
    '''
        This function creates a csv file in directory annotations which contains
        values grouped into two columns: Path to the sample and Class of this sample.
        @param dir: path to the directory with samples.
        @param cancer_dir: name of a subfolder in @param dir which contains
        cancer samples as a matrices - .txt files not images.
        @param noncancer_dir: name of a subfolder in @param dir which contains
        non-cancer samples as a matrices - .txt files not images.
    '''
    samples = {
        "Asymptomatic controls": [],
        "Gynecological": [],
        "Cardiovascular": [],
        "NSCLC": [],
        "Glioma and glioblastoma": [],
        "Gastrointestinal": [],
        "Neurological": []
    }
    # samples = {}
    with open(sampleInfofile, mode='r') as info_file:
        info_reader = csv.DictReader(info_file, delimiter='\t')
        line_count = 0
        for row in info_reader:
            if line_count == 0:
                line_count += 1
                continue
            filename = row['Sample.ID']
            sample_class = row['MultiGroup']
            split = row['TrainTest']
            # class_dir = row['BinaryGroup']
            # if sample_class == "asymptomaticControls" or sample_class == "multipleSclerose" or sample_class == "benignGyn":
            #     sample_class = "nonCancer"
            samples[sample_class].append([os.path.join(split, filename + ".tsv"), split])

    with open('annotations/MultiClass_annotations_mts.csv', mode='w') as cancer_file:
        cancer_writer = csv.writer(cancer_file, delimiter=',')

        cancer_writer.writerow(["Path", "Class", "Split"])
        # Annotate positive samples
        i = 0
        for sample_class in samples.keys():
            for sample_name, sample_split in samples[sample_class]:
                cancer_writer.writerow([os.path.join(matrices_dir, sample_name), i, sample_split])

            i += 1

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
