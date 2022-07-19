import json
import pickle
from tqdm import tqdm


# This function finds which samples from our multiview dataset overlap
#   with the mp3d jsons from sparseplanes and generates an output file
#   that is required for the evaluation process
# mp3d_jsons: list of paths to train/val/test jsons
# multi_view_json: path to multiview dataset json
# output_file: name of output file
# num_images: number of images in multiview dataset
def match_jsons(mp3d_jsons, multi_view_json, output_file, num_images):

    mp3d_files = []

    for idx in range(len(mp3d_jsons)):
        with open(mp3d_jsons[idx], 'r') as f:
            mp3d_files.append(json.load(f)['data'])

    with open(multi_view_json, 'r') as f:
        multi_view_data = json.load(f)['data']

    num_not_found = 0
    tot = 0
    samples_affected = 0

    corr_set = {}
    for i in range(len(multi_view_data)):
        for j in range(num_images):
            filename = multi_view_data[i][str(j)]['file_name']
            if filename not in corr_set.keys():
                tot += 1
                found = False
                for file_num in range(len(mp3d_files)):
                    for k in range(len(mp3d_files[file_num])):
                        if filename == mp3d_files[file_num][k]['0']['file_name']:
                            corr_set[filename] = (file_num, k, '0')
                            found = True
                            break
                        if filename == mp3d_files[file_num][k]['1']['file_name']:
                            corr_set[filename] = (file_num, k, '1')
                            found = True
                            break
                if not found:
                    corr_set[filename] = None
                    num_not_found += 1

        sample_valid = True
        for j in range(num_images):
            if corr_set[multi_view_data[i][str(j)]['file_name']] == None:
                sample_valid = False
                break
        if not sample_valid:
            samples_affected += 1


    print("Num not found %d" % (num_not_found))
    print("Total %d" % (tot))
    print("Samples affected %d" % (samples_affected))

    with open(output_file, 'wb') as f:
        pickle.dump(corr_set, f)