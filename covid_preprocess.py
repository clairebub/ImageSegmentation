import os
import med2image
from tqdm import tqdm


def main(root_dir):
    img_dir = "inputs/covid/images"
    infection_mask_dir = "inputs/covid/masks/0"  # num of labels  - 1 (1-1) infection
    lung_mask_dir = "inputs/covid/masks/1"  # (2-1) left lung, right lung
    lung_infection_mask_dir = 'inputs/covid/masks/2'  # (3-1) left lung, right lung, infection

    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if name != 'ReadMe.txt':
                path = os.path.join(root, name)
                if 'Mask' not in path:
                    cmd = "med2image -i {} -d {} -o {}.png -s -1".format(path, img_dir, name.split('.')[0])
                elif 'Lung_and_Infection_Mask' in path:
                    print ('right path')
                    cmd = "med2image -i {} -d {} -o {}.png -s -1".format(path, lung_infection_mask_dir, name.split('.')[0])
                elif 'Lung_Mask' in path:
                    cmd = "med2image -i {} -d {} -o {}.png -s -1".format(path, lung_mask_dir, name.split('.')[0])
                else:
                    cmd = "med2image -i {} -d {} -o {}.png -s -1".format(path, infection_mask_dir, name.split('.')[0])

                print('Executing cmd', cmd)
                os.system(cmd)

    return


if __name__ == '__main__':
    main('./inputs/covid_segmap_dataset')
