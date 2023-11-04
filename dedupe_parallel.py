from imagededup.utils import plot_duplicates
import multiprocessing
import os
from imagededup.methods import PHash

from PIL import Image
import logging
import argparse


def imgRes(imf):
    im = Image.open(imf)
    width, height = im.size
    im.close()
    return width, height


def filesize(fname):
    return os.path.getsize(fname)


def calchash(image_file1, encodings, phasher):
    # phasher = PHash()
    phash_string = phasher.encode_image(image_file=image_file1)
    # logging.debug(phash_string)
    encodings[image_file1] = phash_string


def draw_images(sdir, dupl, n1):
    plot_duplicates(image_dir=sdir,
                    duplicate_map=dupl,
                    filename=n1)


def main(cpuN, srcdir):

    print('Searching duplicated images in: ', srcdir)

    # logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.ERROR)

#    cpuN = int(multiprocessing.cpu_count() * 1.3)
    manager = multiprocessing.Manager()
    encodings = manager.dict()


#    srcdir = '/home/anton/dedupe.test/Lena.Photo_and_video/'
    # srcdir='/home/anton/dedupe.test/1'
    # srcdir='/home/anton/DVD1'

    # encodings = {}

    phasher = PHash()
    jobs = []

    for (root, dirs, files) in os.walk(srcdir, topdown=True):
        # logging.debug(root)
        j = 0
        for i in files[j::cpuN]:
            for F in files[j:j + cpuN]:
                if F.split('.')[-1].lower() in ['jpg', 'jpeg',
                                                'gif', 'bmp', 'tif', 'tiff']:
                    image_file2 = root + '/' + F
                    # logging.debug(image_file2)
                    pr = multiprocessing.Process(
                        target=calchash, args=(image_file2, encodings, phasher))
                    jobs.append(pr)
                    pr.start()
            for proc in jobs:
                proc.join()
            j = j + cpuN

    # logging.debug(encodings)

    # quit()

    # Generate encodings for all images in an image directory
    # encodings = phasher.encode_images(image_dir=srcdir)
    # Encoding is a dictionary

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    # duplicates is a dictionary

    images1 = []
    images2 = []

    cntdup = 0
    for dup in duplicates:
        A = {}
        a = {}
        if duplicates[dup] and dup not in images1:
            images1.append(dup)
            A[dup] = filesize(dup)
            for d in duplicates[dup]:
                if d not in images1:
                    images1.append(d)
                A[d] = filesize(d)

            a = sorted(A.items(), key=lambda x: x[1], reverse=True)
            # print(a)
            cntdup += 1
            images2.append(a[0][0])
    print('Similar images found: ', cntdup)

    # plot duplicates obtained for a given file using the duplicates dictionary

    # quit()

    for n in images2:
        pn = multiprocessing.Process(
            target=draw_images, args=(srcdir, duplicates, n))
        pn.start()
        files = {}
        files[1] = n
        print('\n\n')
        print(1, ' ', n, 'Size: ', filesize(n), imgRes(n))
        j = 2
        for f in duplicates[n]:
            print(j, ' ', f, 'Size: ', filesize(f), imgRes(f))
            files[j] = f
            j = j + 1

        text = input(
            'Which photos to delete (comma separated numbers) 0 or Nil - None: ')
        if text == '0' or text == '':
            print('Will not be deleted')
            pn.terminate()
        else:
            for i in text.split(','):
                if int(i) in files:
                    print('Removing file :', int(i), files[int(i)])
            # pn.join()
            pn.terminate()


if __name__ == '__main__':

    cpu = int(multiprocessing.cpu_count() * 1.3)
    parser = argparse.ArgumentParser(description="An argparse example")

    parser.add_argument(
        '-d',
        '--dir',
        help='Directory where to do dedupe',
        required=True,
        type=str)
    parser.add_argument(
        '-p',
        '--threads',
        help='Number of parallel threads',
        default=cpu,
        type=int)
    # parser.add_argument('threads', help='Number of parallel threads',)

    args = parser.parse_args()

    main(args.threads, args.dir)
