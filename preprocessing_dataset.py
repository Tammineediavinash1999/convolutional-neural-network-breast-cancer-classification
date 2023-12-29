import os
from PIL import Image


# --------------- Preprocessing Database -----------------------------
def extract_label():
    complete_text = open("C:\\Users\\Avinash\\Desktop\\bccn\\all-mias_preprocessed_128_info.txt")
    lines = [line.rstrip('\n') for line in complete_text]

    details = {}
    current_path = os.getcwd()
    k = 0
    for line in lines:
        path = os.path.join(current_path, 'png')

        value = line.split(' ')
        path = path + '\\' + value[0] + ".png"
        im = Image.open(path)
        im1 = im
        out_file = ""
        print(im.size)
        k = k + 1
        if len(value) <= 4:
            details[value[0]] = 0
            im1 = im
            out_file = str(k)
            im1.save("C:\\Users\\Avinash\\Desktop\\bccn\\images\\" + out_file + ".png")
        else:
            if len(value) == 7:
                r = int(value[6])
                y2 = 1024 - int(value[5])
                y1 = int(value[5])
                x2 = 1024 - int(value[4])
                x1 = int(value[4])
                print(value)
                im1 = im.crop((int(value[4]) - min(64,x1),
                               int(value[5]) - min(64,y1),
                               int(value[4]) + min(64,x2),
                               int(value[5]) + min(64,y2)))
                out_file = str(k) + " crop " + value[3]
                im1.save("C:\\Users\\Avinash\\Desktop\\bccn\\preprocess1\\" + out_file + ".png")
                im1.save("C:\\Users\\Avinash\\Desktop\\bccn\\images\\" + out_file + ".png")
            else:
                details[value[0]] = 2
                im1 = im
                out_file = str(k)
                im1.save("C:\\Users\\Avinash\\Desktop\\bccn\\images\\" + out_file + ".png")


    return details


def rotate_images(angle,dir):
    current_path = os.getcwd()
    path = os.path.join(current_path, dir)
    for image in os.listdir(path):
        path = os.path.join(current_path, dir)
        path = path + '\\' + image
        print(path)
        img = Image.open(path)
        img.rotate(angle).save("C:\\Users\\Avinash\\Desktop\\bccn\\images\\" + str(angle) + " " + image)
        img.close()


def mirror():
    current_path = os.getcwd()
    path = os.path.join(current_path, 'preprocess1')
    for image in os.listdir(path):
        path = os.path.join(current_path, 'preprocess1')
        path = path + '\\' + image
        img = Image.open(path)
        img = img.transpose(Image.FLIP_LEFT_RIGHT).save("C:\\Users\\Avinash\\Desktop\\bccn\\preprocess3\\" + " rotate " + image)
        img = Image.open(path)
        img = img.transpose(Image.FLIP_LEFT_RIGHT).save("C:\\Users\\Avinash\\Desktop\\bccn\\images\\" + " rotate " + image)


labels = extract_label()
rotate_images(90,'preprocess1')
rotate_images(180,'preprocess1')
rotate_images(270,'preprocess1')
mirror()
rotate_images(90,'preprocess3')
rotate_images(180,'preprocess3')
rotate_images(270,'preprocess3')


