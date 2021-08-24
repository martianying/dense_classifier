import os

ROOT = r'C:/Users/liewei/Desktop/test/'

LABEL_PATH = os.path.join(ROOT, 'test.txt')

IMG_PATH = [os.path.join(ROOT, os.path.join(side, 'save')) for side in os.listdir(ROOT)]

_, cls, _ = next(os.walk(IMG_PATH[0]))
mark = '\\'

subs = [os.path.join(IMG_PATH[0], cl) for cl in cls]


def run():
    with open(LABEL_PATH, 'w+') as writer:
        for sub in subs:
            dirs, _, files = next(os.walk(sub))
            files.sort(key=lambda x: int(x.split('.')[0]))
            for file in files:
                path_a = os.path.join(dirs, file)
                path_a.split(mark)

                path_b = path_a.split(mark)[:5]
                path_b.append('B')
                path_b.extend(path_a.split(mark)[6:])
                path_b = mark.join(path_b)

                writer.write(path_a + '\t')
                writer.write(path_b + '\n')

