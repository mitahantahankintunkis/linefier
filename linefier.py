#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
import random
import sys


# Generates a new random point used for bezier curves
def newPoint(img, args):
    x  = random.randint(args.max_arc // 4, img.shape[1] - args.max_arc // 4 - 1)
    y  = random.randint(args.max_arc // 4, img.shape[0] - args.max_arc // 4 - 1)
    dx = random.randint(-args.max_arc, args.max_arc)
    dy = random.randint(-args.max_arc, args.max_arc)
    return [x, y, x - dx, y - dy, x + dx, y + dy]


# https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Cubic_B%C3%A9zier_curves
def cubicBezier(p0, p1, t):
    x = int((1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p0[4] + 3 * (1 - t) * t**2 * p1[2] + t**3 * p1[0])
    y = int((1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p0[5] + 3 * (1 - t) * t**2 * p1[3] + t**3 * p1[1])
    return x, y


# Breaks a bezier curve into a list of line coordinates
def segmentBezier(p0, p1, segments):
    ret = []

    x0, y0 = p0[0], p0[1]
    for i in range(0, segments + 1):
        t = 1 / segments * i
        x1, y1 = cubicBezier(p0, p1, t)
        dx, dy = x1 - x0, y1 - y0

        if dx*dx + dy*dy >= 4*4 or i == segments:
            ret.append((x0, y0, x1, y1))
            x0, y0 = x1, y1

    return ret


# Draws a bezier curve from p0 to p1
def drawBezier(img, p0, p1, args):
    ret      = np.copy(img)
    x0       = p0[0]
    y0       = p0[1]

    lines = segmentBezier(p0, p1, 20)

    for x0, y0, x1, y1 in lines:
        cv2.line(ret, (x0, y0), (x1, y1), int(args.line_color * 255), args.thickness, cv2.LINE_AA)

    return cv2.addWeighted(ret, args.opacity, img, 1 - args.opacity, 0, ret)


# Calculates the squared difference between two images
def fitness(target, img):
    delta = np.square(target.astype(np.int32) - img.astype(np.int32))
    return np.sum(delta) / (delta.shape[0] * delta.shape[1] * 255**2)


def generatePoints(target, args):
    points = []
    img    = np.full(target.shape, int(args.background_color * 255), np.uint8)

    # Creating the starting point
    point = newPoint(img, args)
    while target[point[1]][point[0]] == int(args.background_color * 255):
        point = newPoint(img, args)

    points.append(point)

    prev_fitness = 1
    n_failed = 0
    curves = 0

    while True:
        # Breaking if the user want's to limit the amount of curves
        if args.curves != -1 and curves >= args.curves: break

        # Generating multiple random images and sorting them based on their fitness
        new_imgs = []

        for _ in range(args.tries):
            # Generating points until one is in range
            while True:
                point = newPoint(img, args)
                dx = points[-1][0] - point[0]
                dy = points[-1][1] - point[1]
                if args.min_line_length**2 < dx**2 + dy**2 < args.max_line_length**2:
                    break

            new_imgs.append((drawBezier(img, points[-1], point, args), point))

        new_imgs.sort(key=lambda x: fitness(target, x[0]))

        # Only adding the new curve if it improves the generated image
        if prev_fitness > fitness(target, new_imgs[0][0]):
            img = new_imgs[0][0]
            prev_fitness = fitness(target, img)
            points.append(new_imgs[0][1])
            n_failed = 0
        else:
            # Stopping the program if there are ten failed attempts
            # at creating new curves
            n_failed += 1
            if n_failed >= 10:
                break

        # Saving every 50th image
        if curves % 50 == 0:
            cv2.imwrite('generated.png', img)

        if args.animate:
            if not os.path.exists('./animationFrames'):
                os.makedirs('./animationFrames')
            cv2.imwrite(f'./animationFrames/generated_{curves:0>5}.png', img)

        print(f' Curves: {curves}    Difference: {round(prev_fitness * 100, 1):>5}%      ', end='\r')
        sys.stdout.flush()

        curves += 1

    print()
    cv2.imwrite('generated.png', img)

    return points


# Exports normalized line coordinates into a file
#   Format:
#
#   <aspect_ratio (width / height):float>
#   <x:float> <y:float>
#   <x:float> <y:float>
#   <x:float> <y:float>
#   ....
#   <x:float> <y:float>

def export(target, points, args):
    if args.export:
        ret = [(points[0][0], points[0][1])]

        for p0, p1 in zip(points[:-1], points[1:]):
            lines = segmentBezier(p0, p1, 100)
            for line in lines:
                ret.append((line[2], line[3]))

        with open(args.export, 'w') as f:
            # Aspect ratio
            f.write(f'{target.shape[1] / target.shape[0]}\n')

            # x and y coordinates seperated by a space
            for p in ret:
                normalized_x = p[0] / target.shape[1]
                normalized_y = p[1] / target.shape[0]
                f.write(f'{normalized_x} {normalized_y}\n')



# Loads and preprocesses the target image
def loadTarget(args):
    target = cv2.imread(args.image)

    if target is None:
        print('Invalid image')
        sys.exit(1)

    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Resizing
    if target.shape[0] > target.shape[1]:
        width  = int(target.shape[1] / target.shape[0] * args.size)
        height = args.size
    else:
        width  = args.size
        height = int(target.shape[0] / target.shape[1] * args.size)

    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_AREA)

    return target


# Very basic input validation. Mainly for helping the user, not to improve security
def validateInput(args):
    errors = []
    if args.size            <= 0: errors.append('size must be greater than 0')
    if args.curves          < -1: errors.append('curves must be greater than 0 or -1')
    if args.tries           <= 0: errors.append('tries must be greater than 0')
    if args.max_arc         <= 0: errors.append('max_arc must be greater than 0')
    if args.min_line_length <= 0: errors.append('min_line_length must be greater than 0')
    if args.max_line_length <= 0: errors.append('max_line_length must be greater than 0')
    if args.thickness       <= 0: errors.append('thickness must be greater than 0')

    if not (0 <= args.opacity          <= 1): errors.append('opacity must be in range [0, 1]')
    if not (0 <= args.background_color <= 1): errors.append('background_color must be in range [0, 1]')
    if not (0 <= args.line_color       <= 1): errors.append('line_color must be in range [0, 1]')

    if errors:
        for error in errors:
            print(error)
        sys.exit(1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to an image. The higher the contrast the better')
    parser.add_argument('--export', help='Export line coordinates normalized to width')

    parser.add_argument('--animate', action='store_true', help='Saves generated images during the creation process')

    parser.add_argument('-s', '--size',       default=512,  type=int,   help='Maximum dimension of the generated image. Higher values slow the program down fast')
    parser.add_argument('-n', '--curves',     default=-1,   type=int,   help='Maximum amount of curves the program can draw. By default the program decides itself')
    parser.add_argument('-t', '--tries',      default=200,  type=int,   help='Number of random curves tried before adding one. Higher values produce better images')
    parser.add_argument('--max_arc',          default=64,   type=int,   help='How round the drawn curves are')
    parser.add_argument('--min_line_length',  default=64,   type=int,   help='The smallest distance a single curve can have')
    parser.add_argument('--max_line_length',  default=128,  type=int,   help='The larget distance a single curve can have')
    parser.add_argument('--thickness',        default=1,    type=int,   help='Thickness of the drawn line')
    parser.add_argument('--opacity',          default=0.25, type=float, help='Opacity of the drawn line')
    parser.add_argument('--background_color', default=1,    type=float, help='Color of the background ranging from 0 to 1')
    parser.add_argument('--line_color',       default=0,    type=float, help='Color of the line ranging from 0 to 1')
    args = parser.parse_args()

    validateInput(args)

    target = loadTarget(args)
    points = generatePoints(target, args)

    export(target, points, args)


if __name__ == '__main__': main()
