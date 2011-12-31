#!/usr/bin/env python2

import time
from math import radians, degrees, cos, sin, tan
from copy import deepcopy

from OpenGL.GL import *
from OpenGL.GL.shaders import *

import pygame
from pygame.locals import *
import numpy    
import numpy.linalg

width = 640
height = 480

vertices = numpy.array([-0.5, 0.5, 0.5,
                       -0.5, -0.5, 0.5,
                        0.5, -0.5, 0.5,
                        0.5,  0.5, 0.5,
                        -0.5, 0.5, -0.5,
                       -0.5, -0.5, -0.5,
                        0.5, -0.5, -0.5,
                        0.5,  0.5, -0.5], numpy.float32)
indices = numpy.array([0, 1, 2,
                       0, 2, 3,
                       0, 3, 7,
                       0, 7, 4,
                       0, 4, 5,
                       0, 5, 1,
                       3, 2, 6,
                       3, 6, 7,
                       2, 1, 6,
                       1, 6, 5,
                       5, 6, 4,
                       4, 7, 6], numpy.ushort)
colors = numpy.array([0.0, 1.0, 1.0, 1.0,
                      0.0, 0.0, 1.0, 1.0,
                      1.0, 0.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0,
                      0.0, 1.0, 0.0, 1.0,
                      0.0, 0.0, 0.0, 1.0,
                      1.0, 0.0, 0.0, 1.0,
                      1.0, 1.0, 0.0, 1.0], numpy.float32)

modelview = numpy.identity(4, numpy.float32)

def getFileContents(filename):
    return open(filename, 'r').read()

def init():
    vertexShader = compileShader(getFileContents("rotate.vert"), GL_VERTEX_SHADER)
    fragmentShader = compileShader(getFileContents("rotate.frag"), GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertexShader)
    glAttachShader(program, fragmentShader)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, vertices)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, colors)

    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    glBindAttribLocation(program, 0, "a_Position")
    glBindAttribLocation(program, 1, "a_Color")

    glLinkProgram(program)

    global mvpLoc
    mvpLoc = glGetUniformLocation(program, "u_MVP")

    glClearColor(0.0, 0.0, 0.0, 1.0)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    return program

def drawWithoutVBOs(state):
    # Perspective
    fov = 90.0
    aspect_ratio = width / height
    znear = 0.1
    zfar = 200
    h = 1.0 / tan(radians(fov / 2))
    neg_depth = znear - zfar

    pers = numpy.identity(4, numpy.float32)
    pers[0][0] = h / aspect_ratio
    pers[1][1] = h
    pers[2][2] = (zfar + znear) / neg_depth
    pers[2][3] = -1.0
    pers[3][2] = 2.0 * zfar * znear / neg_depth
    pers[3][3] = 0.0

    # Translation
    trans = numpy.identity(4, numpy.float32)
    # add cube translation here

    # Cube rotation
    xrot = 0
    yrot = 0
    zrot = 0
    rotat = rotation_matrix(xrot, yrot, zrot)

    # Camera translation
    trans_cam = numpy.identity(4, numpy.float32)
    trans_cam[3][0] = -state[controls.pos][0]
    trans_cam[3][1] = -state[controls.pos][1]
    trans_cam[3][2] = -state[controls.pos][2]

    # Camera rotation
    target_vec = state[controls.target]
    up_vec = state[controls.up]
    rotate_cam = get_camera_rotation_matrix(target_vec, up_vec)

    modelview = rotat.dot(trans.dot(trans_cam.dot(rotate_cam.dot(pers))))
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, modelview)

    glDrawElementsus(GL_TRIANGLES, indices)

def rotation_matrix(xrot, yrot, zrot):
    # TODO: optimize this.
    rotat = numpy.identity(4, numpy.float32)
    rotat[0][0] = cos(yrot) * cos(zrot)
    rotat[1][0] = -cos(xrot) * sin(zrot) + sin(xrot) * sin(yrot) * cos(zrot)
    rotat[2][0] = sin(xrot) * sin(zrot) + cos(xrot) * sin(yrot) * cos(zrot)
    rotat[0][1] = cos(yrot) * sin(zrot)
    rotat[1][1] = cos(xrot) * cos(zrot) + sin(xrot) * sin(yrot) * sin(zrot)
    rotat[2][1] = -sin(xrot) * cos(zrot) + cos(xrot) * sin(yrot) * sin(zrot)
    rotat[0][2] = -sin(yrot)
    rotat[1][2] = sin(xrot) * cos(yrot)
    rotat[2][2] = cos(xrot) * cos(yrot)
    return rotat

def normalize(vec):
    vec = normalized(vec)

def normalized(vec):
    return vec / numpy.linalg.norm(vec)

def get_camera_rotation_matrix(target_vec, up_vec):
    n = normalized(target_vec)
    u = normalized(up_vec)
    u = numpy.cross(u, target_vec)
    v = numpy.cross(n, u)
    m = numpy.identity(4, numpy.float32)
    m[0][0] = u[0]
    m[0][1] = v[0]
    m[0][2] = n[0]
    m[1][0] = u[1]
    m[1][1] = v[1]
    m[1][2] = n[1]
    m[2][0] = u[2]
    m[2][1] = v[2]
    m[2][2] = n[2]
    return m

class controls:
    pos = 1
    target = 2
    up = 3
    hrot = 4
    vrot = 5

rot_step = radians(1.0)
pos_step = 0.01

mp = dict()
mp[K_LEFT] = lambda tgt, upv: pos_step * normalized(numpy.cross(tgt, upv))
mp[K_RIGHT] = lambda tgt, upv: -mp[K_LEFT](tgt, upv)
mp[K_PAGEUP] = lambda tgt, upv: pos_step * normalized(upv)
mp[K_PAGEDOWN] = lambda tgt, upv: -mp[K_PAGEUP](tgt, upv)
mp[K_UP] = lambda tgt, upv: pos_step * -normalized(tgt) # TODO: why negated tgt?
mp[K_DOWN] = lambda tgt, upv: -mp[K_UP](tgt, upv)

def main():
    pygame.init()
    pygame.display.set_mode((width, height), HWSURFACE | OPENGL | DOUBLEBUF)

    program = init()

    done = False
    mouse_look = False
    state = dict()
    state[controls.pos] = numpy.array([0, 0, 1.55], numpy.float32)
    state[controls.target] = deepcopy(world_front)
    state[controls.up] = deepcopy(world_up)
    state[controls.hrot] = radians(90)
    state[controls.vrot] = 0
    handle_mouse_move(state, 0, 0)

    pos_delta = dict()

    clock = pygame.time.Clock()
    frames = 0
    start_time = pygame.time.get_ticks()

    while not done:
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)
        drawWithoutVBOs(state)
        pygame.display.flip()

        for delta in pos_delta.values():
            state[controls.pos] += delta

        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key in mp:
                    fun = mp[event.key]
                    pos_delta[event.key] = fun(state[controls.target], state[controls.up])
                elif event.key == K_ESCAPE:
                    done = True
                elif event.key == K_p:
                    print "X, Y, Z:", state[controls.pos][0], state[controls.pos][1], state[controls.pos][2]
                    print "Target:", state[controls.target][0], state[controls.target][1], state[controls.target][2]
                    print "Up:", state[controls.up][0], state[controls.up][1], state[controls.up][2]
                    print "Hrot, Vrot:", degrees(state[controls.hrot]), degrees(state[controls.vrot])
            elif event.type == KEYUP:
                if event.key in mp:
                    fun = mp[event.key]
                    del pos_delta[event.key]
            elif event.type == MOUSEMOTION and event.buttons[0] != 0:
                    handle_mouse_move(state, event.rel[0], event.rel[1])
                    keys_pressed = pygame.key.get_pressed()
                    for key, fun in mp.items():
                        if keys_pressed[key] != 0:
                            pos_delta[key] = fun(state[controls.target], state[controls.up])
            elif event.type == QUIT:
                done = True

        wait_time = clock.tick(60)
        frames += 1
        curr_time = pygame.time.get_ticks()
        if start_time + 2000 < curr_time:
            print "FPS:", frames / 2.0
            start_time = curr_time
            frames = 0

world_front = numpy.array([1, 0, 0], numpy.float32)
world_up = numpy.array([0, 1, 0], numpy.float32)

def handle_mouse_move(state, xdiff, ydiff):
    state[controls.hrot] += xdiff * rot_step
    state[controls.vrot] += ydiff * rot_step

    view = deepcopy(world_front)
    rotate_vector(view, state[controls.hrot], world_up)
    normalize(view)

    haxis = numpy.cross(world_up, view)
    normalize(haxis)
    rotate_vector(view, state[controls.vrot], haxis)
    normalize(view)

    state[controls.target] = view
    state[controls.up] = numpy.cross(state[controls.target], haxis)
    normalize(state[controls.up])

def rotate_vector(vec, ang, axe):
    sinhalf = sin(ang / 2)
    coshalf = cos(ang / 2)

    rx = axe[0] * sinhalf
    ry = axe[1] * sinhalf
    rz = axe[2] * sinhalf
    rw = coshalf

    rot = Quaternion(rx, ry, rz, rw)
    conq = rot.conjugated()
    w1 = conq.mult_v(vec)
    w  = w1.mult_q(rot)

    vec[0] = w[0]
    vec[1] = w[1]
    vec[2] = w[2]

class Quaternion:
    def __init__(self, x, y, z, w):
        self.array = numpy.array([x, y, z, w], numpy.float32)

    def __getitem__(self, x):
        return self.array[x]

    def conjugated(self):
        return Quaternion(-self[0], -self[1], -self[2], self[3])

    def mult_v(self, v):
        w = - (self[0] * v[0]) - (self[1] * v[1]) - (self[2] * v[2])
        x =   (self[3] * v[0]) + (self[1] * v[2]) - (self[2] * v[1])
        y =   (self[3] * v[1]) + (self[2] * v[0]) - (self[0] * v[2])
        z =   (self[3] * v[2]) + (self[0] * v[1]) - (self[1] * v[0])
        return Quaternion(x, y, z, w)

    def mult_q(self, q):
        w = - (self[3] * q[3]) - (self[0] * q[0]) - (self[1] * q[1]) - (self[2] * q[2])
        x =   (self[0] * q[3]) + (self[3] * q[0]) + (self[1] * q[2]) - (self[2] * q[1])
        y =   (self[1] * q[3]) + (self[3] * q[1]) + (self[2] * q[0]) - (self[0] * q[2])
        z =   (self[2] * q[3]) + (self[3] * q[2]) + (self[0] * q[1]) - (self[1] * q[0])
        return Quaternion(x, y, z, w)

if __name__ == '__main__':
    main()

