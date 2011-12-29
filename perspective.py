#!/usr/bin/env python2

import time
from math import radians, degrees, cos, sin, tan

from OpenGL.GL import *
from OpenGL.GL.shaders import *

import pygame
from pygame.locals import *
import numpy    

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
    trans[3][0] = state[controls.xpos]
    trans[3][1] = state[controls.ypos]
    trans[3][2] = state[controls.zpos]

    # Rotation
    xrot = state[controls.xrot]
    yrot = state[controls.yrot]
    zrot = state[controls.zrot]
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

    modelview = rotat.dot(trans.dot(pers))
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, modelview)

    glDrawElementsus(GL_TRIANGLES, indices)

class controls:
    xrot = 1
    yrot = 2
    zrot = 3
    xpos = 4
    ypos = 5
    zpos = 6

rot_step = radians(1.0)
pos_step = 0.01
steps = dict()
steps[controls.xrot] = rot_step
steps[controls.yrot] = rot_step
steps[controls.zrot] = rot_step
steps[controls.xpos] = pos_step
steps[controls.ypos] = pos_step
steps[controls.zpos] = pos_step

mp = dict()
mp[K_q] = (controls.xrot, True)
mp[K_a] = (controls.xrot, False)
mp[K_w] = (controls.yrot, True)
mp[K_s] = (controls.yrot, False)
mp[K_e] = (controls.zrot, True)
mp[K_d] = (controls.zrot, False)
mp[K_LEFT] = (controls.xpos, False)
mp[K_RIGHT] = (controls.xpos, True)
mp[K_PAGEUP] = (controls.ypos, True)
mp[K_PAGEDOWN] = (controls.ypos, False)
mp[K_UP] = (controls.zpos, True)
mp[K_DOWN] = (controls.zpos, False)

def main():
    pygame.init()
    pygame.display.set_mode((width, height), HWSURFACE | OPENGL | DOUBLEBUF)

    program = init()

    done = False
    state = dict()

    state[controls.xrot] = radians(45)
    state[controls.yrot] = radians(135)
    state[controls.zrot] = radians(90)
    state[controls.xpos] = 0
    state[controls.ypos] = 0
    state[controls.zpos] = -1.55
    delta = dict()
    delta[controls.xrot] = 0
    delta[controls.yrot] = 0
    delta[controls.zrot] = 0
    delta[controls.xpos] = 0
    delta[controls.ypos] = 0
    delta[controls.zpos] = 0

    while not done:
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)
        drawWithoutVBOs(state)
        pygame.display.flip()
        for key, value in state.items():
            state[key] += delta[key]
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key in mp:
                    what, up = mp[event.key]
                    step = steps[what]
                    delta[what] = step if up else -step
                elif event.key == K_ESCAPE:
                    done = True
                elif event.key == K_p:
                    print "X, Y, Z:", state[controls.xpos], state[controls.ypos], state[controls.zpos]
                    print "Rotation:", degrees(state[controls.xrot]), degrees(state[controls.yrot]), degrees(state[controls.zrot])
            elif event.type == KEYUP:
                if event.key in mp:
                    what, up = mp[event.key]
                    delta[what] = 0

if __name__ == '__main__':
    main()

