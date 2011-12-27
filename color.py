#!/usr/bin/env python2

import time

from OpenGL.GL import *
from OpenGL.GL.shaders import *

import pygame
from pygame.locals import *
import numpy    

width = 640
height = 480

vertices = numpy.array([-0.5, 0.5, 0.0,
                       -0.5, -0.5, 0.0,
                        0.5, -0.5, 0.0,
                        0.5,  0.5, 0.0,
                        -0.5, 0.5, 1.0,
                       -0.5, -0.5, 1.0,
                        0.5, -0.5, 1.0,
                        0.5,  0.5, 1.0], numpy.float32)
indices = numpy.array([0, 1, 2,
                       0, 2, 3,
                       0, 3, 7,
                       0, 7, 4,
                       0, 4, 5,
                       0, 5, 1,
                       3, 2, 6,
                       3, 6, 7], numpy.ushort)
colors = numpy.array([1.0, 1.0, 1.0, 1.0,
                      1.0, 0.0, 1.0, 1.0,
                      1.0, 0.0, 0.0, 1.0,
                      1.0, 1.0, 0.0, 1.0,
                      0.0, 1.0, 0.0, 1.0,
                      0.0, 1.0, 1.0, 1.0,
                      0.0, 0.0, 1.0, 1.0,
                      0.0, 0.0, 0.0, 1.0], numpy.float32)

def getFileContents(filename):
    return open(filename, 'r').read()

def init():
    vertexShader = compileShader(getFileContents("color.vert"), GL_VERTEX_SHADER)
    fragmentShader = compileShader(getFileContents("color.frag"), GL_FRAGMENT_SHADER)
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

    glClearColor(0.0, 0.0, 0.0, 1.0)
    return program

def drawWithoutVBOs():
    glDrawElementsus(GL_TRIANGLES, indices)

def main():
    pygame.init()
    pygame.display.set_mode((width, height), HWSURFACE|OPENGL|DOUBLEBUF)

    program = init()
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(program)

    drawWithoutVBOs()
    pygame.display.flip()
    time.sleep(2)

if __name__ == '__main__':
    main()

