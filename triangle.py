#!/usr/bin/env python2

import time

from OpenGL.GL import *
from OpenGL.GL.shaders import *

import pygame
from pygame.locals import *
import numpy    

width = 640
height = 480

def getFileContents(filename):
    return open(filename, 'r').read()

def init():
    vertexShader = compileShader(getFileContents("triangle.vert"), GL_VERTEX_SHADER)
    fragmentShader = compileShader(getFileContents("triangle.frag"), GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertexShader)
    glAttachShader(program, fragmentShader)
    glBindAttribLocation(program, 0, "vPosition")
    glLinkProgram(program)

    glClearColor(0.0, 0.0, 0.0, 1.0)
    return program

def draw(program):
    vertices = numpy.array([0.0, 0.5, 0.0,
                           -0.5, -0.5, 0.0,
                            0.5, -0.5, 0.0], numpy.float32)
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(program)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, vertices)
    glEnableVertexAttribArray(0)

    glDrawArrays(GL_TRIANGLES, 0, 3)

    pygame.display.flip()

def main():
    pygame.init()
    pygame.display.set_mode((width, height), HWSURFACE|OPENGL|DOUBLEBUF)

    program = init()
    draw(program)

    time.sleep(4)

if __name__ == '__main__':
    main()

