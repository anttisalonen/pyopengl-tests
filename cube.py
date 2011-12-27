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
vertices = numpy.array([0.0, 0.5, 0.0,
                       -0.5, -0.5, 0.0,
                        0.5, -0.5, 0.0], numpy.float32)
indices = numpy.array([0, 1, 2], numpy.ushort)

def getFileContents(filename):
    return open(filename, 'r').read()

def init():
    vertexShader = compileShader(getFileContents("triangle.vert"), GL_VERTEX_SHADER)
    fragmentShader = compileShader(getFileContents("triangle.frag"), GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertexShader)
    glAttachShader(program, fragmentShader)
    glBindAttribLocation(program, 0, "vPosition")
    glBindAttribLocation(program, 1, "vNormal")
    glBindAttribLocation(program, 2, "vTexcoord")
    glLinkProgram(program)

    glClearColor(0.0, 0.0, 0.0, 1.0)
    return program

def drawWithoutVBOs(vertices, indices):
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    glEnableVertexAttribArray(0)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, vertices)

    glDrawElementsus(GL_TRIANGLES, indices)

def drawWithVBOs(vertices, indices, normals = None, texturecoords = None):
    databuf = [vertices]
    if normals:
        databuf.append(normals)
    if texturecoords:
        databuf.append(texturecoords)
    numbuffers = len(databuf)
    vboids = glGenBuffers(numbuffers + 1)

    for i in xrange(numbuffers):
        glBindBuffer(GL_ARRAY_BUFFER, vboids[i])
        glBufferData(GL_ARRAY_BUFFER, databuf[i], GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboids[numbuffers])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)

    for i in xrange(numbuffers):
        glBindBuffer(GL_ARRAY_BUFFER, vboids[i])
        glEnableVertexAttribArray(i)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    if normals:
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    if texturecoords:
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)

    glDrawElementsus(GL_TRIANGLES, indices)

    glDeleteBuffers(4, vboids)

def main():
    pygame.init()
    pygame.display.set_mode((width, height), HWSURFACE|OPENGL|DOUBLEBUF)

    program = init()
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(program)

    drawWithoutVBOs(vertices, indices)
    pygame.display.flip()
    time.sleep(1)

    glClear(GL_COLOR_BUFFER_BIT)
    drawWithVBOs(vertices, indices)
    pygame.display.flip()
    time.sleep(1)

if __name__ == '__main__':
    main()

