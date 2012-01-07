#!/usr/bin/env python2

import time
from math import radians, degrees, cos, sin, tan, pi
from copy import deepcopy

from OpenGL.GL import *
from OpenGL.GL.shaders import *

import pygame
from pygame.locals import *
import numpy    
import numpy.linalg
from pyassimp import pyassimp

width = 640
height = 480

modelview = numpy.identity(4, numpy.float32)

vertex_shader_code = """
attribute vec3 a_Position;
attribute vec2 a_texCoord;
attribute vec3 a_Normal;

uniform mat4 u_MVP;
uniform vec3 u_pointLightPosition;

varying vec2 v_texCoord;
varying vec3 v_Normal;
varying float v_PointLightDistance;

void main()
{
    gl_Position = u_MVP * vec4(a_Position, 1.0);
    v_texCoord = a_texCoord;
    v_Normal = a_Normal;
    v_PointLightDistance = distance(a_Position, u_pointLightPosition);
}
"""

fragment_shader_code = """
varying vec2 v_texCoord;
varying vec3 v_Normal;
varying float v_PointLightDistance;

uniform sampler2D s_texture;
uniform vec3 u_ambientLight;
uniform vec3 u_directionalLightDirection;
uniform vec3 u_directionalLightColor;
uniform vec3 u_pointLightColor;
uniform vec3 u_pointLightAttenuation;
uniform bool u_ambientLightEnabled;
uniform bool u_directionalLightEnabled;
uniform bool u_pointLightEnabled;

void main()
{
    vec4 light;
    float directionalFactor;
    vec4 directionalLight;
    vec4 pointLight;
    float pointLightFactor;

    light = vec4(0.0);

    if(u_ambientLightEnabled)
        light = vec4(u_ambientLight, 1.0);

    if(u_directionalLightEnabled) {
        directionalFactor = dot(normalize(v_Normal), -u_directionalLightDirection);
        if(directionalFactor > 0.0)
            directionalLight = vec4(u_directionalLightColor, 1.0) * directionalFactor;
        else
            directionalLight = vec4(0.0);

        light += directionalLight;
    }

    if(u_pointLightEnabled) {
        pointLightFactor = 1.0 / (u_pointLightAttenuation.x + u_pointLightAttenuation.y * v_PointLightDistance +
                    u_pointLightAttenuation.z * v_PointLightDistance * v_PointLightDistance);
        pointLightFactor = clamp(pointLightFactor, 0, 1);
        pointLight = vec4(pointLightFactor * u_pointLightColor, 1.0);
        light += pointLight;
    }

    light = clamp(light, 0, 1);
    gl_FragColor = texture2D(s_texture, v_texCoord) * light;
}
"""

def load_opengl_texture(filename):
    img = pygame.image.load(filename)
    imgdata = pygame.image.tostring(img, "RGBA", 1)
    imgwidth, imgheight = img.get_size()
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imgwidth, imgheight, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, imgdata)
    try:
        glGenerateMipmap(GL_TEXTURE_2D)
    except NameError:
        # OpenGL < 3.0
        # TODO: implement generating mipmaps here
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    else:
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    return texid

def init(models):
    program = glCreateProgram()

    for model in models:
        vertices = model.vertices
        indices = model.indices
        texcoords = model.texcoords
        normals = model.normals
        print "# of vertices:", len(vertices) / 3
        print "# of indices:", len(indices)
        print "# of texture coordinates:", len(texcoords) / 2
        print "# of normals:", len(normals) / 3

        vboids = glGenBuffers(4)
        model.buffers = []

        for idx, (data, size, attr) in enumerate([(vertices, 3, 'a_Position'),
                                                  (texcoords, 2, 'a_texCoord'),
                                                  (normals, 3, 'a_Normal')]):
            glBindBuffer(GL_ARRAY_BUFFER, vboids[idx])
            glBufferData(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW)
            glEnableVertexAttribArray(idx)
            glBindAttribLocation(program, idx, attr)
            model.buffers.append(VBO(vboids[idx], idx, size, attr))

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboids[3])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)

        model.program = program

    vertexShader = compileShader(vertex_shader_code, GL_VERTEX_SHADER)
    fragmentShader = compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    glAttachShader(program, vertexShader)
    glAttachShader(program, fragmentShader)
    glLinkProgram(program)

    global mvpLoc
    global stextLoc
    global ambientLoc
    global directionalLightDirLoc
    global directionalLightColLoc
    global pointLightPosLoc
    global pointLightAttnLoc
    global pointLightColLoc
    global ambientLightEnabledLoc
    global directionalLightEnabledLoc
    global pointLightEnabledLoc
    mvpLoc = glGetUniformLocation(program, "u_MVP")
    stextLoc = glGetUniformLocation(program, "s_texture")
    ambientLoc = glGetUniformLocation(program, "u_ambientLight")
    directionalLightDirLoc = glGetUniformLocation(program, "u_directionalLightDirection")
    directionalLightColLoc = glGetUniformLocation(program, "u_directionalLightColor")
    pointLightPosLoc = glGetUniformLocation(program, "u_pointLightPosition")
    pointLightAttnLoc = glGetUniformLocation(program, "u_pointLightAttenuation")
    pointLightColLoc = glGetUniformLocation(program, "u_pointLightColor")
    ambientLightEnabledLoc = glGetUniformLocation(program, "u_ambientLightEnabled")
    directionalLightEnabledLoc = glGetUniformLocation(program, "u_directionalLightEnabled")
    pointLightEnabledLoc = glGetUniformLocation(program, "u_pointLightEnabled")

    glClearColor(0.0, 0.0, 0.0, 1.0)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    return program

def drawWithVBOs(entities, state, lights):
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

    # Camera translation
    trans_cam = numpy.identity(4, numpy.float32)
    trans_cam[3][0] = -state[controls.pos][0]
    trans_cam[3][1] = -state[controls.pos][1]
    trans_cam[3][2] = -state[controls.pos][2]

    # Camera rotation
    target_vec = state[controls.target]
    up_vec = state[controls.up]
    rotate_cam = get_camera_rotation_matrix(target_vec, up_vec)

    # Ambient light
    curr_time = pygame.time.get_ticks()
    glUniform1i(ambientLightEnabledLoc, int(lights.ambientLightEnabled))
    if lights.ambientLightEnabled:
        point_in_time = radians((curr_time / 80) % 360)
        rvalue = 0.5 * sin(point_in_time)
        gvalue = 0.5 * sin(point_in_time + 2 * pi / 3)
        bvalue = 0.5 * sin(point_in_time + 4 * pi / 3)
        glUniform3f(ambientLoc, rvalue, gvalue, bvalue)

    # Directional light
    glUniform1i(directionalLightEnabledLoc, int(lights.directionalLightEnabled))
    if lights.directionalLightEnabled:
        directional_dir = normalized(numpy.array([-1, -1, 1], numpy.float32))
        glUniform3f(directionalLightDirLoc, directional_dir[0],
                directional_dir[1], directional_dir[2])
        glUniform3f(directionalLightColLoc, 1.0, 1.0, 1.0)

    # Point light
    glUniform1i(pointLightEnabledLoc, int(lights.pointLightEnabled))
    glUniform3f(pointLightAttnLoc, 0, 0, 6)
    glUniform3f(pointLightColLoc, 1, 1, 1)

    pointLightTime = radians((curr_time / 10) % 360)
    pointLightPos = [sin(pointLightTime), 0.5, cos(pointLightTime)]
    entities[2].position[0] = pointLightPos[0]
    entities[2].position[1] = pointLightPos[1]
    entities[2].position[2] = pointLightPos[2]

    for entity in entities:
        # Texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, entity.model.texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glUniform1i(stextLoc, 0)

        # Translation
        trans = numpy.identity(4, numpy.float32)
        trans[3][0] = entity.position[0]
        trans[3][1] = entity.position[1]
        trans[3][2] = entity.position[2]

        # Rotation
        xrot = entity.rotation[0]
        yrot = entity.rotation[1]
        zrot = entity.rotation[2]
        rotat = rotation_matrix(xrot, yrot, zrot)

        # Modelview
        modelview = rotat.dot(trans.dot(trans_cam.dot(rotate_cam.dot(pers))))
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, modelview)

        if lights.pointLightEnabled:
            glUniform3f(pointLightPosLoc, pointLightPos[0] - entity.position[0],
                    pointLightPos[1] - entity.position[1],
                    pointLightPos[2] - entity.position[2])

        for buf in entity.model.buffers:
            glBindBuffer(GL_ARRAY_BUFFER, buf.vboid)
            glVertexAttribPointer(buf.attribidx, buf.attrsize, GL_FLOAT, GL_FALSE, 0, None)

        glDrawElements(GL_TRIANGLES, len(entity.model.indices), GL_UNSIGNED_SHORT, None)

def rotation_matrix(xrot, yrot, zrot):
    # NOTE: this could be optimized.
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

def load_model(filename):
    return AssetModel(filename)

def flatten(l):
    return [item for sublist in l for item in sublist]

class Model:
    def __init__(self):
        pass

class VBO:
    def __init__(self, vboid, attribidx, attrsize, attr):
        self.vboid = vboid
        self.attribidx = attribidx
        self.attrsize = attrsize
        self.attr = attr

    def __del__(self):
        glDeleteBuffers(1, [self.vboid])

class AssetModel(Model):
    def __init__(self, filename):
        self.model = pyassimp.load(filename)
        self.vertices = numpy.array(flatten(self.model.meshes[0].vertices), numpy.float32)
        self.make_indices()
        # NOTE: this drops the W from UVW-coordinates.
        self.texcoords = numpy.array([item for sublist in self.model.meshes[0].texcoords[0] for item in sublist[:2]], numpy.float32)
        self.normals = numpy.array(flatten(self.model.meshes[0].normals), numpy.float32)

    def __del__(self):
        pyassimp.release(self.model)

    def make_indices(self):
        initial_indices = [face.indices for face in self.model.meshes[0].faces]
        indices = []
        for face in initial_indices:
            if len(face) == 4:
                indices += [face[0], face[1], face[2], face[0], face[2], face[3]]
            elif len(face) == 3:
                indices += face
            else:
                raise ValueError, "Face has %d indices" % len(face)
        self.indices = numpy.array(indices, numpy.ushort)

class Entity:
    def __init__(self, model):
        self.position = numpy.array([0, 0, 0], numpy.float32)
        self.rotation = numpy.array([0, 0, 0], numpy.float32)
        self.model = model

class Lights:
    ambientLightEnabled = True
    directionalLightEnabled = True
    pointLightEnabled = True

def main():
    model = load_model('textured-cube.obj')
    model2 = load_model('bigger-cube.obj')
    model3 = load_model('ball-lamp.obj')
    pygame.init()
    pygame.display.set_mode((width, height), HWSURFACE | OPENGL | DOUBLEBUF)
    lights = Lights()

    models = [model, model2, model3]
    program = init(models)
    texid = load_opengl_texture('snow.jpg')
    texid2 = load_opengl_texture('blue.jpg')
    model.texid = texid
    model2.texid = texid2
    model3.texid = texid2
    entities = [Entity(model), Entity(model2), Entity(model3)]
    entities[1].position[0] = -1.5
    entities[1].position[2] = -1.6
    entities[1].rotation[0] = radians(60)
    entities[1].rotation[1] = radians(35)

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
        glEnable(GL_TEXTURE_2D)
        glUseProgram(program)
        drawWithVBOs(entities, state, lights)
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
                elif event.key == K_F1:
                    lights.ambientLightEnabled = not lights.ambientLightEnabled
                elif event.key == K_F2:
                    lights.directionalLightEnabled = not lights.directionalLightEnabled
                elif event.key == K_F3:
                    lights.pointLightEnabled = not lights.pointLightEnabled
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

    glDeleteTextures(texid2)
    glDeleteTextures(texid)
    glDeleteProgram(program)

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

