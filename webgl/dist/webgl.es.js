import { vec3, mat3, mat4, vec4, quat, vec2 } from './gl-matrix/index.js';

class Bounds {
    constructor() {
        this.min = vec3.fromValues(Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
        this.max = vec3.fromValues(Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE);
    }
    static createFromMinMax(min, max) {
        const bounds = new Bounds();
        bounds.setMinMax(min, max);
        return bounds;
    }
    setMinMax(min, max) {
        vec3.copy(this.min, min);
        vec3.copy(this.max, max);
    }
    center() {
        let c = vec3.create();
        vec3.add(c, this.min, this.max);
        vec3.scale(c, c, 0.5);
        return c;
    }
    invalidate() {
        vec3.set(this.min, Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
        vec3.set(this.max, Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE);
    }
    encapsulateBounds(bounds) {
        this.encapsulatePoint(bounds.min);
        this.encapsulatePoint(bounds.max);
    }
    encapsulatePoint(point) {
        if (point[0] < this.min[0])
            this.min[0] = point[0];
        if (point[0] > this.max[0])
            this.max[0] = point[0];
        if (point[1] < this.min[1])
            this.min[1] = point[1];
        if (point[1] > this.max[1])
            this.max[1] = point[1];
        if (point[2] < this.min[2])
            this.min[2] = point[2];
        if (point[2] > this.max[2])
            this.max[2] = point[2];
    }
    // Based on code from: https://github.com/erich666/GraphicsGems/blob/master/gems/TransBox.c
    static transform(out, matrix, bounds) {
        const transform = mat3.create();
        mat3.fromMat4(transform, matrix);
        const translate = vec3.fromValues(matrix[12], matrix[13], matrix[14]);
        /* Take care of translation by beginning at T. */
        out.setMinMax(translate, translate);
        /* Now find the extreme points by considering the product of the min and max with each component of M.  */
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                const a = (transform[i * 3 + j] * bounds.min[j]);
                const b = (transform[i * 3 + j] * bounds.max[j]);
                if (a < b) {
                    out.min[i] += a;
                    out.max[i] += b;
                }
                else {
                    out.min[i] += b;
                    out.max[i] += a;
                }
            }
        }
        return out;
    }
}

class Camera {
    constructor(node) {
        this.node = node;
        this._near = 0.1;
        this._far = 1000.0;
        this._aspect = 1.33;
        this._fovy = 45.0;
        this._matricesDirty = false;
        this._projectionMatrix = mat4.create();
        this._viewMatrix = mat4.create();
        this.cullingMask = 0xFFFF;
    }
    _updateMatrices() {
        if (!this._matricesDirty)
            return;
        mat4.perspective(this._projectionMatrix, this._fovy * Math.PI / 180, this._aspect, this._near, this._far);
        const nodePos = this.node.position;
        let target = this.node.forward();
        vec3.add(target, nodePos, target);
        let up = this.node.up();
        mat4.lookAt(this._viewMatrix, nodePos, target, up);
        this._matricesDirty = false;
    }
    get near() { return this._near; }
    get far() { return this._far; }
    get aspect() { return this._aspect; }
    get fovy() { return this._fovy; }
    get projectionMatrix() {
        this._updateMatrices();
        return this._projectionMatrix;
    }
    get viewMatrix() {
        this._updateMatrices();
        return this._viewMatrix;
    }
    set near(value) {
        if (value != this._near) {
            this._near = value;
            this._matricesDirty = true;
        }
    }
    set far(value) {
        if (value != this._far) {
            this._far = value;
            this._matricesDirty = true;
        }
    }
    set aspect(value) {
        if (value != this._aspect) {
            this._aspect = value;
            this._matricesDirty = true;
        }
    }
    set fovy(value) {
        if (value != this._fovy) {
            this._fovy = value;
            this._matricesDirty = true;
        }
    }
}
class Cameras {
    constructor() {
        this.items = [];
    }
    create(node) {
        const camera = new Camera(node);
        node.components.camera = camera;
        this.items.push(camera);
        return camera;
    }
    clear() {
        this.items = [];
    }
}

var LightType;
(function (LightType) {
    LightType[LightType["Directional"] = 0] = "Directional";
    LightType[LightType["Point"] = 1] = "Point";
    LightType[LightType["Spot"] = 2] = "Spot";
})(LightType || (LightType = {}));
class Light {
    constructor(type, node) {
        this.type = type;
        this.node = node;
        this.color = vec3.fromValues(1.0, 1.0, 1.0);
        this.range = 10.0;
        this.intensity = 1.0;
        this.spotInnerAngle = 12.5;
        this.spotOuterAngle = 45.0;
        this.layerMask = 0xFFFF;
    }
    get direction() {
        return this.node.forward();
    }
}
class Lights {
    constructor() {
        this.items = [];
    }
    clear() {
        this.items = [];
    }
    create(node, type) {
        if (this.items.length === Lights.maxLightCount)
            throw new Error("Maximum number of lights have already been created.");
        const light = new Light(type, node);
        this.items.push(light);
        return light;
    }
}
Lights.maxLightCount = 5;

class Material {
    constructor(shader) {
        this.shader = shader;
        this.program = null;
    }
}

/**
 * The value of each attribute corresponds to the attribute location in the shader
 */
var AttributeType;
(function (AttributeType) {
    AttributeType[AttributeType["Position"] = 0] = "Position";
    AttributeType[AttributeType["Normal"] = 1] = "Normal";
    AttributeType[AttributeType["TexCoord0"] = 2] = "TexCoord0";
})(AttributeType || (AttributeType = {}));
/**
 * Bitwise values indicating which attributes are active in a shader
 */
var AttributeFlag;
(function (AttributeFlag) {
    AttributeFlag[AttributeFlag["None"] = 0] = "None";
    AttributeFlag[AttributeFlag["Position"] = 1] = "Position";
    AttributeFlag[AttributeFlag["Normal"] = 2] = "Normal";
    AttributeFlag[AttributeFlag["TexCoord0"] = 4] = "TexCoord0";
})(AttributeFlag || (AttributeFlag = {}));
class Attribute {
    constructor(type, componentType, componentCount, count, offset, stride, buffer) {
        this.type = type;
        this.componentType = componentType;
        this.componentCount = componentCount;
        this.count = count;
        this.offset = offset;
        this.stride = stride;
        this.buffer = buffer;
    }
}
class ElementBuffer {
    constructor(componentType, count, offset, buffer) {
        this.componentType = componentType;
        this.count = count;
        this.offset = offset;
        this.buffer = buffer;
    }
}
class Primitive {
    constructor(type, indices, attributes, bounds, baseMaterial) {
        this.type = type;
        this.indices = indices;
        this.attributes = attributes;
        this.bounds = bounds;
        this.baseMaterial = baseMaterial;
        this.attributeMask = this._calculateAttributeMask();
    }
    _calculateAttributeMask() {
        let mask = AttributeFlag.None;
        for (const attribute of this.attributes) {
            switch (attribute.type) {
                case AttributeType.Position:
                    mask |= AttributeFlag.Position;
                    break;
                case AttributeType.Normal:
                    mask |= AttributeFlag.Normal;
                    break;
                case AttributeType.TexCoord0:
                    mask |= AttributeFlag.TexCoord0;
                    break;
            }
        }
        return mask;
    }
}
class Mesh {
    constructor(primitives) {
        this.primitives = primitives;
    }
    freeGlResources(gl) {
        const buffers = new Set();
        for (const primitive of this.primitives) {
            buffers.add(primitive.indices.buffer);
            for (const attribute of primitive.attributes) {
                buffers.add(attribute.buffer);
            }
        }
        buffers.forEach((buffer) => {
            gl.deleteBuffer(buffer);
        });
    }
}
class Meshes {
    constructor(_gl, _shaders) {
        this._gl = _gl;
        this._shaders = _shaders;
        this.items = [];
    }
    create(primitives) {
        const mesh = new Mesh(primitives);
        this.items.push(mesh);
        return mesh;
    }
    clear() {
        for (const mesh of this.items) {
            mesh.freeGlResources(this._gl);
        }
        this.items = [];
    }
}

class ShaderLibrary {
}
ShaderLibrary.commonHeader = `#version 300 es

precision mediump float;
precision mediump int;

const int WGL_LIGHT_DIRECTIONAL = 0;
const int WGL_LIGHT_POINT = 1;
const int WGL_LIGHT_SPOT = 2;

#define M_PI 3.1415926535897932384626433832795
#define WGL_MAX_LIGHTS 5

struct wglLight {
    int type;
    float range;
    float intensity;
    float align1;
    vec3 position;
    float spot_inner_angle;
    vec3 direction;
    float spot_outer_angle;
    vec3 color;
    float align2;
};

layout(std140) uniform wglData {
    mat4 camera_projection;
    mat4 camera_view;
    vec3 camera_world_pos;
    float ambient_light_intensity;
    vec3 ambient_light_color;
    int light_count;
    wglLight lights[WGL_MAX_LIGHTS];
} wgl;

layout(std140) uniform wglObjectData {
    mat4 matrix;
    mat4 normal_matrix;
} wgl_object;
`;
ShaderLibrary.commonVertex = `layout(location = 0) in vec4 wgl_position;
layout(location = 1) in vec3 wgl_normal;
layout(location = 2) in vec2 wgl_tex_coord0;
`;
ShaderLibrary.unlitFragment = `out vec4 final_color;
uniform vec4 diffuse_color;

#ifdef WGL_TEXTURE_COORDS0
in vec2 wgl_tex_coords0;
#endif

#ifdef WGL_UNLIT_DIFFUSE_MAP
uniform sampler2D diffuse_sampler;
#endif

void main() {
    #if defined(WGL_TEXTURE_COORDS0) && defined(WGL_UNLIT_DIFFUSE_MAP)
        final_color = texture(diffuse_sampler, wgl_tex_coords0) * diffuse_color;
    #else
        final_color = diffuse_color;
    #endif
}

`;
ShaderLibrary.unlitVertex = `#ifdef WGL_TEXTURE_COORDS0
out vec2 wgl_tex_coords0;
#endif

void main() {
    gl_Position = wgl.camera_projection * wgl.camera_view * wgl_object.matrix * wgl_position;

    #ifdef WGL_TEXTURE_COORDS0
        wgl_tex_coords0 = wgl_tex_coord0;
    #endif
}
`;
ShaderLibrary.phongFragment = `#ifdef WGL_TEXTURE_COORDS0
in vec2 wgl_tex_coords0;
#endif

#ifdef WGL_PHONG_DIFFUSE_MAP
uniform sampler2D diffuse_sampler;
#endif

#ifdef WGL_PHONG_SPECULAR_MAP
uniform sampler2D specular_sampler;
#endif

#ifdef WGL_PHONG_EMISSION_MAP
uniform sampler2D emission_sampler;
#endif


out vec4 finalColor;

// phong params
uniform vec4 diffuse_color;
uniform float specular_strength;
uniform float shininess;

in vec3 normal;
in vec3 frag_pos;

vec4 calculateSpecular(wglLight light, vec3 light_dir) {
    vec3 norm = normalize(normal);
    vec3 view_dir = normalize(wgl.camera_world_pos - frag_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0f), shininess);
    vec4 specular_color = vec4(specular_strength * spec * light.color , 1.0f);

    #if defined(WGL_TEXTURE_COORDS0) && defined(WGL_PHONG_SPECULAR_MAP)
    specular_color *= texture(specular_sampler, wgl_tex_coords0);
    #endif

    return specular_color;
}

vec4 directionalLight(wglLight light, vec4 object_diffuse_color) {
    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(-light.direction); // note that direction is specified from source, but our calculations following assume light dir is to the source.
    float diff = max(dot(norm, light_dir), 0.0f);
    vec4 diffuse_color = vec4(diff * light.color, 1.0f) * object_diffuse_color;

    diffuse_color += calculateSpecular(light, light_dir);
    return clamp(diffuse_color, 0.0f, 1.0f);
}


vec4 pointLight(wglLight light, vec4 object_diffuse_color) {
    float distance = distance(light.position, frag_pos);

    // approximate inverse square law
    float attenuation = clamp(1.0 - (distance * distance) / (light.range * light.range), 0.0, 1.0);
    attenuation *= attenuation;

    //float b = 1.0 / (light.range * light.range * 0.01);
    //attenuation = 1.0 / (1.0 + (b * distance * distance));

    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(light.position - frag_pos);
    float diff = max(dot(norm, light_dir), 0.0f);
    vec4 diffuse_color = vec4(diff * light.color * light.intensity, 1.0f) * object_diffuse_color;

    diffuse_color += calculateSpecular(light, light_dir) * attenuation;

    return clamp(diffuse_color, 0.0f, 1.0f);
}

vec4 spotLight(wglLight light, vec4 object_diffuse_color) {
    vec3 frag_to_light = normalize(light.position - frag_pos);
    float theta = dot(frag_to_light, -light.direction);

    float angle = acos(theta) * 180.0 / M_PI;
    float epsilon = light.spot_inner_angle - light.spot_outer_angle;
    float spot_modifier = smoothstep(0.0, 1.0, (angle - light.spot_outer_angle) / epsilon);

    vec4 diffuse_color = pointLight(light, object_diffuse_color) * spot_modifier;

    return clamp(diffuse_color, 0.0f, 1.0f);
}

void main() {
    vec4 object_diffuse_color = diffuse_color;
    #if defined(WGL_TEXTURE_COORDS0) && defined(WGL_PHONG_DIFFUSE_MAP)
    object_diffuse_color *= texture(diffuse_sampler, wgl_tex_coords0);
    #endif

    // calculate ambient
    vec4 frag_color = vec4(wgl.ambient_light_color * wgl.ambient_light_intensity, 1.0f) * object_diffuse_color;

    // calculate diffuse
    for (int i = 0; i < wgl.light_count; i++) {
        switch(wgl.lights[i].type) {
            case WGL_LIGHT_DIRECTIONAL:
                frag_color += directionalLight(wgl.lights[i], object_diffuse_color);
                break;

            case WGL_LIGHT_POINT:
                frag_color += pointLight(wgl.lights[i], object_diffuse_color);
                break;

            case WGL_LIGHT_SPOT:
                frag_color += spotLight(wgl.lights[i], object_diffuse_color);
                break;
        }
    }

    // calculate emission
    #if defined(WGL_TEXTURE_COORDS0) && defined(WGL_PHONG_EMISSION_MAP)
    vec4 emission = texture(emission_sampler, wgl_tex_coords0);
    frag_color += emission;
    #endif

    finalColor = frag_color;
}

`;
ShaderLibrary.phongVertex = `#ifdef WGL_TEXTURE_COORDS0
out vec2 wgl_tex_coords0;
#endif

out vec3 normal;
out vec3 frag_pos;

void main() {
    gl_Position = wgl.camera_projection * wgl.camera_view * wgl_object.matrix * wgl_position;

    #ifdef WGL_TEXTURE_COORDS0
    wgl_tex_coords0 = wgl_tex_coord0;
    #endif

    normal = mat3(wgl_object.normal_matrix) * wgl_normal;
    frag_pos = (wgl_object.matrix * wgl_position).xyz;
}
`;

class ShaderProgram {
    constructor(
    /** The unique hash for this program.
     * This distinguishes it from other compiled versions of the same shader source code.
     * The hash is a combination of primitive (16 bits) attributes and shader features (16 bits).
     */
    programHash, 
    /** Handle to compiled shader object. */
    program, 
    /** The index of global uniform block.  This uniform block represents common data such as camera and lighting info that is passed to all shaders from the webgl engine */
    globalBlockIndex, 
    /** Index of the object block.  This is object specific data such as transform  and normal matrix passed to the shader by the webgl engine. */
    objectBlockIndex) {
        this.programHash = programHash;
        this.program = program;
        this.globalBlockIndex = globalBlockIndex;
        this.objectBlockIndex = objectBlockIndex;
    }
}
class Shaders {
    constructor(_gl, defaultPhong, defaultUnlit) {
        this._gl = _gl;
        this.defaultPhong = defaultPhong;
        this.defaultUnlit = defaultUnlit;
        this._shaderPrograms = new Map();
    }
    updateProgram(material, primitive) {
        material.program = null;
        if (material.shader === null)
            return;
        let shaderPrograms = this._shaderPrograms.get(material.shader);
        if (!shaderPrograms) {
            shaderPrograms = [];
            this._shaderPrograms.set(material.shader, shaderPrograms);
        }
        const programHash = primitive.attributeMask | (material.featureMask() << 16);
        for (const program of shaderPrograms) {
            if (program.programHash === programHash) {
                material.program = program;
                return;
            }
        }
        if (material.program === null) {
            this._compileShader(primitive, material, programHash);
        }
    }
    clear() {
        this._shaderPrograms.forEach((programs) => {
            for (const program of programs) {
                this._gl.deleteProgram(program.program);
            }
        });
        this._shaderPrograms.clear();
    }
    _compileShader(primitive, material, programHash) {
        try {
            const attributePreprocessorStatements = Shaders.attributePreprocessorStatements(primitive);
            const shaderPreprocessorStatements = material.shader.preprocessorStatements(material);
            const preprocessorCode = attributePreprocessorStatements.concat(shaderPreprocessorStatements).join('\n') + "\n";
            const vertexSource = ShaderLibrary.commonHeader + ShaderLibrary.commonVertex + preprocessorCode + material.shader.vertexSource;
            const fragmentSource = ShaderLibrary.commonHeader + preprocessorCode + material.shader.fragmentSource;
            const webglProgram = this._webglCompile(vertexSource, fragmentSource);
            const wglGlobalDataIndex = this._gl.getUniformBlockIndex(webglProgram, 'wglData');
            const wglObjectDataLocation = this._gl.getUniformBlockIndex(webglProgram, 'wglObjectData');
            console.assert(wglGlobalDataIndex !== -1 && wglObjectDataLocation !== -1);
            const shaderProgram = material.shader.programCompiled(this._gl, material, programHash, webglProgram, wglGlobalDataIndex, wglObjectDataLocation);
            material.program = shaderProgram;
            let shaderPrograms = this._shaderPrograms.get(material.shader);
            shaderPrograms.push(shaderProgram);
        }
        catch (e) {
            throw new Error(`${material.shader.name}: ${e.toString()}`);
        }
    }
    _webglCompile(vertexSource, fragmentSource) {
        const gl = this._gl;
        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexSource);
        gl.compileShader(vertexShader);
        if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
            const errorMessage = `Error compiling vertex shader: ${gl.getShaderInfoLog(vertexShader)}`;
            gl.deleteShader(vertexShader);
            throw new Error(errorMessage);
        }
        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragmentSource);
        gl.compileShader(fragmentShader);
        if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
            const errorMessage = `Error compiling fragment shader: ${gl.getShaderInfoLog(fragmentShader)}`;
            gl.deleteShader(vertexShader);
            gl.deleteShader(fragmentShader);
            throw new Error(errorMessage);
        }
        const shaderProgram = gl.createProgram();
        gl.attachShader(shaderProgram, vertexShader);
        gl.attachShader(shaderProgram, fragmentShader);
        gl.linkProgram(shaderProgram);
        if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
            const errorMessage = `Error linking shader program: ${gl.getProgramInfoLog(shaderProgram)}`;
            gl.deleteProgram(shaderProgram);
            gl.deleteShader(vertexShader);
            gl.deleteShader(fragmentShader);
            throw new Error(errorMessage);
        }
        return shaderProgram;
    }
    static attributePreprocessorStatements(primitive) {
        const preprocessorStatements = [];
        for (const attribute of primitive.attributes) {
            switch (attribute.type) {
                case AttributeType.Position:
                    preprocessorStatements.push("#define WGL_POSITIONS");
                    break;
                case AttributeType.Normal:
                    preprocessorStatements.push("#define WGL_NORMALS");
                    break;
                case AttributeType.TexCoord0:
                    preprocessorStatements.push("#define WGL_TEXTURE_COORDS0");
                    break;
            }
        }
        return preprocessorStatements;
    }
}

var PhongShaderFeatures;
(function (PhongShaderFeatures) {
    PhongShaderFeatures[PhongShaderFeatures["DiffuseMap"] = 1] = "DiffuseMap";
    PhongShaderFeatures[PhongShaderFeatures["SpecularMap"] = 2] = "SpecularMap";
    PhongShaderFeatures[PhongShaderFeatures["EmissionMap"] = 4] = "EmissionMap";
})(PhongShaderFeatures || (PhongShaderFeatures = {}));
class PhongMaterial extends Material {
    constructor(shader) {
        super(shader);
        this.diffuseColor = vec4.fromValues(1.0, 1.0, 1.0, 1.0);
        this.specularStrength = 0.5;
        this.shininess = 32.0;
        this.diffuseMap = null;
        this.specularMap = null;
        this.emissionMap = null;
    }
    featureMask() {
        let mask = 0;
        if (this.diffuseMap)
            mask |= PhongShaderFeatures.DiffuseMap;
        if (this.specularMap)
            mask |= PhongShaderFeatures.SpecularMap;
        if (this.emissionMap)
            mask |= PhongShaderFeatures.EmissionMap;
        return mask;
    }
    copy() {
        const material = new PhongMaterial(this.shader);
        vec4.copy(material.diffuseColor, this.diffuseColor);
        material.specularStrength = this.specularStrength;
        material.shininess = this.shininess;
        material.diffuseMap = this.diffuseMap;
        material.specularMap = this.specularMap;
        material.emissionMap = this.emissionMap;
        return material;
    }
}
class PhongShaderProgram extends ShaderProgram {
}
class PhongShader {
    constructor() {
        this.name = "Phong";
        this.vertexSource = ShaderLibrary.phongVertex;
        this.fragmentSource = ShaderLibrary.phongFragment;
    }
    preprocessorStatements(material) {
        const phongMaterial = material;
        const statements = [];
        if (phongMaterial.diffuseMap)
            statements.push("#define WGL_PHONG_DIFFUSE_MAP");
        if (phongMaterial.specularMap)
            statements.push("#define WGL_PHONG_SPECULAR_MAP");
        if (phongMaterial.emissionMap)
            statements.push("#define WGL_PHONG_EMISSION_MAP");
        return statements;
    }
    programCompiled(gl, material, programHash, program, globalBlockIndex, objectBlockIndex) {
        const phongShaderProgram = new PhongShaderProgram(programHash, program, globalBlockIndex, objectBlockIndex);
        phongShaderProgram.diffuseColorLocation = gl.getUniformLocation(program, "diffuse_color");
        phongShaderProgram.specularStrengthLocation = gl.getUniformLocation(program, "specular_strength");
        phongShaderProgram.shininessLocation = gl.getUniformLocation(program, "shininess");
        phongShaderProgram.diffuseSamplerLocation = gl.getUniformLocation(program, "diffuse_sampler");
        phongShaderProgram.specularSamplerLocation = gl.getUniformLocation(program, "specular_sampler");
        phongShaderProgram.emissionSamplerLocation = gl.getUniformLocation(program, "emission_sampler");
        return phongShaderProgram;
    }
    setUniforms(gl, material) {
        const phongMaterial = material;
        const phongProgram = material.program;
        gl.uniform4fv(phongProgram.diffuseColorLocation, phongMaterial.diffuseColor);
        gl.uniform1f(phongProgram.specularStrengthLocation, phongMaterial.specularStrength);
        gl.uniform1f(phongProgram.shininessLocation, phongMaterial.shininess);
        let textureUnit = 0;
        if (phongMaterial.diffuseMap) {
            gl.activeTexture(gl.TEXTURE0 + textureUnit);
            gl.bindTexture(gl.TEXTURE_2D, phongMaterial.diffuseMap.handle);
            gl.uniform1i(phongProgram.diffuseSamplerLocation, textureUnit++);
        }
        if (phongMaterial.specularMap) {
            gl.activeTexture(gl.TEXTURE0 + textureUnit);
            gl.bindTexture(gl.TEXTURE_2D, phongMaterial.specularMap.handle);
            gl.uniform1i(phongProgram.specularSamplerLocation, textureUnit++);
        }
        if (phongMaterial.emissionMap) {
            gl.activeTexture(gl.TEXTURE0 + textureUnit);
            gl.bindTexture(gl.TEXTURE_2D, phongMaterial.emissionMap.handle);
            gl.uniform1i(phongProgram.emissionSamplerLocation, textureUnit++);
        }
    }
}

class PrimitiveData {
    constructor(elements, material) {
        this.elements = elements;
        this.material = material;
    }
}
class MeshData {
    constructor() {
        this.positions = null;
        this.normals = null;
        this.texCoords0 = null;
        this.bounds = new Bounds();
        this.primitives = [];
        vec3.set(this.bounds.min, 0.0, 0.0, 0.0);
        vec3.set(this.bounds.max, 0.0, 0.0, 0.0);
    }
    addPrimitive(elements, material) {
        this.primitives.push(new PrimitiveData(elements, material));
    }
    _vertexBufferSize() {
        let size = 0;
        if (this.positions)
            size += this.positions.byteLength;
        if (this.normals)
            size += this.normals.byteLength;
        if (this.texCoords0)
            size += this.texCoords0.byteLength;
        return size;
    }
    create(scene) {
        const gl = scene.gl;
        const vertexGlBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexGlBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this._vertexBufferSize(), gl.STATIC_DRAW);
        const attributes = [];
        let offset = 0;
        if (this.positions) {
            gl.bufferSubData(gl.ARRAY_BUFFER, offset, this.positions);
            attributes.push(new Attribute(AttributeType.Position, gl.FLOAT, 3, this.positions.length / 3, offset, 0, vertexGlBuffer));
            offset += this.positions.byteLength;
        }
        if (this.normals) {
            gl.bufferSubData(gl.ARRAY_BUFFER, offset, this.normals);
            attributes.push(new Attribute(AttributeType.Normal, gl.FLOAT, 3, this.normals.length / 3, offset, 0, vertexGlBuffer));
            offset += this.normals.byteLength;
        }
        if (this.texCoords0) {
            gl.bufferSubData(gl.ARRAY_BUFFER, offset, this.texCoords0);
            attributes.push(new Attribute(AttributeType.TexCoord0, gl.FLOAT, 2, this.texCoords0.length / 2, offset, 0, vertexGlBuffer));
            offset += this.texCoords0.length;
        }
        const primitives = [];
        for (const primitiveData of this.primitives) {
            const elementGlBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elementGlBuffer);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, primitiveData.elements, gl.STATIC_DRAW);
            const elementBuffer = new ElementBuffer(gl.UNSIGNED_SHORT, primitiveData.elements.length, 0, elementGlBuffer);
            const material = new PhongMaterial(scene.shaders.defaultPhong);
            const primitive = new Primitive(gl.TRIANGLES, elementBuffer, attributes, this.bounds, material);
            scene.shaders.updateProgram(material, primitive);
            primitives.push(primitive);
        }
        return scene.meshes.create(primitives);
    }
}

class MeshInstance {
    constructor(node, mesh) {
        this.worldBounds = new Bounds();
        this.layerMask = 1;
        this.node = node;
        this.mesh = mesh;
        this._instanceMaterials = new Array(this.mesh.primitives.length);
        this.updateBounds();
    }
    getReadonlyMaterial(index) {
        return this._instanceMaterials[index] ? this._instanceMaterials[index] : this.mesh.primitives[index].baseMaterial;
    }
    updateBounds() {
        Bounds.transform(this.worldBounds, this.node.worldMatrix, this.mesh.primitives[0].bounds);
    }
}
class MeshInstances {
    constructor(_renderer) {
        this._renderer = _renderer;
    }
    create(node, mesh) {
        const meshInstance = this._renderer.createMeshInstance(node, mesh);
        node.components.meshInstance = meshInstance;
        return meshInstance;
    }
}

class MathUtil {
    static extractTRS(matrix, translation, rotation, scale) {
        mat4.getScaling(scale, matrix);
        // To extract a correct rotation, the scaling component must be eliminated.
        const mn = mat4.create();
        for (const col of [0, 1, 2]) {
            mn[col] = matrix[col] / scale[0];
            mn[col + 4] = matrix[col + 4] / scale[1];
            mn[col + 8] = matrix[col + 8] / scale[2];
        }
        mat4.getRotation(rotation, mn);
        quat.normalize(rotation, rotation);
        mat4.getTranslation(translation, matrix);
    }
}

class Node {
    constructor(name = null) {
        this.name = name;
        this.parent = null;
        this.children = new Array();
        this.position = vec3.fromValues(0.0, 0.0, 0.0);
        this.rotation = quat.create();
        this.scale = vec3.fromValues(1.0, 1.0, 1.0);
        this.localMatrix = mat4.create();
        this.worldMatrix = mat4.create();
        this.components = {};
    }
    addChild(child) {
        // remove the child from its parent's children array
        if (child.parent)
            child.parent.children.filter((c) => { return c != child; });
        child.parent = this;
        this.children.push(child);
        child.updateMatrix();
        return child;
    }
    createChild(name) {
        return this.addChild(new Node(name));
    }
    getChild(index) {
        return this.children[index];
    }
    getChildCount() {
        return this.children.length;
    }
    setTransformFromMatrix(matrix) {
        MathUtil.extractTRS(matrix, this.position, this.rotation, this.scale);
    }
    updateMatrix() {
        if (Node.freeze)
            return;
        mat4.identity(this.localMatrix);
        mat4.fromRotationTranslationScale(this.localMatrix, this.rotation, this.position, this.scale);
        if (this.parent)
            mat4.multiply(this.worldMatrix, this.parent.worldMatrix, this.localMatrix);
        else
            mat4.copy(this.worldMatrix, this.localMatrix);
        if (this.components.meshInstance)
            this.components.meshInstance.updateBounds();
        for (const child of this.children)
            child.updateMatrix();
    }
    forward() {
        const fwd = vec3.fromValues(0.0, 0.0, 1.0);
        vec3.transformQuat(fwd, fwd, this.rotation);
        vec3.normalize(fwd, fwd);
        return fwd;
    }
    up() {
        const upp = vec3.fromValues(0.0, 1.0, 0.0);
        vec3.transformQuat(upp, upp, this.rotation);
        vec3.normalize(upp, upp);
        return upp;
    }
    lookAt(target, up) {
        const lookAtMatrix = mat4.create();
        mat4.targetTo(lookAtMatrix, target, this.position, up);
        mat4.getRotation(this.rotation, lookAtMatrix);
        quat.normalize(this.rotation, this.rotation);
    }
    static cleanupNode(node) {
        node.parent = null;
        for (const child of node.children)
            Node.cleanupNode(child);
    }
}
Node.freeze = false;

class Texture {
    constructor(width, height, handle) {
        this.width = width;
        this.height = height;
        this.handle = handle;
    }
    freeGlResources(gl) {
        gl.deleteTexture(this.handle);
    }
}
class Textures {
    constructor(_gl) {
        this._gl = _gl;
        this._textures = new Set();
    }
    createFromImage(image) {
        const gl = this._gl;
        const handle = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, handle);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        gl.generateMipmap(gl.TEXTURE_2D);
        const texture = new Texture(image.width, image.height, handle);
        this._textures.add(texture);
        return texture;
    }
    createFromRGBAData(width, height, data) {
        const gl = this._gl;
        const handle = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, handle);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data, 0);
        gl.generateMipmap(gl.TEXTURE_2D);
        const texture = new Texture(width, height, handle);
        this._textures.add(texture);
        return texture;
    }
    async createFromUrl(url) {
        const image = new Image();
        image.crossOrigin = "anonymous";
        image.src = url;
        await image.decode();
        return this.createFromImage(image);
    }
    async createFromBuffer(buffer, mimeType) {
        const blob = new Blob([buffer], { type: mimeType });
        const url = URL.createObjectURL(blob);
        try {
            return await this.createFromUrl(url);
        }
        finally {
            URL.revokeObjectURL(url);
        }
    }
    clear() {
        this._textures.forEach((texture) => {
            texture.freeGlResources(this._gl);
        });
        this._textures.clear();
    }
}
Textures.defaultWhite = null;
Textures.defaultBlack = null;

class RenderTarget {
    constructor(_width, _height, _handle, _colorTexture, _depthTexture) {
        this._width = _width;
        this._height = _height;
        this._handle = _handle;
        this._colorTexture = _colorTexture;
        this._depthTexture = _depthTexture;
    }
    freeGlResources(gl) {
        this._colorTexture.freeGlResources(gl);
        this._depthTexture.freeGlResources(gl);
        gl.deleteFramebuffer(this._handle);
    }
    get colorTexture() {
        return this._colorTexture;
    }
    get depthTexture() {
        return this._depthTexture;
    }
    get handle() {
        return this._handle;
    }
    get width() {
        return this._colorTexture.width;
    }
    get height() {
        return this._colorTexture.height;
    }
}
class RenderTargets {
    constructor(_gl) {
        this._gl = _gl;
        this.items = [];
    }
    create(width, height) {
        const gl = this._gl;
        const colorTexture = this._createColorTexture(width, height);
        const depthTexture = this._createDepthTexture(width, height);
        const framebuffer = gl.createFramebuffer();
        this._gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTexture.handle, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTexture.handle, 0);
        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status !== gl.FRAMEBUFFER_COMPLETE) {
            colorTexture.freeGlResources(gl);
            depthTexture.freeGlResources(gl);
            throw new Error("Failed to create render target");
        }
        const renderTarget = new RenderTarget(width, height, framebuffer, colorTexture, depthTexture);
        this.items.push(renderTarget);
        return renderTarget;
    }
    _createColorTexture(width, height) {
        const gl = this._gl;
        const colorTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, colorTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, width, height, 0, gl.RGB, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
        return new Texture(width, height, colorTexture);
    }
    _createDepthTexture(width, height) {
        const gl = this._gl;
        const depthTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, depthTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, width, height, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
        return new Texture(width, height, depthTexture);
    }
    clear() {
        for (const renderTarget of this.items) {
            renderTarget.freeGlResources(this._gl);
        }
        this.items = [];
    }
}

/**
 * This class encapsulates the per-object uniform data that is set before each draw call
 */
class ObjectUniformBuffer {
    constructor(gl) {
        this.gl = gl;
        this._data = new ArrayBuffer(ObjectUniformBuffer.size);
        this._matrixView = new Float32Array(this._data, 0, 16);
        this._normalMatrixView = new Float32Array(this._data, 16 * 4, 16);
        this._glBuffer = this.gl.createBuffer();
        gl.bindBuffer(gl.UNIFORM_BUFFER, this._glBuffer);
        gl.bufferData(gl.UNIFORM_BUFFER, this._data, gl.DYNAMIC_DRAW);
        gl.bindBufferRange(gl.UNIFORM_BUFFER, ObjectUniformBuffer.defaultBindIndex, this._glBuffer, 0, this._data.byteLength);
    }
    updateGpuBuffer() {
        this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this._glBuffer);
        this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 0, this._data);
    }
    get matrix() {
        return this._matrixView;
    }
    get normalMatrix() {
        return this._normalMatrixView;
    }
}
ObjectUniformBuffer.size = (16 + 16) * 4;
ObjectUniformBuffer.defaultBindIndex = 1;
/**
 * This class encasulates the global uniform data that is set once per shader program activation.
    Layout:
    mat4 camera_projection (64)
    mat4 camera_view       (64)
    vec3 camera_world_pos   (12)
    float ambient_light_intensity (4)
    vec3 ambient_light_color (12)
    int lightCount                  (4)
    Light lights[5]                 (64) * 5
 */
class UniformBuffer {
    constructor(gl) {
        this.gl = gl;
        this._data = new ArrayBuffer(this.sizeInBytes);
        this._floatView = new Float32Array(this._data);
        this._dataView = new DataView(this._data);
        // set up the default uniform buffer that is passed to all shaders
        // note that the default uniform buffer is bound at location 0
        this._glBuffer = this.gl.createBuffer();
        gl.bindBuffer(gl.UNIFORM_BUFFER, this._glBuffer);
        gl.bufferData(gl.UNIFORM_BUFFER, this._data, gl.DYNAMIC_DRAW);
        gl.bindBufferRange(gl.UNIFORM_BUFFER, UniformBuffer.defaultBindIndex, this._glBuffer, 0, this._data.byteLength);
    }
    get sizeInBytes() {
        return UniformBuffer.baseDataSize + (UniformBuffer.lightStructSize * Lights.maxLightCount);
    }
    // upload the latest standard shader data to the gl buffer on gpu
    updateGpuBuffer() {
        this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this._glBuffer);
        this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 0, this._data);
    }
    set cameraProjection(projectionMatrix) {
        this._floatView.set(projectionMatrix, 0);
    }
    set cameraView(viewMatrix) {
        this._floatView.set(viewMatrix, 16);
    }
    set cameraWorldPos(position) {
        this._floatView.set(position, 32);
    }
    set ambientColor(color) {
        this._floatView.set(color, 36);
    }
    set ambientIntensity(intensity) {
        this._dataView.setFloat32(140, intensity, true);
    }
    set lightCount(value) {
        this._dataView.setInt32(156, value, true);
    }
    setLight(index, light) {
        const lightBaseByteIndex = UniformBuffer.baseDataSize + (index * UniformBuffer.lightStructSize);
        const lightBaseFloatIndex = lightBaseByteIndex / 4;
        this._dataView.setInt32(lightBaseByteIndex, light.type, true);
        this._dataView.setFloat32(lightBaseByteIndex + 4, light.range, true);
        this._dataView.setFloat32(lightBaseByteIndex + 8, light.intensity, true);
        this._dataView.setFloat32(lightBaseByteIndex + 28, light.spotInnerAngle, true);
        this._dataView.setFloat32(lightBaseByteIndex + 44, light.spotOuterAngle, true);
        this._floatView.set(light.node.position, lightBaseFloatIndex + 4);
        this._floatView.set(light.direction, lightBaseFloatIndex + 8);
        this._floatView.set(light.color, lightBaseFloatIndex + 12);
    }
}
UniformBuffer.defaultBindIndex = 0;
UniformBuffer.baseDataSize = 40 * 4;
UniformBuffer.lightStructSize = 16 * 4;

class DrawCall {
    constructor(meshInstance, primitive) {
        this.meshInstance = meshInstance;
        this.primitive = primitive;
    }
}
class Renderer {
    constructor(gl, lights) {
        this.camera = null;
        this._drawCalls = new Map();
        this._lightMask = 0xFFFF;
        this._meshInstances = [];
        this.renderTarget = null;
        this.gl = gl;
        this._lights = lights;
        this._uniformBuffer = new UniformBuffer(this.gl);
        this._uniformBuffer.ambientColor = vec4.fromValues(1.0, 1.0, 1.0, 1.0);
        this._uniformBuffer.ambientIntensity = 0.1;
        this._objectUniformBuffer = new ObjectUniformBuffer(this.gl);
    }
    _updateCamera() {
        if (this.renderTarget)
            this.camera.aspect = this.renderTarget.width / this.renderTarget.height;
        else
            this.camera.aspect = this.gl.canvas.width / this.gl.canvas.height;
        this._uniformBuffer.cameraProjection = this.camera.projectionMatrix;
        this._uniformBuffer.cameraView = this.camera.viewMatrix;
        this._uniformBuffer.cameraWorldPos = this.camera.node.position;
    }
    prepareDraw() {
        this._drawCalls.clear();
        for (const meshInstance of this._meshInstances) {
            if ((this.camera.cullingMask & meshInstance.layerMask) === 0)
                continue;
            for (let i = 0; i < meshInstance.mesh.primitives.length; i++) {
                // Temporary
                if (meshInstance.mesh.primitives[i].type != this.gl.TRIANGLES)
                    return;
                const material = meshInstance.getReadonlyMaterial(i);
                const drawCall = new DrawCall(meshInstance, i);
                if (this._drawCalls.has(material.program))
                    this._drawCalls.get(material.program).push(drawCall);
                else
                    this._drawCalls.set(material.program, [drawCall]);
            }
        }
    }
    createMeshInstance(node, mesh) {
        this._meshInstances.push(new MeshInstance(node, mesh));
        return this._meshInstances[this._meshInstances.length - 1];
    }
    clear() {
        this._meshInstances = [];
        this.camera = null;
    }
    updateLights() {
        let lightCount = 0;
        for (let i = 0; i < this._lights.items.length; i++) {
            const light = this._lights.items[i];
            if ((this._lightMask & light.layerMask) === 0)
                continue;
            this._uniformBuffer.setLight(lightCount++, light);
        }
        this._uniformBuffer.lightCount = lightCount;
    }
    draw() {
        const gl = this.gl;
        this._updateCamera();
        this.prepareDraw();
        this._lightMask = 0;
        if (this.renderTarget) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.renderTarget.handle);
            gl.viewport(0, 0, this.renderTarget.colorTexture.width, this.renderTarget.colorTexture.height);
        }
        else {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        }
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        this._drawCalls.forEach((drawables, program) => {
            this._drawList(program, drawables);
        });
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }
    _drawList(shaderProgram, drawCalls) {
        this.gl.useProgram(shaderProgram.program);
        // send the default data to the newly active shader
        this.gl.uniformBlockBinding(shaderProgram.program, shaderProgram.globalBlockIndex, UniformBuffer.defaultBindIndex);
        this.gl.uniformBlockBinding(shaderProgram.program, shaderProgram.objectBlockIndex, ObjectUniformBuffer.defaultBindIndex);
        // todo: dont create me over and over
        const normalMatrix = mat4.create();
        for (const drawCall of drawCalls) {
            // check the lighting state
            if (drawCall.meshInstance.layerMask != this._lightMask) {
                this._lightMask = drawCall.meshInstance.layerMask;
                this.updateLights();
                this._uniformBuffer.updateGpuBuffer();
            }
            const matrix = drawCall.meshInstance.node.worldMatrix;
            const material = drawCall.meshInstance.getReadonlyMaterial(drawCall.primitive);
            const primitive = drawCall.meshInstance.mesh.primitives[drawCall.primitive];
            // set the uniform buffer values for this particular object and upload to GPU
            this._objectUniformBuffer.matrix.set(matrix, 0);
            mat4.invert(normalMatrix, matrix);
            mat4.transpose(normalMatrix, normalMatrix);
            this._objectUniformBuffer.normalMatrix.set(normalMatrix, 0);
            this._objectUniformBuffer.updateGpuBuffer();
            material.shader.setUniforms(this.gl, material);
            for (const attribute of primitive.attributes) {
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, attribute.buffer);
                this.gl.vertexAttribPointer(attribute.type, attribute.componentCount, attribute.componentType, false, attribute.stride, attribute.offset);
                this.gl.enableVertexAttribArray(attribute.type);
            }
            this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, primitive.indices.buffer);
            this.gl.drawElements(primitive.type, primitive.indices.count, primitive.indices.componentType, primitive.indices.offset);
            for (const attribute of primitive.attributes) {
                this.gl.disableVertexAttribArray(attribute.type);
            }
        }
    }
}

var UnlitShaderFeatures;
(function (UnlitShaderFeatures) {
    UnlitShaderFeatures[UnlitShaderFeatures["DiffuseMap"] = 1] = "DiffuseMap";
})(UnlitShaderFeatures || (UnlitShaderFeatures = {}));
class UnlitShaderProgram extends ShaderProgram {
}
class UnlitShader {
    constructor() {
        this.name = "Unlit";
        this.vertexSource = ShaderLibrary.unlitVertex;
        this.fragmentSource = ShaderLibrary.unlitFragment;
    }
    preprocessorStatements(material) {
        const unlitMaterial = material;
        const statements = [];
        if (unlitMaterial.diffuseMap)
            statements.push("#define WGL_UNLIT_DIFFUSE_MAP");
        return statements;
    }
    programCompiled(gl, material, programHash, program, globalBlockIndex, objectBlockIndex) {
        const unlitShaderProgram = new UnlitShaderProgram(programHash, program, globalBlockIndex, objectBlockIndex);
        unlitShaderProgram.diffuseColorLocation = gl.getUniformLocation(program, "diffuse_color");
        unlitShaderProgram.diffuseSamplerLocation = gl.getUniformLocation(program, "diffuse_sampler");
        return unlitShaderProgram;
    }
    setUniforms(gl, material) {
        const unlitMaterial = material;
        const unlitProgram = material.program;
        gl.uniform4fv(unlitProgram.diffuseColorLocation, unlitMaterial.diffuseColor);
        let textureUnit = 0;
        if (unlitMaterial.diffuseMap) {
            gl.activeTexture(gl.TEXTURE0 + textureUnit);
            gl.bindTexture(gl.TEXTURE_2D, unlitMaterial.diffuseMap.handle);
            gl.uniform1i(unlitProgram.diffuseSamplerLocation, textureUnit++);
        }
    }
}

class Scene {
    constructor(canvasOrSelector) {
        this.worldBounding = Bounds.createFromMinMax(vec3.fromValues(-1.0, -1.0, -1.0), vec3.fromValues(1.0, 1.0, 1.0));
        this.rootNode = null;
        if (canvasOrSelector instanceof HTMLCanvasElement) {
            this.canvas = canvasOrSelector;
        }
        else if (typeof (canvasOrSelector) === "string") {
            const glCanvas = document.querySelector("#gl-canvas");
            if (glCanvas === null)
                throw new Error(`Unable to locate canvas with selector: ${canvasOrSelector}`);
            this.canvas = glCanvas;
        }
        this.canvasResized();
        this.gl = this.canvas.getContext('webgl2');
        if (this.gl === null) {
            throw new Error("Unable to initialize WebGL 2.0");
        }
        this.cameras = new Cameras();
        this.lights = new Lights();
        this.renderer = new Renderer(this.gl, this.lights);
        this.textures = new Textures(this.gl);
        this.shaders = new Shaders(this.gl, new PhongShader(), new UnlitShader());
        this.meshes = new Meshes(this.gl, this.shaders);
        this.renderTargets = new RenderTargets(this.gl);
        this.meshInstances = new MeshInstances(this.renderer);
    }
    async init() {
        this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
        this.gl.clearDepth(1.0);
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.depthFunc(this.gl.LEQUAL);
        this.createDefault();
    }
    clear() {
        Node.cleanupNode(this.rootNode);
        this.cameras.clear();
        this.meshes.clear();
        this.textures.clear();
        this.renderer.clear();
        this.shaders.clear();
        this.lights.clear();
        this.renderTargets.clear();
    }
    canvasResized() {
        // set the client's width and height to their actual screen size.
        // This is needed in order for the webgl drawing buffer to be correctly sized.
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
    }
    calculateWorldBounding() {
        this.worldBounding.invalidate();
        Scene._getBoundsRec(this.rootNode, this.worldBounding);
        return this.worldBounding;
    }
    getNodeBounding(node, bounding) {
        Scene._getBoundsRec(node, bounding);
    }
    static _getBoundsRec(node, bounds) {
        if (node.components.meshInstance)
            bounds.encapsulateBounds(node.components.meshInstance.worldBounds);
        const childCount = node.getChildCount();
        for (let i = 0; i < childCount; i++) {
            Scene._getBoundsRec(node.getChild(i), bounds);
        }
    }
    createDefault() {
        this.rootNode = new Node("root");
        const cameraNode = new Node("Main Camera");
        vec3.set(cameraNode.position, 0.0, 7.0, 10.0);
        cameraNode.updateMatrix();
        cameraNode.lookAt(vec3.fromValues(0.0, 1.0, 0.0), cameraNode.up());
        this.cameras.create(cameraNode);
        this.rootNode.addChild(cameraNode);
        this.renderer.camera = this.cameras.items[0];
        const directionalLightNode = new Node("Directional Light");
        directionalLightNode.components.light = this.lights.create(directionalLightNode, LightType.Directional);
        directionalLightNode.position = vec3.fromValues(0.0, 3, 0.0);
        quat.fromEuler(directionalLightNode.rotation, 50.0, -30.0, 0.0);
        directionalLightNode.updateMatrix();
        this.rootNode.addChild(directionalLightNode);
    }
}

class Headlight {
    constructor(light, camera) {
        this.light = light;
        this.camera = camera;
        this.update();
    }
    update() {
        vec3.copy(this.light.node.position, this.camera.node.position);
        quat.copy(this.light.node.rotation, this.camera.node.rotation);
    }
    reset(light, camera) {
        this.light = light;
        this.camera = camera;
        this.update();
    }
}

class Arcball {
    constructor(camera, scene) {
        this._distance = 0.0;
        this._diagonal = 0.0;
        this._rotX = 0.0;
        this._rotY = 0.0;
        this._target = vec3.create();
        this.rotationSpeed = 90.0;
        this._dragging = false;
        this._previousTime = performance.now();
        this._previousPos = vec2.create();
        this._currentPos = vec2.create();
        this.camera = camera;
        this._scene = scene;
        scene.canvas.onpointerdown = (event) => { this._onPointerDown(event); };
        scene.canvas.onpointermove = (event) => { this._onPointerMove(event); };
        scene.canvas.onpointerup = (event) => { this._onPointerUp(event); };
        scene.canvas.onwheel = (event) => { this._onWheel(event); };
    }
    update(updateTime) {
        if (this._dragging && !vec2.equals(this._previousPos, this._currentPos)) {
            const deltaX = this._currentPos[0] - this._previousPos[0];
            const deltaY = this._currentPos[1] - this._previousPos[1];
            const deltaTime = (updateTime - this._previousTime) / 1000;
            this._orbit(deltaX, deltaY, deltaTime);
            vec2.copy(this._previousPos, this._currentPos);
        }
        this._previousTime = updateTime;
    }
    setInitial(worldBounding) {
        this._target = worldBounding.center();
        this._diagonal = vec3.distance(worldBounding.min, worldBounding.max);
        this._distance = this._diagonal * 2.0;
        vec3.copy(this.camera.node.position, worldBounding.max);
        this._setCameraPos();
    }
    _orbit(deltaX, deltaY, deltaTime) {
        const rotationAmount = this.rotationSpeed * deltaTime;
        this._rotY -= deltaX * rotationAmount;
        this._rotX -= deltaY * rotationAmount;
        this._setCameraPos();
    }
    _zoom(delta) {
        this._distance += delta * this._diagonal * 0.1;
        this._setCameraPos();
    }
    _setCameraPos() {
        this._rotX = Math.max(Math.min(this._rotX, 90.0), -90.0);
        const q = quat.create();
        quat.fromEuler(q, this._rotX, this._rotY, 0.0);
        const orbitPos = vec3.fromValues(0.0, 0.0, 1.0);
        vec3.transformQuat(orbitPos, orbitPos, q);
        vec3.normalize(orbitPos, orbitPos);
        const upVec = vec3.fromValues(0.0, 1.0, 0.0);
        vec3.transformQuat(upVec, upVec, q);
        vec3.normalize(upVec, upVec);
        const dir = vec3.create();
        vec3.add(orbitPos, orbitPos, this._target);
        vec3.subtract(dir, orbitPos, this._target);
        vec3.normalize(dir, dir);
        vec3.scale(dir, dir, this._distance);
        vec3.add(this.camera.node.position, this._target, dir);
        this.camera.node.lookAt(this._target, upVec);
        this.camera.node.components.camera._matricesDirty = true;
    }
    _setCurrentPos(event) {
        const clientRect = this._scene.canvas.getBoundingClientRect();
        vec2.set(this._currentPos, event.clientX - clientRect.left, event.clientY - clientRect.top);
    }
    _onPointerDown(event) {
        this._dragging = true;
        this._setCurrentPos(event);
        vec2.copy(this._previousPos, this._currentPos);
    }
    _onPointerMove(event) {
        if (!this._dragging)
            return;
        this._setCurrentPos(event);
    }
    _onPointerUp(_event) {
        this._dragging = false;
    }
    _onWheel(event) {
        event.preventDefault();
        const delta = event.deltaY > 0 ? 1 : -1;
        this._zoom(delta);
    }
}

var ComponentType;
(function (ComponentType) {
    ComponentType[ComponentType["Byte"] = 5120] = "Byte";
    ComponentType[ComponentType["UnsignedByte"] = 5121] = "UnsignedByte";
    ComponentType[ComponentType["Short"] = 5122] = "Short";
    ComponentType[ComponentType["UnsignedShort"] = 5123] = "UnsignedShort";
    ComponentType[ComponentType["UnsignedInt"] = 5125] = "UnsignedInt";
    ComponentType[ComponentType["Float"] = 5126] = "Float";
})(ComponentType || (ComponentType = {}));
var PrimitiveMode;
(function (PrimitiveMode) {
    PrimitiveMode[PrimitiveMode["Points"] = 0] = "Points";
    PrimitiveMode[PrimitiveMode["Lines"] = 1] = "Lines";
    PrimitiveMode[PrimitiveMode["LineLoop"] = 2] = "LineLoop";
    PrimitiveMode[PrimitiveMode["LineStrip"] = 3] = "LineStrip";
    PrimitiveMode[PrimitiveMode["Triangles"] = 4] = "Triangles";
    PrimitiveMode[PrimitiveMode["TriangleStrip"] = 5] = "TriangleStrip";
    PrimitiveMode[PrimitiveMode["TriangleFan"] = 6] = "TriangleFan";
})(PrimitiveMode || (PrimitiveMode = {}));
var BufferViewTarget;
(function (BufferViewTarget) {
    BufferViewTarget[BufferViewTarget["ArrayBuffer"] = 34962] = "ArrayBuffer";
    BufferViewTarget[BufferViewTarget["ElementArrayBuffer"] = 34963] = "ElementArrayBuffer";
})(BufferViewTarget || (BufferViewTarget = {}));

class BinaryGltfChunk {
    constructor(type, view) {
        this.type = type;
        this.view = view;
    }
}
const _MagicNum = 0x46546C67;
const _JsonChunk = 0x4E4F534A;
const _BinaryChunk = 0x004E4942;
function _validateHeader(buffer) {
    const view = new DataView(buffer, 0, 12);
    const magic = view.getUint32(0, true);
    if (magic !== _MagicNum)
        throw new Error(`Unexpected magic number in header: ${magic}.`);
    const version = view.getUint32(4, true);
    if (version !== 2)
        throw new Error(`Unexpected glTF version in header: ${version}.`);
    const dataLength = view.getUint32(8, true);
    if (dataLength !== buffer.byteLength)
        throw new Error(`Unexpected data length in header: ${dataLength}.  Expected: ${view.byteLength}.`);
}
function _parseChunks(buffer) {
    const view = new DataView(buffer);
    const chunks = [];
    let offset = 12; // sizeof(header)
    while (offset < view.byteLength) {
        const chunkLength = view.getUint32(offset, true);
        const chunkType = view.getUint32(offset + 4, true);
        const chunkView = new DataView(buffer, offset + 8, chunkLength);
        chunks.push(new BinaryGltfChunk(chunkType, chunkView));
        offset += 8 + chunkLength;
    }
    return chunks;
}
function _parse(data) {
    _validateHeader(data);
    const chunks = _parseChunks(data);
    if (chunks[0].type !== _JsonChunk)
        throw new Error("First binary glTF chunk should have type of 'Structured JSON content'.");
    const textDecoder = new TextDecoder();
    const json = JSON.parse(textDecoder.decode(chunks[0].view));
    let binary = null;
    const extras = [];
    for (let i = 1; i < chunks.length; i++) {
        if (chunks[i].type === _BinaryChunk)
            binary = chunks[i].view;
        else
            extras.push(chunks[i]);
    }
    return new BinaryGltf(json, binary, extras);
}
class BinaryGltf {
    constructor(json, binary, extras) {
        this.json = json;
        this.binary = binary;
        this.extras = extras;
    }
    static parse(data) {
        return _parse(data);
    }
}

class GLTFLoader {
    constructor(_scene) {
        this._scene = _scene;
        this._gltf = null;
        this._glb = null;
        this._meshes = null;
        this._arrayBuffers = null;
        this._bufferViews = null;
        this._glBuffers = null;
        this._textures = null;
        this._materials = null;
        this.autoscaleScene = true;
        this.meshInstanceLayerMask = 1;
        this.rootNode = null;
        if (GLTFLoader._attributeNameToType.size === 0) {
            GLTFLoader._attributeNameToType.set("POSITION", AttributeType.Position);
            GLTFLoader._attributeNameToType.set("NORMAL", AttributeType.Normal);
            GLTFLoader._attributeNameToType.set("TEXCOORD_0", AttributeType.TexCoord0);
        }
    }
    async _requestResource(url) {
        const response = await fetch(url);
        if (response.status != 200)
            throw new Error(`Unable to load gltf file at: ${url}`);
        const index = url.lastIndexOf("/");
        this._baseUrl = index >= 0 ? url.substring(0, index + 1) : "";
        return response;
    }
    async _load() {
        if (this.rootNode === null)
            this.rootNode = this._scene.rootNode;
        this._meshes = this._gltf.meshes ? new Array(this._gltf.meshes.length) : null;
        this._arrayBuffers = this._gltf.buffers ? new Array(this._gltf.buffers.length) : null;
        this._bufferViews = this._gltf.bufferViews ? new Array(this._gltf.bufferViews.length) : null;
        this._glBuffers = this._gltf.bufferViews ? new Array(this._gltf.bufferViews.length) : null;
        this._textures = this._gltf.images ? new Array(this._gltf.images.length) : null;
        this._materials = this._gltf.materials ? new Array(this._gltf.materials.length) : null;
        if (this._gltf.scenes && this._gltf.scenes.length > 0)
            return await this._loadScene(this._gltf.scenes[0]);
        else
            return null;
    }
    get meshes() { return this._meshes; }
    get materials() { return this._materials; }
    async load(url) {
        if (url.endsWith(".glb"))
            return this.loadBinary(url);
        else
            return this.loadSeparate(url);
    }
    async loadSeparate(url) {
        const response = await this._requestResource(url);
        this._gltf = JSON.parse(await response.text());
        await this._load();
    }
    async loadBinary(url) {
        const response = await this._requestResource(url);
        this._glb = BinaryGltf.parse(await response.arrayBuffer());
        this._gltf = this._glb.json;
        await this._load();
    }
    async _loadScene(scene) {
        Node.freeze = true;
        let webglNodes = new Array(this._gltf.nodes.length);
        // create all the nodes
        for (let i = 0; i < this._gltf.nodes.length; i++) {
            const node = await this._createNode(this._gltf.nodes[i]);
            webglNodes[i] = node;
        }
        // set the root nodes
        for (const rootNode of scene.nodes) {
            this.rootNode.addChild(webglNodes[rootNode]);
        }
        // set children nodes
        for (let i = 0; i < this._gltf.nodes.length; i++) {
            const gltfNode = this._gltf.nodes[i];
            if (!gltfNode.hasOwnProperty("children"))
                continue;
            for (const child of gltfNode.children) {
                webglNodes[i].addChild(webglNodes[child]);
            }
        }
        // update all matrices
        Node.freeze = false;
        this.rootNode.updateMatrix();
        if (this.autoscaleScene)
            this._autoscaleScene();
        return scene.nodes.map((index) => { return webglNodes[index]; });
    }
    async _createNode(gltfNode) {
        const wglNode = new Node();
        if (gltfNode.translation)
            vec3.copy(wglNode.position, gltfNode.translation);
        if (gltfNode.scale)
            vec3.copy(wglNode.scale, gltfNode.scale);
        if (gltfNode.rotation) {
            quat.copy(wglNode.rotation, gltfNode.rotation);
            quat.normalize(wglNode.rotation, wglNode.rotation);
        }
        if (gltfNode.matrix)
            wglNode.setTransformFromMatrix(gltfNode.matrix);
        if (gltfNode.name)
            wglNode.name = gltfNode.name;
        if (gltfNode.hasOwnProperty("mesh")) {
            const meshInstance = this._scene.meshInstances.create(wglNode, await this._getMesh(gltfNode.mesh));
            meshInstance.layerMask = this.meshInstanceLayerMask;
        }
        return wglNode;
    }
    async _getMesh(index) {
        if (!this._meshes[index]) {
            const gltfMesh = this._gltf.meshes[index];
            const primitives = new Array();
            for (const meshPrimitive of gltfMesh.primitives) {
                const type = this._getPrimitiveType(meshPrimitive);
                const baseMaterial = await this._getMaterial(meshPrimitive);
                const attributes = new Array();
                const attributeNames = Object.keys(meshPrimitive.attributes);
                const bounds = new Bounds();
                for (const attributeName of attributeNames) {
                    const attribute = await this._getAttribute(attributeName, meshPrimitive.attributes[attributeName]);
                    if (attribute !== null)
                        attributes.push(attribute);
                    // position accessor must specify min and max properties
                    if (attributeName == "POSITION") {
                        const gltfAccessor = this._gltf.accessors[meshPrimitive.attributes[attributeName]];
                        vec3.copy(bounds.min, gltfAccessor.min);
                        vec3.copy(bounds.max, gltfAccessor.max);
                    }
                }
                const indicesBuffer = await this._getElementBuffer(meshPrimitive.indices);
                const primitive = new Primitive(type, indicesBuffer, attributes, bounds, baseMaterial);
                primitives.push(primitive);
                this._scene.shaders.updateProgram(baseMaterial, primitive);
            }
            this._meshes[index] = this._scene.meshes.create(primitives);
        }
        return this._meshes[index];
    }
    _getPrimitiveType(primitive) {
        let mode = primitive.hasOwnProperty("mode") ? primitive.mode : PrimitiveMode.Triangles;
        switch (mode) {
            case PrimitiveMode.Triangles:
                return this._scene.gl.TRIANGLES;
            default:
                throw new Error(`Unsupported Primitive Mode: ${mode}`);
        }
    }
    static _getComponentType(componentType, gl) {
        switch (componentType) {
            case ComponentType.Float:
                return gl.FLOAT;
            case ComponentType.UnsignedShort:
                return gl.UNSIGNED_SHORT;
            case ComponentType.UnsignedInt:
                return gl.UNSIGNED_INT;
            case ComponentType.UnsignedByte:
                return gl.UNSIGNED_BYTE;
            default:
                throw new Error(`Unsupported Component Type: ${componentType}`);
        }
    }
    static _getComponentElementCount(type) {
        switch (type) {
            case "SCALAR":
                return 1;
            case "VEC2":
                return 2;
            case "VEC3":
                return 3;
            case "VEC4":
                return 4;
            case "MAT2":
                return 4;
            case "MAT3":
                return 9;
            case "MAT4":
                return 16;
            default:
                throw new Error(`Unsupported Component Type: ${type}`);
        }
    }
    async _getAttribute(gltfName, index) {
        const accessor = this._gltf.accessors[index];
        const bufferView = this._gltf.bufferViews[accessor.bufferView];
        const attributeType = GLTFLoader._attributeNameToType.get(gltfName);
        if (attributeType === undefined)
            return null;
        return new Attribute(attributeType, GLTFLoader._getComponentType(accessor.componentType, this._scene.gl), GLTFLoader._getComponentElementCount(accessor.type), accessor.count, accessor.byteOffset, bufferView.byteStride ? bufferView.byteStride : 0, await this._createGlBufferFromView(accessor.bufferView));
    }
    async _getElementBuffer(index) {
        const accessor = this._gltf.accessors[index];
        return new ElementBuffer(GLTFLoader._getComponentType(accessor.componentType, this._scene.gl), accessor.count, accessor.byteOffset, await this._createGlBufferFromView(accessor.bufferView));
    }
    async _getBufferView(index) {
        if (!this._bufferViews[index]) {
            const bufferView = this._gltf.bufferViews[index];
            const buffer = await this._getBuffer(bufferView.buffer);
            this._bufferViews[index] = new DataView(buffer.buffer, bufferView.byteOffset + buffer.byteOffset, bufferView.byteLength);
        }
        return this._bufferViews[index];
    }
    async _createGlBufferFromView(index) {
        if (!this._glBuffers[index]) {
            const gltfBufferView = this._gltf.bufferViews[index];
            const bufferView = await this._getBufferView(index);
            const gl = this._scene.gl;
            const target = gltfBufferView.target === BufferViewTarget.ArrayBuffer ? gl.ARRAY_BUFFER : gl.ELEMENT_ARRAY_BUFFER;
            const glBuffer = gl.createBuffer();
            gl.bindBuffer(target, glBuffer);
            gl.bufferData(target, bufferView, gl.STATIC_DRAW);
            this._glBuffers[index] = glBuffer;
        }
        return this._glBuffers[index];
    }
    _getFetchUri(uri) {
        if (uri.startsWith("data:"))
            return uri;
        return this._baseUrl + uri;
    }
    async _getBuffer(index) {
        if (!this._arrayBuffers[index]) {
            const buffer = this._gltf.buffers[index];
            // if the buffer does not have a URI, then we are loading from GLB
            if (index === 0 && this._glb !== null && !buffer.uri) {
                this._arrayBuffers[index] = this._glb.binary;
            }
            else {
                const response = await fetch(this._getFetchUri(buffer.uri));
                if (response.status != 200) {
                    throw new Error(`unable to fetch buffer: ${buffer.uri}`);
                }
                this._arrayBuffers[index] = new DataView(await response.arrayBuffer());
            }
        }
        return this._arrayBuffers[index];
    }
    async _getMaterial(primitive) {
        if (primitive.hasOwnProperty("material")) {
            if (!this._materials[primitive.material]) {
                const gltfMaterial = this._gltf.materials[primitive.material];
                let faceMaterial = new PhongMaterial(this._scene.shaders.defaultPhong);
                if (gltfMaterial.pbrMetallicRoughness.baseColorTexture) {
                    faceMaterial.diffuseMap = await this._getTexture(gltfMaterial.pbrMetallicRoughness.baseColorTexture.index);
                }
                if (gltfMaterial.pbrMetallicRoughness.baseColorFactor) {
                    const color = gltfMaterial.pbrMetallicRoughness.baseColorFactor;
                    vec4.set(faceMaterial.diffuseColor, color[0], color[1], color[2], color[3]);
                }
                this._materials[primitive.material] = faceMaterial;
            }
            return this._materials[primitive.material];
        }
        else {
            return new PhongMaterial(this._scene.shaders.defaultPhong);
        }
    }
    async _getTexture(index) {
        if (!this._textures[index]) {
            const image = this._gltf.images[index];
            if (image.bufferView) {
                const bufferView = await this._getBufferView(image.bufferView);
                this._textures[index] = await this._scene.textures.createFromBuffer(bufferView, image.mimeType);
            }
            else {
                this._textures[index] = await this._scene.textures.createFromUrl(this._getFetchUri(image.uri));
            }
        }
        return this._textures[index];
    }
    _autoscaleScene() {
        const worldBounding = this._scene.calculateWorldBounding();
        const minValue = Math.min(worldBounding.min[0], Math.min(worldBounding.min[1], worldBounding.min[2]));
        const maxValue = Math.max(worldBounding.max[0], Math.max(worldBounding.max[1], worldBounding.max[2]));
        const deltaValue = maxValue - minValue;
        const scale = 1.0 / deltaValue;
        vec3.set(this._scene.rootNode.scale, scale, scale, scale);
        this._scene.rootNode.updateMatrix();
        vec3.scale(this._scene.worldBounding.min, this._scene.worldBounding.min, scale);
        vec3.scale(this._scene.worldBounding.max, this._scene.worldBounding.max, scale);
    }
}
GLTFLoader._attributeNameToType = new Map();

class Cube {
    static create(scene) {
        const meshData = new MeshData();
        meshData.positions = new Float32Array([
            -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
            -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
            -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
            -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
            -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
        ]);
        meshData.normals = new Float32Array([
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
        ]);
        meshData.texCoords0 = new Float32Array([
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        ]);
        vec3.set(meshData.bounds.min, -1.0, -1.0, -1.0);
        vec3.set(meshData.bounds.max, 1.0, 1.0, 1.0);
        const elementData = new Uint16Array([
            1, 0, 3, 1, 3, 2,
            5, 4, 7, 5, 7, 6,
            9, 8, 11, 9, 11, 10,
            13, 12, 15, 13, 15, 14,
            17, 16, 19, 17, 19, 18,
            21, 20, 23, 21, 23, 22 // Left
        ]);
        const material = new PhongMaterial(scene.shaders.defaultPhong);
        meshData.addPrimitive(elementData, material);
        return meshData.create(scene);
    }
}

export { Arcball, Attribute, Bounds, Camera, Cube, ElementBuffer, GLTFLoader, Headlight, Light, LightType, Material, Mesh, MeshData, MeshInstance, Node, Primitive, RenderTarget, Scene, ShaderProgram, Texture };
