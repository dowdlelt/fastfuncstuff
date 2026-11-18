#version 440
// The domain's bounding box, in scanner mm; the fragment shader marches the
// ray through it. Uniform layout must match viewer/vox3d.py:pack_uniforms.

layout(location = 0) in vec3 pos;

layout(location = 0) out vec3 vWorld;

layout(std140, binding = 0) uniform Block {
    mat4 mvp;
    mat4 domainFromMm;
    mat4 ovFromMm;
    mat4 anFromMm;
    vec4 eye;      // xyz camera in mm, w samples per unit of domain texture
    vec4 ovShade;  // lo, hi, threshold, opacity
    vec4 anShade;  // lo, hi, opacity, -
    vec4 flags;    // mode, sign mode, has overlay, has anatomy
    vec4 clip;     // plane normal xyz (mm), offset; zero normal = no cut
    vec4 bg;       // background rgb, has stat
};

void main()
{
    vWorld = pos;
    gl_Position = mvp * vec4(pos, 1.0);
}
