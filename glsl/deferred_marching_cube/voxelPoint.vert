#version 450

layout (location = 0) in vec3 inPos;

layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 modelview;
} ubo;

void main() 
{
	gl_Position = ubo.projection * ubo.modelview * vec4(inPos.xyz, 1.0);
	gl_PointSize = 16.0;
}
