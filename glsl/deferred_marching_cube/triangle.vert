#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;

layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 modelview;
} ubo;

layout (location = 0) out vec3 outWorldPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outTangent;

void main() 
{
	// Vertex position in world space
	outWorldPos = inPos;

	outNormal = inNormal;

	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) / 4.0;
	
	outTangent = inTangent;
	
	gl_Position = ubo.projection * ubo.modelview * vec4(inPos.xyz, 1.0);
	//gl_Position = vec4(inPos.xyz, 1.0); // tesselation
}
