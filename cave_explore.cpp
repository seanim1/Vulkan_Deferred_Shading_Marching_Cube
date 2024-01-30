/*
* Based on Sascha Willems's Vulkan repo
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"
#include "VulkanFrameBuffer.hpp"
#include "frustum.hpp"
#include "marchingCube.h"
#include "generateVoxel.h"
#include <queue>
#include <thread>
#include <mutex> 

#define PARTICLE_SIZE 10.0f
#define FLAME_RADIUS 2.0f

// Shadowmap properties
#define SHADOWMAP_DIM 2048
// deferred framebuffer size
#define FB_DIM 2048

class VulkanExample : public VulkanExampleBase
{
public:
	bool tessellation = true;
	int max_emitters_count = 32;
	// Max. number of concurrent threads
	uint32_t numThreads;
	std::mutex mutex_lock;
	int32_t debugDisplayTarget = 0;
	// Custom
	// 16x16x16 voxels
	unsigned int cmdBufferBuildCount = 0;
	//struct Vertices vertices;
	struct Indices {
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} indices;
	struct Voxels {
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} voxels;
	std::vector<glm::vec3> voxelBuffer;
	unsigned int total_terrain_triangle_count;
	
	Chunk* chunkListBuffer[CHUNK_COUNT];
	// Custom end
	struct {
		// particle system
		struct {
			vks::Texture2D fire;
			// Use a custom sampler to change sampler attributes required for rotating the uvs in the shader for alpha blended textures
			VkSampler sampler;
		} particles;
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} ground;
	} textures;

	struct {
		vkglTF::Model skysphere;
	} models;

	// Per-instance data block
	struct InstanceData {
		glm::vec3 pos;
		float vertexCount;
	};

	// Contains the instanced data
	vks::Buffer instanceBuffer;
	// Contains the indirect drawing commands
	vks::Buffer indirectCommandsBuffer;
	vks::Buffer indirectDrawCountBuffer;
	uint32_t indirectDrawCount{ 0 };

	// Indirect draw statistics (updated via compute)
	struct {
		uint32_t drawCount;						// Total number of indirect draw counts to be issued
	} indirectStats;

	// Store the indirect draw commands containing index offsets and instance count per object
	std::vector<VkDrawIndirectCommand> indirectCommands;

	struct Light {
		glm::vec4 position;
		glm::vec3 color;
		float radius;
	};
	std::vector<glm::vec3> colors;
	struct UniformData {
		glm::mat4 projection;
		glm::mat4 view;
		glm::vec4 frustumPlanes[6];
		//float tessLevel = 3.0f;
	} uniformData;

	struct {
		Light lights[34]; // Must match the LIGHT_COUNT define in the shadow and deferred shaders
		glm::vec4 viewPos;
		int debugDisplayTarget = 0;
	} uboComposition;

	struct UBOFire {
		glm::mat4 projection;
		glm::mat4 modelView;
		glm::vec2 viewportDim;
		float pointSize = PARTICLE_SIZE;
	} uboFire;

	vks::Buffer uniformBuffer;
	struct {
		vks::Buffer fire; // Particle System
		vks::Buffer composition; // Deferred
	} uniformBuffers;

	struct {
		VkPipeline ground{ VK_NULL_HANDLE };
		VkPipeline skysphere{ VK_NULL_HANDLE };
		VkPipeline triangle{ VK_NULL_HANDLE };
		VkPipeline voxelPoint{ VK_NULL_HANDLE };
		VkPipeline particles{ VK_NULL_HANDLE };
		VkPipeline composition{ VK_NULL_HANDLE };
	} pipelines;

	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };

	// View frustum for culling invisible objects
	vks::Frustum frustum;
	bool fixedFrustum = false;

	struct Particle {
		glm::vec4 pos;
		glm::vec4 color;
		float alpha;
		float size;
		float rotation;
		// Attributes not used in shader
		glm::vec4 vel;
		float rotationSpeed;
	};
	std::vector<Particle> particleBuffer;
	glm::vec3 minVel = glm::vec3(-3.0f, 0.5f, -3.0f);
	glm::vec3 maxVel = glm::vec3(3.0f, 7.0f, 3.0f);
	std::default_random_engine rndEngine;
	struct {
		VkBuffer buffer;
		VkDeviceMemory memory;
		// Store the mapped address of the particle data for reuse
		void* mappedMemory;
		// Size of the particle buffer in bytes
		size_t size;
	} particles;
	
	struct {
		VkDescriptorSet particles;
		VkDescriptorSet gBuffers;
	} descriptorSets;
	unsigned int lastHit_particle_count = 16;
	unsigned int particle_count = max_emitters_count + lastHit_particle_count;
	std::vector<glm::vec3> emitter_positions;
	int lastHitPositionIndex = 0;
	

	// Deferred
	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		VkImage image;
		VkDeviceMemory mem;
		VkImageView view;
		VkFormat format;
	};
	struct FrameBuffer {
		int32_t width, height;
		VkFramebuffer frameBuffer;
		FrameBufferAttachment position, normal, albedo;
		FrameBufferAttachment depth;
		VkRenderPass renderPass;
	} offScreenFrameBuf;
	// One sampler for the frame buffer color attachments
	VkSampler colorSampler;

	VkCommandBuffer offScreenCmdBuffer = VK_NULL_HANDLE;

	// Semaphore used to synchronize between offscreen and final scene rendering
	VkSemaphore offscreenSemaphore = VK_NULL_HANDLE;

	int highestPowerOf2(int N) {
		return std::pow(2, std::floor(std::log2(N)));
	}

	VulkanExample() : VulkanExampleBase()
	{
		title = "Marching Cube";
		camera.type = Camera::CameraType::firstperson;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 128.0f);
		camera.setRotation(glm::vec3(-45.0f, 135.0f, 0.0f));
		camera.setTranslation(glm::vec3(-5.0f, 0.0f, -5.0f));
		camera.movementSpeed = 15.0f;
		numThreads = highestPowerOf2(std::thread::hardware_concurrency());
		colors.push_back(glm::vec3(1.0f, 0.1f, 0.1f)); // Red
		colors.push_back(glm::vec3(1.0f, 0.5f, 0.1f)); // Orange
		colors.push_back(glm::vec3(1.0f, 1.0f, 0.1f)); // Yellow
		colors.push_back(glm::vec3(0.4f, 1.0f, 0.1f)); // Green
		colors.push_back(glm::vec3(0.1f, 1.0f, 0.6f)); // Lime
		colors.push_back(glm::vec3(0.1f, 1.0, 1.0f)); // Cyan
		colors.push_back(glm::vec3(0.1f, 0.4f, 1.0f)); // Blue
		colors.push_back(glm::vec3(0.4f, 0.1f, 1.0f)); // Purple
		colors.push_back(glm::vec3(0.7f, 0.1f, 1.0f)); // Violet
		colors.push_back(glm::vec3(1.0f, 0.1f, 0.6f)); // Pink
		memset(&indirectStats, 0, sizeof(indirectStats));
	}

	~VulkanExample()
	{
		if (device) {
			vkDestroyPipeline(device, pipelines.ground, nullptr);
			vkDestroyPipeline(device, pipelines.skysphere, nullptr);
			vkDestroyPipeline(device, pipelines.triangle, nullptr);
			vkDestroyPipeline(device, pipelines.voxelPoint, nullptr);
			vkDestroyPipeline(device, pipelines.particles, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
			textures.particles.fire.destroy();
			textures.ground.colorMap.destroy();
			textures.ground.normalMap.destroy();
			instanceBuffer.destroy();
			indirectCommandsBuffer.destroy();
			uniformBuffer.destroy();
			uniformBuffers.fire.destroy();
		}
	}

	// Enable physical device features required for this example
	virtual void getEnabledFeatures()
	{
		// Tessellation shader support enable
		//if (deviceFeatures.tessellationShader) {
		//	enabledFeatures.tessellationShader = VK_TRUE;
		//}
		//else {
		//	vks::tools::exitFatal("Selected GPU does not support tessellation shaders!", VK_ERROR_FEATURE_NOT_PRESENT);
		//}
		// Example uses multi draw indirect if available
		if (deviceFeatures.multiDrawIndirect) {
			enabledFeatures.multiDrawIndirect = VK_TRUE;
		}
		// Enable anisotropic filtering if supported
		if (deviceFeatures.samplerAnisotropy) {
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		}

		// Enable largePoints if supported
		if (deviceFeatures.largePoints) {
			enabledFeatures.largePoints = VK_TRUE;
		}
		enabledFeatures.wideLines = VK_TRUE;
		float minPointSize = deviceProperties.limits.pointSizeRange[0];
		float maxPointSize = deviceProperties.limits.pointSizeRange[1];
		float gran = deviceProperties.limits.pointSizeGranularity;
		deviceProperties.limits.maxComputeWorkGroupInvocations;
		deviceProperties.limits.maxComputeWorkGroupCount;
		deviceProperties.limits.maxComputeWorkGroupSize;

	};

	bool frustumCheck(glm::vec3 pos, float radius)
	{
		// Check sphere against frustum planes
		for (int i = 0; i < 6; i++)
		{
			if (glm::dot(glm::vec4(pos, 1.0f), frustum.planes.data()[i]) + radius < 0.0)
			{
				return false;
			}
		}
		return true;
	}
	void buildCommandBuffers()
	{
		cmdBufferBuildCount++;
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = { { 0.18f, 0.27f, 0.5f, 0.0f } };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = VulkanExampleBase::frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets.gBuffers, 0, nullptr);
			
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.composition);
			// Final composition as full screen quad
			// Note: Also used for debug display if debugDisplayTarget > 0
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			//// Voxel points
			////vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.voxelPoint);
			////vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &voxels.buffer, offsets);
			////vkCmdDraw(drawCmdBuffers[i], voxels.count, 1, 0, 0);

			drawUI(drawCmdBuffers[i]);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}
	// Build command buffer for rendering the scene to the offscreen frame buffer attachments
	void buildDeferredCommandBuffer()
	{
		if (offScreenCmdBuffer == VK_NULL_HANDLE)
		{
			offScreenCmdBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, false);
		}

		// Create a semaphore used to synchronize offscreen rendering and usage
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &offscreenSemaphore));

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		// Clear values for all attachments written in the fragment shader
		std::array<VkClearValue, 4> clearValues;
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[3].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = offScreenFrameBuf.renderPass;
		renderPassBeginInfo.framebuffer = offScreenFrameBuf.frameBuffer;
		renderPassBeginInfo.renderArea.extent.width = offScreenFrameBuf.width;
		renderPassBeginInfo.renderArea.extent.height = offScreenFrameBuf.height;
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		VK_CHECK_RESULT(vkBeginCommandBuffer(offScreenCmdBuffer, &cmdBufInfo));

		vkCmdBeginRenderPass(offScreenCmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport = vks::initializers::viewport((float)offScreenFrameBuf.width, (float)offScreenFrameBuf.height, 0.0f, 1.0f);
		vkCmdSetViewport(offScreenCmdBuffer, 0, 1, &viewport);

		VkRect2D scissor = vks::initializers::rect2D(offScreenFrameBuf.width, offScreenFrameBuf.height, 0, 0);
		vkCmdSetScissor(offScreenCmdBuffer, 0, 1, &scissor);

		//////DRAW////////////
		//////DRAW////////////
		VkDeviceSize offsets[1] = { 0 };

		vkCmdBindDescriptorSets(offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
		// Skysphere
		vkCmdBindPipeline(offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.skysphere);
		models.skysphere.draw(offScreenCmdBuffer);

		// Terrain
		vkCmdBindPipeline(offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.triangle);
		//vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertices.buffer, offsets);
		//vkCmdDraw(drawCmdBuffers[i], vertices.count, 1, 0, 0);
		for (int chunkIndex = 0; chunkIndex < CHUNK_COUNT; chunkIndex++) {
			if (frustumCheck((genVoxel::chunkIndex_to_pos(chunkIndex) + glm::vec3(0.5)) * 16.0f, CHUNK_RAIDUS)) {
				vkCmdBindVertexBuffers(offScreenCmdBuffer, 0, 1, &chunkListBuffer[chunkIndex]->vertices_per_chunk.buffer, offsets);
				vkCmdDraw(offScreenCmdBuffer, chunkListBuffer[chunkIndex]->vertices_per_chunk.count, 1, 0, 0);
				//vkCmdDrawIndirect(drawCmdBuffers[i], indirectCommandsBuffer.buffer, chunkIndex * sizeof(VkDrawIndirectCommand), 1, sizeof(VkDrawIndirectCommand));
			}
		}
		// Particle system (no index buffer)
		//vkCmdBindDescriptorSets(	offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets.particles, 0, nullptr);
		//vkCmdBindPipeline(			offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.particles);
		//vkCmdBindVertexBuffers(		offScreenCmdBuffer, 0, 1, &particles.buffer, offsets);
		//vkCmdDraw(					offScreenCmdBuffer, particle_count, 1, 0, 0);
		//////DRAW////////////
		//////DRAW////////////

		vkCmdEndRenderPass(offScreenCmdBuffer);

		VK_CHECK_RESULT(vkEndCommandBuffer(offScreenCmdBuffer));
	}

	void loadAssets()
	{
		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
		models.skysphere.loadFromFile(getAssetPath() + "models/sphere.gltf", vulkanDevice, queue, glTFLoadingFlags);
		textures.ground.colorMap.loadFromFile(getAssetPath() + "textures/stonefloor01_color_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
		textures.ground.normalMap.loadFromFile(getAssetPath() + "textures/stonefloor01_normal_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
		// Particles
		textures.particles.fire.loadFromFile(getAssetPath() + "textures/particle_fire.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
		// Create a custom sampler to be used with the particle textures
		// Create sampler
		VkSamplerCreateInfo samplerCreateInfo = vks::initializers::samplerCreateInfo();
		samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		// Different address mode
		samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
		samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
		samplerCreateInfo.mipLodBias = 0.0f;
		samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
		samplerCreateInfo.minLod = 0.0f;
		// Both particle textures have the same number of mip maps
		samplerCreateInfo.maxLod = float(textures.particles.fire.mipLevels);

		if (vulkanDevice->features.samplerAnisotropy)
		{
			// Enable anisotropic filtering
			samplerCreateInfo.maxAnisotropy = 8.0f;
			samplerCreateInfo.anisotropyEnable = VK_TRUE;
		}

		// Use a different border color (than the normal texture loader) for additive blending
		samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
		VK_CHECK_RESULT(vkCreateSampler(device, &samplerCreateInfo, nullptr, &textures.particles.sampler));
	}
	// Create a frame buffer attachment
	void createAttachment(
		VkFormat format,
		VkImageUsageFlagBits usage,
		FrameBufferAttachment* attachment)
	{
		VkImageAspectFlags aspectMask = 0;
		VkImageLayout imageLayout;

		attachment->format = format;

		if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
		{
			aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		}
		if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			if (format >= VK_FORMAT_D16_UNORM_S8_UINT)
				aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		}

		assert(aspectMask > 0);

		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = format;
		image.extent.width = offScreenFrameBuf.width;
		image.extent.height = offScreenFrameBuf.height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = usage | VK_IMAGE_USAGE_SAMPLED_BIT;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &attachment->image));
		vkGetImageMemoryRequirements(device, attachment->image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &attachment->mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, attachment->image, attachment->mem, 0));

		VkImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
		imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageView.format = format;
		imageView.subresourceRange = {};
		imageView.subresourceRange.aspectMask = aspectMask;
		imageView.subresourceRange.baseMipLevel = 0;
		imageView.subresourceRange.levelCount = 1;
		imageView.subresourceRange.baseArrayLayer = 0;
		imageView.subresourceRange.layerCount = 1;
		imageView.image = attachment->image;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageView, nullptr, &attachment->view));
	}
	// Prepare a new framebuffer and attachments for offscreen rendering (G-Buffer)
	void prepareOffscreenFramebuffer()
	{
		offScreenFrameBuf.width = FB_DIM;
		offScreenFrameBuf.height = FB_DIM;

		// Color attachments

		// (World space) Positions
		createAttachment(
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			&offScreenFrameBuf.position);

		// (World space) Normals
		createAttachment(
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			&offScreenFrameBuf.normal);

		// Albedo (color)
		createAttachment(
			VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			&offScreenFrameBuf.albedo);

		// Depth attachment

		// Find a suitable depth format
		VkFormat attDepthFormat;
		VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &attDepthFormat);
		assert(validDepthFormat);

		createAttachment(
			attDepthFormat,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			&offScreenFrameBuf.depth);

		// Set up separate renderpass with references to the color and depth attachments
		std::array<VkAttachmentDescription, 4> attachmentDescs = {};

		// Init attachment properties
		for (uint32_t i = 0; i < 4; ++i)
		{
			attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
			attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			if (i == 3)
			{
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			}
			else
			{
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}
		}

		// Formats
		attachmentDescs[0].format = offScreenFrameBuf.position.format;
		attachmentDescs[1].format = offScreenFrameBuf.normal.format;
		attachmentDescs[2].format = offScreenFrameBuf.albedo.format;
		attachmentDescs[3].format = offScreenFrameBuf.depth.format;

		std::vector<VkAttachmentReference> colorReferences;
		colorReferences.push_back({ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		colorReferences.push_back({ 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		colorReferences.push_back({ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 3;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.pColorAttachments = colorReferences.data();
		subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
		subpass.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for attachment layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.pAttachments = attachmentDescs.data();
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &offScreenFrameBuf.renderPass));

		std::array<VkImageView, 4> attachments;
		attachments[0] = offScreenFrameBuf.position.view;
		attachments[1] = offScreenFrameBuf.normal.view;
		attachments[2] = offScreenFrameBuf.albedo.view;
		attachments[3] = offScreenFrameBuf.depth.view;

		VkFramebufferCreateInfo fbufCreateInfo = {};
		fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbufCreateInfo.pNext = NULL;
		fbufCreateInfo.renderPass = offScreenFrameBuf.renderPass;
		fbufCreateInfo.pAttachments = attachments.data();
		fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		fbufCreateInfo.width = offScreenFrameBuf.width;
		fbufCreateInfo.height = offScreenFrameBuf.height;
		fbufCreateInfo.layers = 1;
		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offScreenFrameBuf.frameBuffer));

		// Create sampler to sample from the color attachments
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_NEAREST;
		sampler.minFilter = VK_FILTER_NEAREST;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &colorSampler));
	}
	float rnd(float range)
	{
		std::uniform_real_distribution<float> rndDist(0.0f, range);
		return rndDist(rndEngine);
	}
	void initParticle(Particle* particle, glm::vec3 emitterPos)
	{
		particle->vel = glm::vec4(0.0f, 8.0f + rnd(8.0f), 0.0f, 0.0f);
		particle->alpha = 0.2f + rnd(1.0f);
		particle->size = 0.7f + rnd(0.5f);
		particle->color = glm::vec4(1.0f, 0.5f, 0.5f, 1.0f);
		particle->rotation = rnd(2.0f * float(M_PI));
		particle->rotationSpeed = rnd(0.8f);

		// Get random sphere point
		float theta = rnd(2.0f * float(M_PI));
		float phi = rnd(float(M_PI)) - float(M_PI) / 2.0f;
		float r = rnd(FLAME_RADIUS);

		particle->pos.x = r * cos(theta) * cos(phi);
		particle->pos.y = r * sin(phi);
		particle->pos.z = r * sin(theta) * cos(phi);

		particle->pos -= glm::vec4(emitterPos, 0.0f);
	}
	void prepareParticles()
	{
		particleBuffer.resize(particle_count);
		emitter_positions.resize(particle_count, glm::vec3(0.0f));
		for (auto& particle : particleBuffer)
		{
			initParticle(&particle, glm::vec3(0.0f, 0.0f, 0.0f));
		}

		particles.size = particleBuffer.size() * sizeof(Particle);

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			particles.size,
			&particles.buffer,
			&particles.memory,
			particleBuffer.data()));

		// Map the memory and store the pointer for reuse
		VK_CHECK_RESULT(vkMapMemory(device, particles.memory, 0, particles.size, 0, &particles.mappedMemory));
	}


	void updateUniformBuffer()
	{
		// Voxel Terrain / Tessellation
		uniformData.projection = camera.matrices.perspective;
		uniformData.view = camera.matrices.view;
		if (!fixedFrustum)
		{
			//uniformData.cameraPos = glm::vec4(camera.position, 1.0f) * -1.0f;
			frustum.update(uniformData.projection * uniformData.view);
			memcpy(uniformData.frustumPlanes, frustum.planes.data(), sizeof(glm::vec4) * 6);
		}
		memcpy(uniformBuffer.mapped, &uniformData, sizeof(uniformData));

		// Particle system fire
		uboFire.projection = camera.matrices.perspective;
		uboFire.modelView = camera.matrices.view;
		uboFire.viewportDim = glm::vec2((float)width, (float)height);
		memcpy(uniformBuffers.fire.mapped, &uboFire, sizeof(uboFire));
	}
	// Update lights and parameters passed to the composition shaders
	void updateUniformBufferComposition()
	{
		// White
		uboComposition.lights[33].position = glm::vec4(-camera.position, 0.0f);
		uboComposition.lights[33].color = glm::vec3(1.0f, 0.95f, 0.84f);
		uboComposition.lights[33].radius = 20.0f;

		int index = 0;
		for (int i = 0; i < max_emitters_count; i++) {
			uboComposition.lights[i].position = glm::vec4(-emitter_positions[i], 0.0f);
			uboComposition.lights[i].color = colors[index];
			uboComposition.lights[i].radius = 25.0f;
			if (colors.size() - 1 == index) {
				index = 0;
			}
			else {
				index++;
			}
		}
		uboComposition.lights[32].position = glm::vec4(-emitter_positions[lastHitPositionIndex], 0.0f);
		uboComposition.lights[32].color = glm::vec3(0.03f, 0.87f, 1.0f);
		uboComposition.lights[32].radius = 30.0f;
		

		// Current view position
		uboComposition.viewPos = glm::vec4(-camera.position, 0.0f);// *glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

		uboComposition.debugDisplayTarget = debugDisplayTarget;

		memcpy(uniformBuffers.composition.mapped, &uboComposition, sizeof(uboComposition));
	}
	void updateParticles()
	{
		float particleTimer = frameTimer * 0.2f;
		int i = 0;
		for (auto& particle : particleBuffer)
		{
			particle.pos.y -= particle.vel.y * particleTimer * 5.0f;
			particle.alpha += particleTimer * 16.0f;
			particle.size -= particleTimer * 8.0f;
			particle.rotation += particleTimer * particle.rotationSpeed;
			// Transition particle state
			if (particle.alpha > 2.0f)
			{
				initParticle(&particle, emitter_positions[i]);
			}
			i++;
		}
		size_t size = particleBuffer.size() * sizeof(Particle);
		memcpy(particles.mappedMemory, particleBuffer.data(), size);
	}
	void setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6), // setLayoutBindings * number of descriptorSets
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9),
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 3);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}
	void setupDescriptorSetLayout()
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 
				VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, 0),
			// Binding 1 : Fragment shader image sampler (Fire texture/ plants texture array)
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			// Binding 2 : Fragment shader image sampler (ground texture)
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
			// Binding 3 : Albedo texture target
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
			// Binding 4 : Fragment shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));
	}
	void setupDescriptorSet()
	{
		std::vector<VkWriteDescriptorSet> writeDescriptorSets;
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

		// < Terrain >
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		writeDescriptorSets = {
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffer.descriptor),
			// Binding 1: Color map
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &textures.ground.colorMap.descriptor),
			// Binding 2: Normal map
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &textures.ground.normalMap.descriptor)
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

		// < Particles >
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.particles));
		// Image descriptor for the color map texture
		VkDescriptorImageInfo texDescriptorFire =
			vks::initializers::descriptorImageInfo(
				textures.particles.sampler,
				textures.particles.fire.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		writeDescriptorSets = {
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSets.particles, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.fire.descriptor),
			// Binding 1: Fire texture array
			vks::initializers::writeDescriptorSet(descriptorSets.particles, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &texDescriptorFire)
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

		// < Deferred composition >
		// Image descriptors for the offscreen color attachments
		VkDescriptorImageInfo texDescriptorPosition =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				offScreenFrameBuf.position.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		VkDescriptorImageInfo texDescriptorNormal =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				offScreenFrameBuf.normal.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		VkDescriptorImageInfo texDescriptorAlbedo =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				offScreenFrameBuf.albedo.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.gBuffers));
		writeDescriptorSets = {
			// Binding 1 : Position texture target
			vks::initializers::writeDescriptorSet(descriptorSets.gBuffers, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &texDescriptorPosition),
			// Binding 2 : Normals texture target
			vks::initializers::writeDescriptorSet(descriptorSets.gBuffers, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &texDescriptorNormal),
			// Binding 3 : Albedo texture target
			vks::initializers::writeDescriptorSet(descriptorSets.gBuffers, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &texDescriptorAlbedo),
			// Binding 4 : Fragment shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSets.gBuffers, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4, &uniformBuffers.composition.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}

	void preparePipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 4> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = 2;
		pipelineCI.pStages = shaderStages.data();

		// This example uses two different input states, one for the instanced part and one for non-instanced rendering
		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		std::vector<VkVertexInputBindingDescription> bindingDescriptions;
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
		#define VERTEX_BUFFER_BIND_ID 0
		// <Composition rendering>
		// deferred
		{
			VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
			pipelineCI.pVertexInputState = &emptyInputState;
			// Final fullscreen composition pass pipeline
			
			shaderStages[0] = loadShader(getShadersPath() + "deferred_marching_cube/deferred.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader(getShadersPath() + "deferred_marching_cube/deferred.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

			rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
			pipelineCI.stageCount = 2;
			// Empty vertex input state, vertices are generated by the vertex shader
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.composition));
		}
		// <Offscreen rendering>
		// Separate render pass
		pipelineCI.renderPass = offScreenFrameBuf.renderPass;
		// Blend attachment states required for all color attachments
		// This is important, as color write mask will otherwise be 0x0 and you
		// won't see anything rendered to the attachment
		std::array<VkPipelineColorBlendAttachmentState, 3> blendAttachmentStates = {
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE)
		};

		colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
		colorBlendState.pAttachments = blendAttachmentStates.data();

		bindingDescriptions = {
				vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, sizeof(vkglTF::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		attributeDescriptions = {
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),					// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3),	// Location 1: Normal
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 6),		// Location 2: Texture coordinates
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 3, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 8),	// Location 3: Color
		};
		// Skysphere
		{
			vertexInputState.pVertexBindingDescriptions = bindingDescriptions.data();
			vertexInputState.pVertexAttributeDescriptions = attributeDescriptions.data();
			vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
			vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
			pipelineCI.pVertexInputState = &vertexInputState;
			depthStencilState.depthWriteEnable = VK_FALSE;
			rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
			shaderStages[0] = loadShader(getShadersPath() + "deferred_marching_cube/skysphere.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader(getShadersPath() + "deferred_marching_cube/skysphere.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.skysphere));
		}
		bindingDescriptions = {
				vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		// Triangle
		{
			vertexInputState.pVertexBindingDescriptions = bindingDescriptions.data();
			vertexInputState.pVertexAttributeDescriptions = attributeDescriptions.data();
			vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
			vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
			pipelineCI.pVertexInputState = &vertexInputState;

			shaderStages[0] = loadShader(getShadersPath() + "deferred_marching_cube/triangle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader(getShadersPath() + "deferred_marching_cube/triangle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
			// Tessellation used
			//shaderStages[2] = loadShader(getShadersPath() + "deferred_marching_cube/triangle.tesc.spv", VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
			//shaderStages[3] = loadShader(getShadersPath() + "deferred_marching_cube/triangle.tese.spv", VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
			//pipelineCI.stageCount = 4;
			//VkPipelineTessellationStateCreateInfo tessellationState = vks::initializers::pipelineTessellationStateCreateInfo(3);
			//pipelineCI.pTessellationState = &tessellationState;
			//inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;

			depthStencilState.depthWriteEnable = VK_TRUE;
			rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
			
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.triangle));
		}
		// Reset tessellation state
		pipelineCI.pTessellationState = nullptr;
		pipelineCI.stageCount = 2;
		inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		// Voxel Point
		//{
		//	bindingDescriptions = {
		//		// Binding point 0: Mesh vertex layout description at per-vertex rate
		//		vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX),
		//	};
		//
		//	shaderStages[0] = loadShader(getShadersPath() + "deferred_marching_cube/voxelPoint.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		//	shaderStages[1] = loadShader(getShadersPath() + "deferred_marching_cube/voxelPoint.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		//	inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		//	depthStencilState.depthWriteEnable = VK_FALSE;
		//	rasterizationState.cullMode = VK_CULL_MODE_NONE;
		//	VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.voxelPoint));
		//}
		//pipelineCI.renderPass = renderPass;
		//colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		blendAttachmentStates = {
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_TRUE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_TRUE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_TRUE)
		};
		// Particle rendering pipeline
		{
			bindingDescriptions = {
				// Binding point 0: Mesh vertex layout description at per-vertex rate
				vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX),
			};
			attributeDescriptions = {
				vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, VK_FORMAT_R32G32B32A32_SFLOAT,	offsetof(Particle, pos)),	// Location 0: Position
				vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, VK_FORMAT_R32G32B32A32_SFLOAT,	offsetof(Particle, color)),	// Location 1: Color
				vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2, VK_FORMAT_R32_SFLOAT, offsetof(Particle, alpha)),			// Location 2: Alpha
				vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 3, VK_FORMAT_R32_SFLOAT, offsetof(Particle, size)),			// Location 3: Size
				vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 4, VK_FORMAT_R32_SFLOAT, offsetof(Particle, rotation)),		// Location 4: Rotation
			};
			vertexInputState.pVertexBindingDescriptions = bindingDescriptions.data();
			vertexInputState.pVertexAttributeDescriptions = attributeDescriptions.data();
			vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
			vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());

			pipelineCI.pVertexInputState = &vertexInputState;

			inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
			// Don t' write to depth buffer
			depthStencilState.depthWriteEnable = VK_FALSE;

			// Premulitplied alpha
			blendAttachmentStates[0].blendEnable = VK_TRUE;
			blendAttachmentStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
			blendAttachmentStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			blendAttachmentStates[0].colorBlendOp = VK_BLEND_OP_ADD;
			blendAttachmentStates[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			blendAttachmentStates[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			blendAttachmentStates[0].alphaBlendOp = VK_BLEND_OP_ADD;
			blendAttachmentStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

			blendAttachmentStates[1].blendEnable = VK_TRUE;
			blendAttachmentStates[1].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
			blendAttachmentStates[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			blendAttachmentStates[1].colorBlendOp = VK_BLEND_OP_ADD;
			blendAttachmentStates[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			blendAttachmentStates[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			blendAttachmentStates[1].alphaBlendOp = VK_BLEND_OP_ADD;
			blendAttachmentStates[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

			blendAttachmentStates[2].blendEnable = VK_TRUE;
			blendAttachmentStates[2].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
			blendAttachmentStates[2].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			blendAttachmentStates[2].colorBlendOp = VK_BLEND_OP_ADD;
			blendAttachmentStates[2].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			blendAttachmentStates[2].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			blendAttachmentStates[2].alphaBlendOp = VK_BLEND_OP_ADD;
			blendAttachmentStates[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

			colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
			colorBlendState.pAttachments = blendAttachmentStates.data();

			shaderStages[0] = loadShader(getShadersPath() + "deferred_marching_cube/particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader(getShadersPath() + "deferred_marching_cube/particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.particles));
		}
	}

	void updateIndirectData() {
		for (int chunkIndex = 0; chunkIndex < CHUNK_COUNT; chunkIndex++) {
			indirectCommands[chunkIndex].vertexCount = chunkListBuffer[chunkIndex]->vertices_per_chunk.count;
		}
		vks::Buffer stagingBuffer;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			indirectCommands.size() * sizeof(VkDrawIndirectCommand),
			indirectCommands.data()));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&indirectCommandsBuffer,
			stagingBuffer.size));
		vulkanDevice->copyBuffer(&stagingBuffer, &indirectCommandsBuffer, queue);
	}

	template <typename T>
	bool contains(const std::vector<T>& vec, T data) {
		auto it = std::find(vec.begin(), vec.end(), data);
		return it != vec.end();
	}
	
	void populate_chunk(Chunk* chunkBuffer, unsigned int index, std::vector<MarchingCube::Cell>& grid) {
		glm::vec3 cellOffsets[8] = {
			glm::vec3(0, 0, 0),	//0
			glm::vec3(0, 1, 0),	//1
			glm::vec3(1, 1, 0),	//2
			glm::vec3(1, 0, 0),	//3
			glm::vec3(0, 0, 1),	//4
			glm::vec3(0, 1, 1),	//5
			glm::vec3(1, 1, 1),	//6
			glm::vec3(1, 0, 1),	//7
		};
		for (int x = 0; x < CHUNK_DIMENSION - 1; x++) {
			for (int y = 0; y < CHUNK_DIMENSION - 1; y++) {
				for (int z = 0; z < CHUNK_DIMENSION - 1; z++) {
					MarchingCube::Cell cell;
					cell.val = 0;
					cell.p = glm::vec3(x, y, z) + genVoxel::chunkIndex_to_pos(index) * (float) (CHUNK_DIMENSION);
					uint8_t presentBit = 1;
					for (int i = 0; i < 8; i++) {
						if (chunkBuffer->voxel[ genVoxel::return_index(glm::vec3(x, y, z) + cellOffsets[i] )] & presentBit) {
							cell.val |= (1 << i);
						}
					}
					grid.push_back(cell);
				}
			}
		}
	}
	void populate_triangles_list(std::vector<MarchingCube::GRIDCELL>& grid, std::vector<MarchingCube::TRIANGLE>& tri_list) {
		for (std::vector<MarchingCube::GRIDCELL>::iterator it = grid.begin(); it != grid.end(); ++it) {
			Polygonise(*it, 0.5f, tri_list);
		}
	}
	void populate_triangles_list_chunk(std::vector<MarchingCube::Cell>& grid, std::vector<MarchingCube::TRIANGLE>& tri_list) {
		for (std::vector<MarchingCube::Cell>::iterator it = grid.begin(); it != grid.end(); ++it) {
			Polygonise_Cell(*it, tri_list);
		}
	}
	void gen_vertex_buffers(std::vector<MarchingCube::TRIANGLE>& tri_list, std::vector<Vertex>& vertexBuffer) {
		Vertex vertex;
		glm::vec3 tri_point;
		// in Vulkan, X -> -Z, Y -> X, Z -> -Y.
		for (int i = 0; i < tri_list.size(); i++) {
			tri_point = tri_list[i].p[0];
			glm::vec3 A = tri_point;
			tri_point = tri_list[i].p[1];
			glm::vec3 B = tri_point;
			tri_point = tri_list[i].p[2];
			glm::vec3 C = tri_point;
			vertex.normal = glm::normalize(glm::cross((B - A), (C - A)));
			vertex.tangent = glm::normalize((C - B));

			tri_point = tri_list[i].p[0];
			vertex.pos = glm::vec3(tri_point.x, tri_point.y, tri_point.z);
			vertexBuffer.push_back(vertex);

			tri_point = tri_list[i].p[1];
			vertex.pos = glm::vec3(tri_point.x, tri_point.y, tri_point.z);
			vertexBuffer.push_back(vertex);

			tri_point = tri_list[i].p[2];
			vertex.pos = glm::vec3(tri_point.x, tri_point.y, tri_point.z);
			vertexBuffer.push_back(vertex);
		}
	}
	void polygonizeVoxelsChunks(std::unordered_set<int>& damagedChunkIndices) {
		for (const int& number : damagedChunkIndices) {
			polygonizeVoxels(number);
		}
	}
	void polygonizeVoxels(int chunkIndex) {
		// remove old per-Chunk data
		chunkListBuffer[chunkIndex]->grid_of_cells_per_chunk.clear();
		total_terrain_triangle_count -= chunkListBuffer[chunkIndex]->tri_list_per_chunk.size(); // remove the old triangles
		chunkListBuffer[chunkIndex]->tri_list_per_chunk.clear();
		chunkListBuffer[chunkIndex]->vertexBuffer_per_chunk.clear();
		// make new per-Chunk data
		populate_chunk(chunkListBuffer[chunkIndex], chunkIndex, chunkListBuffer[chunkIndex]->grid_of_cells_per_chunk);
		populate_triangles_list_chunk(chunkListBuffer[ chunkIndex ]->grid_of_cells_per_chunk, chunkListBuffer[ chunkIndex ]->tri_list_per_chunk);
		total_terrain_triangle_count += chunkListBuffer[chunkIndex]->tri_list_per_chunk.size();
		gen_vertex_buffers(chunkListBuffer[ chunkIndex ]->tri_list_per_chunk, chunkListBuffer[ chunkIndex ]->vertexBuffer_per_chunk);
		// same thing with Init.
		chunkListBuffer[chunkIndex]->vertices_per_chunk.count = static_cast<uint32_t>(chunkListBuffer[chunkIndex]->vertexBuffer_per_chunk.size());
		uint32_t vertexBufferSize = chunkListBuffer[chunkIndex]->vertices_per_chunk.count * sizeof(Vertex);
		struct StagingBuffer {
			VkBuffer buffer;
			VkDeviceMemory memory;
		} vertexStaging;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			vertexBufferSize,
			&vertexStaging.buffer,
			&vertexStaging.memory,
			chunkListBuffer[ chunkIndex ]->vertexBuffer_per_chunk.data()));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			vertexBufferSize,
			&chunkListBuffer[chunkIndex]->vertices_per_chunk.buffer,
			&chunkListBuffer[chunkIndex]->vertices_per_chunk.memory,
			nullptr));
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, chunkListBuffer[chunkIndex]->vertices_per_chunk.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		vkDestroyBuffer(vulkanDevice->logicalDevice, vertexStaging.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, vertexStaging.memory, nullptr);
	}
	void polygonizeVoxelsInit() {
		for (int i = 0; i < CHUNK_COUNT; i++) {
			chunkListBuffer[i] = new Chunk();
			genVoxel::Fill_Chunk(chunkListBuffer[i]);
		}
		for (int i = 0; i < CHUNK_COUNT; i++) {
			populate_chunk(chunkListBuffer[i], i, chunkListBuffer[i]->grid_of_cells_per_chunk);
		}
		for (int i = 0; i < CHUNK_COUNT; i++) {
			populate_triangles_list_chunk(chunkListBuffer[i]->grid_of_cells_per_chunk, chunkListBuffer[i]->tri_list_per_chunk);
		}
		total_terrain_triangle_count = 0;
		for (int i = 0; i < CHUNK_COUNT; i++) {
			total_terrain_triangle_count += chunkListBuffer[i]->tri_list_per_chunk.size();
			gen_vertex_buffers(chunkListBuffer[i]->tri_list_per_chunk, chunkListBuffer[i]->vertexBuffer_per_chunk);
		}
		for (int i = 0; i < CHUNK_COUNT; i++) {
			chunkListBuffer[i]->vertices_per_chunk.count = static_cast<uint32_t>(chunkListBuffer[i]->vertexBuffer_per_chunk.size());
			uint32_t vertexBufferSize = chunkListBuffer[i]->vertices_per_chunk.count * sizeof(Vertex);
			struct StagingBuffer {
				VkBuffer buffer;
				VkDeviceMemory memory;
			} vertexStaging, indexStaging, voxelStaging;
			if (vertexBufferSize) {
				VK_CHECK_RESULT(vulkanDevice->createBuffer(
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					vertexBufferSize,
					&vertexStaging.buffer,
					&vertexStaging.memory,
					chunkListBuffer[i]->vertexBuffer_per_chunk.data()));
				VK_CHECK_RESULT(vulkanDevice->createBuffer(
					VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					vertexBufferSize,
					&chunkListBuffer[i]->vertices_per_chunk.buffer,
					&chunkListBuffer[i]->vertices_per_chunk.memory,
					nullptr));
				// Put buffer region copies into command buffer
				VkBufferCopy copyRegion = {};
				copyRegion.size = vertexBufferSize;
				// Copy from staging buffers
				VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
				vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, chunkListBuffer[i]->vertices_per_chunk.buffer, 1, &copyRegion);
				vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
				// Note: Staging buffer must not be deleted before the copies have been submitted and executed
				vkDestroyBuffer(vulkanDevice->logicalDevice, vertexStaging.buffer, nullptr);
				vkFreeMemory(vulkanDevice->logicalDevice, vertexStaging.memory, nullptr);
			}
		}
	}
	void polygonizeVoxelsInitMultiThread(unsigned int threadID) {
		int Lower_Chunk_Index = (CHUNK_COUNT / numThreads) * threadID;
		int Upper_Chunk_Index = (CHUNK_COUNT / numThreads) * (threadID + 1);
		// Generate volumetric data
		//genVoxel::Cube(glm::vec3(0, 0, 0), 1, voxelBuffer);
		for (int i = Lower_Chunk_Index; i < Upper_Chunk_Index; i++) {
			chunkListBuffer[i] = new Chunk();
			genVoxel::Fill_Chunk(chunkListBuffer[i]);
		}
		// Loop over a block of space. Based on the volumetric data, populate Grid cells with values 
		//std::vector<MarchingCube::GRIDCELL> grid;
		for (int i = Lower_Chunk_Index; i < Upper_Chunk_Index; i++) {
			populate_chunk(chunkListBuffer[i], i, chunkListBuffer[i]->grid_of_cells_per_chunk);
		}
		// Run Marching Cube algorithm on each Grid cell, which returns a list of triangles based on the cells' value
		for (int i = Lower_Chunk_Index; i < Upper_Chunk_Index; i++) {
			populate_triangles_list_chunk(chunkListBuffer[i]->grid_of_cells_per_chunk, chunkListBuffer[i]->tri_list_per_chunk);
		}
		// Using the triangles list, Generate vertex and index buffers
		//std::vector<uint32_t> indexBuffer;
		total_terrain_triangle_count = 0;
		for (int i = Lower_Chunk_Index; i < Upper_Chunk_Index; i++) {
			total_terrain_triangle_count += chunkListBuffer[i]->tri_list_per_chunk.size();
			gen_vertex_buffers(chunkListBuffer[i]->tri_list_per_chunk, chunkListBuffer[i]->vertexBuffer_per_chunk);
		}
		// Static data like vertex and index buffer should be stored on the device memory for optimal (and fastest) access by the GPU
		//
		// To achieve this we use so-called "staging buffers" :
		// - Create a buffer that's visible to the host (and can be mapped)
		// - Copy the data to this buffer
		// - Create another buffer that's local on the device (VRAM) with the same size
		// - Copy the data from the host to the device using a command buffer
		// - Delete the host visible (staging) buffer
		// - Use the device local buffers for rendering
		//
		// Note: On unified memory architectures where host (CPU) and GPU share the same memory, staging is not necessary
		// To keep this sample easy to follow, there is no check for that in place
		for (int i = Lower_Chunk_Index; i < Upper_Chunk_Index; i++) {
			chunkListBuffer[i]->vertices_per_chunk.count = static_cast<uint32_t>(chunkListBuffer[i]->vertexBuffer_per_chunk.size());
			uint32_t vertexBufferSize = chunkListBuffer[i]->vertices_per_chunk.count * sizeof(Vertex);

			//voxels.count = static_cast<uint32_t>(voxelBuffer.size());
			//uint32_t voxelBufferSize = voxels.count * sizeof(glm::vec3);
			struct StagingBuffer {
				VkBuffer buffer;
				VkDeviceMemory memory;
			} vertexStaging, indexStaging, voxelStaging;

			
			if (vertexBufferSize) {
				// Voxel buffer
				//VK_CHECK_RESULT(vulkanDevice->createBuffer(
				//	VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				//	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				//	voxelBufferSize,
				//	&voxelStaging.buffer,
				//	&voxelStaging.memory,
				//	voxelBuffer.data()));
				// Vertex buffer
				VK_CHECK_RESULT(vulkanDevice->createBuffer(
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					vertexBufferSize,
					&vertexStaging.buffer,
					&vertexStaging.memory,
					chunkListBuffer[i]->vertexBuffer_per_chunk.data()));
				// Create vulkanDevice local buffers
				// // Voxel buffer
				//VK_CHECK_RESULT(vulkanDevice->createBuffer(
				//	VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				//	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				//	voxelBufferSize,
				//	&voxels.buffer,
				//	&voxels.memory,
				//	nullptr));
				// Vertex buffer
				VK_CHECK_RESULT(vulkanDevice->createBuffer(
					VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					vertexBufferSize,
					&chunkListBuffer[i]->vertices_per_chunk.buffer,
					&chunkListBuffer[i]->vertices_per_chunk.memory,
					nullptr));

				// Put buffer region copies into command buffer
				VkBufferCopy copyRegion = {};
				copyRegion.size = vertexBufferSize;

				mutex_lock.lock(); // VkCommandPool cannot be access simultaneously
				// Copy from staging buffers
				VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
				
				vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, chunkListBuffer[i]->vertices_per_chunk.buffer, 1, &copyRegion);

				//copyRegion.size = voxelBufferSize;
				//vkCmdCopyBuffer(copyCmd, voxelStaging.buffer, voxels.buffer, 1, &copyRegion);

				vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

				mutex_lock.unlock();

				// Note: Staging buffer must not be deleted before the copies have been submitted and executed
				vkDestroyBuffer(vulkanDevice->logicalDevice, vertexStaging.buffer, nullptr);
				vkFreeMemory(vulkanDevice->logicalDevice, vertexStaging.memory, nullptr);
				//vkDestroyBuffer(vulkanDevice->logicalDevice, voxelStaging.buffer, nullptr);
				//vkFreeMemory(vulkanDevice->logicalDevice, voxelStaging.memory, nullptr);
			}
		}
	}
	void createVertexBuffer()
	{
		polygonizeVoxelsInit();
	}
	void createVertexBufferMultiThread()
	{
		// Setup vertices
		// multithread
		std::vector<std::thread> threads;
		for (int threadID = 0; threadID < numThreads; threadID++) {
			threads.emplace_back(&VulkanExample::polygonizeVoxelsInitMultiThread, this, threadID); /* Resource->Buffer */
		}
		// Wait for all threads to finish
		for (auto& thread : threads) {
			thread.join();
		}
	}
	void prepareUniformBuffers()
	{
		// Offscreen vertex shader / tessellation shader stages
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffer,
			sizeof(uniformData)));
		// Particle shader
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.fire,
			sizeof(uboFire)));
		// Deferred fragment shader
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.composition,
			sizeof(uboComposition)));

		// Map persistent
		VK_CHECK_RESULT(uniformBuffer.map());
		VK_CHECK_RESULT(uniformBuffers.fire.map());
		VK_CHECK_RESULT(uniformBuffers.composition.map());
		updateUniformBuffer();
		updateUniformBufferComposition();
	}
	void prepare()
	{
		// Order doesn't matter when placed on same line
		VulkanExampleBase::prepare();
		loadAssets(); prepareOffscreenFramebuffer(); prepareParticles();
		createVertexBufferMultiThread(); // Multithreaded version only becomes faster at 32x32x32 World Dimension
		prepareUniformBuffers();
		setupDescriptorPool(); setupDescriptorSetLayout();

		setupDescriptorSet(); // Buffer -> Descriptor
		preparePipelines();
		buildCommandBuffers();
		buildDeferredCommandBuffer();
		prepared = true;
	}
	void draw()
	{
		VulkanExampleBase::prepareFrame();
		// Offscreen rendering
		// Wait for swap chain presentation to finish
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		// Signal ready with offscreen semaphore
		submitInfo.pSignalSemaphores = &offscreenSemaphore;
		// Submit work
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &offScreenCmdBuffer;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		// Scene rendering
		// Wait for offscreen semaphore
		submitInfo.pWaitSemaphores = &offscreenSemaphore;
		// Signal ready with render complete semaphore
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;
		// Submit work
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
	}

	std::chrono::steady_clock::time_point lastTime_build_CMD_BUFFER;
	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused)
		{
			updateParticles();
			updateUniformBufferComposition();
		}
		if (camera.updated)
		{
			updateUniformBuffer();
		}
		// only happens every 1-2 second
		std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime_build_CMD_BUFFER).count() >= 1000) {
			buildDeferredCommandBuffer(); // Frustum culling
		}
	}
	
	virtual void action()
	{
		std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime_build_CMD_BUFFER).count() >= 150) {
			std::unordered_set<int> damagedChunkIndices;
			glm::vec3 rayHitLocation;
			if (genVoxel::RayCast(camera.position, camera.getCameraFront(), chunkListBuffer, emitter_positions, &rayHitLocation)) {
				genVoxel::Remove_Voxel(rayHitLocation, chunkListBuffer, damagedChunkIndices);

				if (lastHitPositionIndex <= max_emitters_count - 1) {
					lastHitPositionIndex++;
					emitter_positions[lastHitPositionIndex] = rayHitLocation;
				}
				else {
					emitter_positions[0] = rayHitLocation;
					lastHitPositionIndex = 0;
				}
				polygonizeVoxelsChunks(damagedChunkIndices);
				buildDeferredCommandBuffer();
			}
			lastTime_build_CMD_BUFFER = currentTime;
		}
	}
	virtual void no_action() {
		// Particle & Light location
		for (int i = 0; i < lastHit_particle_count; i++) {
			emitter_positions[i + max_emitters_count] = emitter_positions[lastHitPositionIndex];
		}
	}
	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (!vulkanDevice->features.multiDrawIndirect) {
			if (overlay->header("Info")) {
				overlay->text("multiDrawIndirect not supported");
			}
		}
		if (overlay->header("Settings")) {
			overlay->checkBox("Freeze frustum", &fixedFrustum);
		}
		if (overlay->header("Statistics")) {
			//overlay->text("Visible objects: %d", indirectStats.drawCount);
			// in Vulkan, X -> -Z, Y -> X, Z -> -Y.
			overlay->text("My Position: <X : %.1f, Y : %.1f, Z : %.1f>", camera.position.x, camera.position.y, camera.position.z);
			overlay->text("RayHit: <X : %.1f, Y : %.1f, Z : %.1f>", emitter_positions[lastHitPositionIndex].x, emitter_positions[lastHitPositionIndex].y, emitter_positions[lastHitPositionIndex].z);
			//overlay->text("Camera Front: <X : %.1f, Y : %.1f, Z : %.1f>", camera.getCameraFront().x, camera.getCameraFront().y, camera.getCameraFront().z);
			int voxel_index_within_chunk = genVoxel::pos_to_voxelIndex(camera.position);
			overlay->text("voxel_index_within_chunk: %d", voxel_index_within_chunk);
			glm::vec3 voxel_pos_within_chunk = genVoxel::voxelIndex_to_pos(voxel_index_within_chunk);
			overlay->text("voxel_pos_within_chunk: <X : %.1f, Y : %.1f, Z : %.1f>", voxel_pos_within_chunk.x, voxel_pos_within_chunk.y, voxel_pos_within_chunk.z);
			overlay->text("Voxel: <X : %d, Y : %d, Z : %d>", ((int)camera.position.x) % (CHUNK_DIMENSION), ((int)camera.position.y) % (CHUNK_DIMENSION), ((int)camera.position.z) % (CHUNK_DIMENSION));
			overlay->text("Chunk: <X : %d, Y : %d, Z : %d>", ((int)camera.position.x) / (CHUNK_DIMENSION), ((int)camera.position.y) / (CHUNK_DIMENSION), ((int)camera.position.z) / (CHUNK_DIMENSION));
			overlay->text("Chunk Index: %d", genVoxel::pos_to_chunkIndex(camera.position));
			glm::vec3 chunk_pos = genVoxel::chunkIndex_to_pos(genVoxel::pos_to_chunkIndex(camera.position));
			overlay->text("Chunk Pos: <X : %.1f, Y : %.1f, Z : %.1f>", chunk_pos.x , chunk_pos.y, chunk_pos.z);
			//overlay->text("Movement Speed: %.1f", camera.movementSpeed);
			//overlay->text("sizeof(chunkListBuffer): %d", debugDisplayTarget);
		}
		if (overlay->comboBox("Display", &debugDisplayTarget, { "Final composition", "Position", "Normals", "Albedo", "Specular" }))
		{
			updateUniformBufferComposition();
		}
		//if (overlay->header("Frustum")) {
		//	//enum side { LEFT = 0, RIGHT = 1, TOP = 2, BOTTOM = 3, BACK = 4, FRONT = 5 };
		//	overlay->text("LEFT: <X : %.1f, Y : %.1f, Z : %.1f, W : %.1f>", frustum.planes.data()[0].x, frustum.planes.data()[0].y, frustum.planes.data()[0].z, frustum.planes.data()[0].w);
		//	overlay->text("RIGHT: <X : %.1f, Y : %.1f, Z : %.1f, W : %.1f>", frustum.planes.data()[1].x, frustum.planes.data()[1].y, frustum.planes.data()[1].z, frustum.planes.data()[1].w);
		//	overlay->text("TOP: <X : %.1f, Y : %.1f, Z : %.1f, W : %.1f>", frustum.planes.data()[2].x, frustum.planes.data()[2].y, frustum.planes.data()[2].z, frustum.planes.data()[2].w);
		//	overlay->text("BOTTOM: <X : %.1f, Y : %.1f, Z : %.1f, W : %.1f>", frustum.planes.data()[3].x, frustum.planes.data()[3].y, frustum.planes.data()[3].z, frustum.planes.data()[3].w);
		//	overlay->text("BACK/Near: <X : %.1f, Y : %.1f, Z : %.1f, W : %.1f>", frustum.planes.data()[4].x, frustum.planes.data()[4].y, frustum.planes.data()[4].z, frustum.planes.data()[4].w);
		//	overlay->text("FRONT/Far: <X : %.1f, Y : %.1f, Z : %.1f, W : %.1f>", frustum.planes.data()[5].x, frustum.planes.data()[5].y, frustum.planes.data()[5].z, frustum.planes.data()[5].w);
		//}
		overlay->text("CommandBuffer build count: %d", cmdBufferBuildCount);
	}
};

VULKAN_EXAMPLE_MAIN()
