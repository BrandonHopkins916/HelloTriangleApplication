#pragma once
#include <algorithm>
#include <array>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <vector>
#include "helpers.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.hpp>
#include <glm/gtc/matrix_transform.hpp>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

// Enable the WSI extensions
#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

// REQUIRED only for GLFW CreateWindowSurface.
#define GLFW_INCLUDE_VULKAN

// Tell SDL not to mess with main()
#define SDL_MAIN_HANDLED

// Image
#define STB_IMAGE_IMPLEMENTATION
#include "vcpkg_installed/x64-windows/include/stb_image.h"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<char const*> validationLayers =
{
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

constexpr bool disableValidationLayer = false;

using namespace std;

std::vector<const char*> requiredDeviceExtension = {
vk::KHRSwapchainExtensionName,
vk::KHRSpirv14ExtensionName,
vk::KHRSynchronization2ExtensionName,
vk::KHRCreateRenderpass2ExtensionName
};

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;
	glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        return
        {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat,	offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
			vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat,	offsetof(Vertex, texCoord))
        };
    }

};

struct UniformBufferObject
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {10.0f, 1.0f}}
};

const std::vector<uint16_t> indices =
{
    0, 1, 2, 2, 3, 0
};


class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:

	GLFWwindow* window									= nullptr;

	vk::raii::Context context;
	vk::raii::Instance instance							= nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger		= nullptr;
	vk::raii::SurfaceKHR surface						= nullptr;

	vk::raii::PhysicalDevice physicalDevice				= nullptr;
	vk::raii::Device device								= nullptr;
	uint32_t queueIndex									= ~0;

	vk::raii::Queue graphicsQueue						= nullptr;
	vk::raii::Queue presentQueue						= nullptr;

	vk::raii::Queue	queue								= nullptr;
	vk::raii::SwapchainKHR swapChain					= nullptr;
	std::vector<vk::Image> swapChainImages;
	vk::Format swapChainImageFormat						= vk::Format::eUndefined;
	vk::Extent2D										swapChainExtent;
	std::vector<vk::raii::ImageView>					swapChainImageViews;

	vk::raii::PipelineLayout pipelineLayout				= nullptr;
	vk::raii::DescriptorSetLayout descriptorSetLayout	= nullptr;
	vk::raii::Pipeline graphicsPipeline					= nullptr;

	vk::raii::CommandPool commandPool					= nullptr;
	std::vector<vk::raii::CommandBuffer>				commandBuffers;
	uint32_t graphicsIndex								= 0;

	std::vector<vk::raii::Semaphore>					presentCompleteSemaphore;
	std::vector<vk::raii::Semaphore>					renderFinishedSemaphore;
	std::vector<vk::raii::Fence>						inFlightFences;
	uint32_t semaphoreIndex								= 0;
	uint32_t currentFrame								= 0;
	bool framebufferResized								= false;

    vk::raii::Image        textureImage					= nullptr;
    vk::raii::DeviceMemory textureImageMemory			= nullptr;
    vk::raii::ImageView    textureImageView				= nullptr;
    vk::raii::Sampler      textureSampler				= nullptr;

	vk::raii::Buffer vertexBuffer						= nullptr;
	vk::raii::DeviceMemory vertexBufferMemory			= nullptr;
	vk::raii::Buffer indexBuffer						= nullptr;
	vk::raii::DeviceMemory indexBufferMemory			= nullptr;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

	vk::raii::DescriptorPool descriptorPool				= nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	// Initialize Vulkan
	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}

		device.waitIdle();
	}

    void cleanupSwapChain()
    {
        swapChainImageViews.clear();
		swapChain = nullptr;
    }

	void cleanup()
	{
		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		device.waitIdle();

		cleanupSwapChain();
		createSwapChain();
		createImageViews();
	}

	void createInstance()
	{
		vk::ApplicationInfo					appInfo;
		appInfo.pApplicationName			= "Hello Triangle";
		appInfo.applicationVersion			= VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName					= "No Engine";
		appInfo.engineVersion				= VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion					= vk::ApiVersion14;

		// Get the required layers
		std::vector<char const*> requiredLayers;
		if (enableValidationLayers)
		{
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		// Check if the required layers are supported by the Vulkan implementation.
		auto layerProperties = context.enumerateInstanceLayerProperties();
		for (auto const& requiredLayer : requiredLayers)
		{
			if (std::ranges::none_of(layerProperties,
				[requiredLayer](auto const& layerProperty)
				{ return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
			{
				throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
			}
		}

		// Get the required extensions.
		auto requiredExtensions = getRequiredExtensions(context, enableValidationLayers);

		// Check if the required extensions are supported by the Vulkan implementation.
		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		for (auto const& requiredExtension : requiredExtensions)
		{
			if (std::ranges::none_of(extensionProperties,
				[requiredExtension](auto const& extensionProperty)
				{ return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
			{
				throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
			}
		}

		vk::InstanceCreateInfo					createInfo;
		createInfo.pApplicationInfo				= &appInfo;
		createInfo.enabledLayerCount			= static_cast<uint32_t>(requiredLayers.size());
		createInfo.ppEnabledLayerNames			= requiredLayers.data();
		createInfo.enabledExtensionCount		= static_cast<uint32_t>(requiredExtensions.size());
		createInfo.ppEnabledExtensionNames		= requiredExtensions.data();

		instance = vk::raii::Instance(context, createInfo);
	};

	void createSurface()
	{
		VkSurfaceKHR _surface;

		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		{
			throw std::runtime_error("Failed to Create Window Surface!");
		}

		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	void pickPhysicalDevice()
	{
		std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
		const auto                            devIter = std::ranges::find_if(
			devices,
			[&](auto const& device)
			{
				// Check if the device supports the Vulkan 1.3 API version
				bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

				// Check if any of the queue families support graphics operations
				auto queueFamilies = device.getQueueFamilyProperties();
				bool supportsGraphics =
					std::ranges::any_of(queueFamilies, [](auto const& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

				// Check if all required device extensions are available
				auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
				bool supportsAllRequiredExtensions =
					std::ranges::all_of(requiredDeviceExtension,
						[&availableDeviceExtensions](auto const& requiredDeviceExtension)
						{
							return std::ranges::any_of(availableDeviceExtensions,
								[requiredDeviceExtension](auto const& availableDeviceExtension)
								{ return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
						});

				auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
				bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
					features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

				return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
			});
		if (devIter != devices.end())
		{
			physicalDevice = *devIter;
		}
		else
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}

	}

	void createLogicalDevice()
	{
		// find the index of the first queue family that supports graphics
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		// get the first index into queueFamilyProperties which supports graphics
		auto graphicsQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties, [](auto const& qfp)
			{ return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); });
		assert(graphicsQueueFamilyProperty != queueFamilyProperties.end() && "No graphics queue family found!");

		auto graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));

		// Determine a queueFamilyIndex that supports present
		// First check if the graphicsIndex is good enough
		auto presentIndex = physicalDevice.getSurfaceSupportKHR(graphicsIndex, *surface) ? graphicsIndex : static_cast<uint32_t>(queueFamilyProperties.size());

		if (presentIndex == queueFamilyProperties.size())
		{
			// the graphicsIndex doesn't support present -> look for another family index that supports both
			// graphics and present
			for (size_t i = 0; i < queueFamilyProperties.size(); i++)
			{
				if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
					physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
				{
					graphicsIndex = static_cast<uint32_t>(i);
					presentIndex = graphicsIndex;
					break;
				}
			}
			if (presentIndex == queueFamilyProperties.size())
			{
				// there's nothing like a single family index that supports both graphics and present -> look for another
				// family index that supports present
				for (size_t i = 0; i < queueFamilyProperties.size(); i++)
				{
					if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
					{
						presentIndex = static_cast<uint32_t>(i);
						break;
					}
				}
			}
		}

		if ((graphicsIndex == queueFamilyProperties.size()) || (presentIndex == queueFamilyProperties.size()))
		{
			throw std::runtime_error("Could not find a queue for graphics or present -> terminating");
		}

		// Query for Vulkan 1.3 features
		auto features = physicalDevice.getFeatures2();
		vk::PhysicalDeviceVulkan13Features vulkan13Features;
		vulkan13Features.sType = vk::StructureType::ePhysicalDeviceVulkan13Features;
		vulkan13Features.dynamicRendering = vk::True;

		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures;
		extendedDynamicStateFeatures.sType = vk::StructureType::ePhysicalDeviceExtendedDynamicStateFeaturesEXT;
		extendedDynamicStateFeatures.extendedDynamicState = vk::True;

		vulkan13Features.pNext = &extendedDynamicStateFeatures;
		features.pNext = &vulkan13Features;

		vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain(
			vk::PhysicalDeviceFeatures2{},
			vulkan13Features,
			extendedDynamicStateFeatures
		);

		// Create a Device
		float                     queuePriority = 0.0f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo;
		deviceQueueCreateInfo.queueFamilyIndex = graphicsIndex;
		deviceQueueCreateInfo.queueCount = 1;
		deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

		vk::DeviceCreateInfo deviceCreateInfo;
		deviceCreateInfo.pNext = &features;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
		deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size());
		deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();

		//vk::PhysicalDeviceFeatures deviceFeatures;
		//deviceFeatures.samplerAnisotropy = vk::True;

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
		presentQueue = vk::raii::Queue(device, presentIndex, 0);
	}

	void createSwapChain()
	{
		auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
		swapChainImageFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
		swapChainExtent = chooseSwapExtent(window, surfaceCapabilities);
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;

		vk::SwapchainCreateInfoKHR swapChainCreateInfo;
		swapChainCreateInfo.surface = surface;
		swapChainCreateInfo.minImageCount = minImageCount;
		swapChainCreateInfo.imageFormat = swapChainImageFormat;
		swapChainCreateInfo.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
		swapChainCreateInfo.imageExtent = swapChainExtent;
		swapChainCreateInfo.imageArrayLayers = 1;
		swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
		swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
		swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
		swapChainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
		swapChainCreateInfo.presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface));
		swapChainCreateInfo.clipped = true;

		swapChain		= vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	void createImageViews()
	{
		assert(swapChainImageViews.empty());

		vk::ImageViewCreateInfo					imageViewCreateInfo;
		imageViewCreateInfo.viewType			= vk::ImageViewType::e2D;
		imageViewCreateInfo.format				= swapChainImageFormat;
		imageViewCreateInfo.subresourceRange	= { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

        for (auto image : swapChainImages)
        {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(device, imageViewCreateInfo);
        }
	}

	void createDescriptorSetLayout()
	{
		std::array bindings =
		{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
			vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.flags = {};
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

        /*vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutCreateInfo layoutInfo;
        layoutInfo.flags = {};
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;
        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);*/
	}

	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule( readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
		vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
		vertShaderStageInfo.module = shaderModule;
		vertShaderStageInfo.pName = "vertMain";

		vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
		fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
		fragShaderStageInfo.module = shaderModule;
		fragShaderStageInfo.pName = "fragMain";

		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
		inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		vk::PipelineViewportStateCreateInfo viewportState;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		vk::PipelineRasterizationStateCreateInfo rasterizer;
		rasterizer.depthClampEnable = vk::False;
		rasterizer.rasterizerDiscardEnable = vk::False;
		rasterizer.polygonMode = vk::PolygonMode::eFill;
		rasterizer.cullMode = vk::CullModeFlagBits::eBack;
		rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
		rasterizer.depthBiasEnable = vk::False;
		rasterizer.depthBiasSlopeFactor = 1.0f;
		rasterizer.lineWidth = 1.0f;

		vk::PipelineMultisampleStateCreateInfo multisampling;
		multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
		multisampling.sampleShadingEnable = vk::False;

		vk::PipelineColorBlendAttachmentState colorBlendAttachment;
		colorBlendAttachment.blendEnable = vk::False;
		colorBlendAttachment.colorWriteMask = (vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

		vk::PipelineColorBlendStateCreateInfo colorBlending;
		colorBlending.logicOpEnable = vk::False;
		colorBlending.logicOp = vk::LogicOp::eCopy;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		std::vector dynamicStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};

		vk::PipelineDynamicStateCreateInfo dynamicState;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo;
		pipelineRenderingCreateInfo.colorAttachmentCount = 1;
		pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapChainImageFormat;

		vk::GraphicsPipelineCreateInfo pipelineInfo;
		pipelineInfo.pNext = &pipelineRenderingCreateInfo;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pTessellationState = nullptr;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = nullptr;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
	}

	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo;
		poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		poolInfo.queueFamilyIndex = graphicsIndex;

		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	void createTextureImage()
	{
		int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) 
		{
            throw std::runtime_error("Failed to load texture image!");
        }

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, 
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, imageSize);
        stagingBufferMemory.unmapMemory();

        stbi_image_free(pixels);

        createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, 
			vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
	}

    void createTextureImageView()
    {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
    }

    void createTextureSampler() 
	{
		vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

		vk::SamplerCreateInfo samplerInfo;
		samplerInfo.flags = {};
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.anisotropyEnable = 1;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.compareEnable = vk::False;
		samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
		samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
		samplerInfo.unnormalizedCoordinates = vk::False;

		textureSampler = vk::raii::Sampler(device, samplerInfo);

    }

	void createImage(uint32_t width,
		uint32_t height,
		vk::Format format,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		vk::raii::Image& image,
		vk::raii::DeviceMemory& imageMemory)
	{
		vk::ImageCreateInfo imageInfo{};
		imageInfo.flags = {};
		imageInfo.imageType = vk::ImageType::e2D;
		imageInfo.format = format;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = vk::SampleCountFlagBits::e1;
		imageInfo.tiling = tiling;
		imageInfo.usage = usage;
		imageInfo.sharingMode = vk::SharingMode::eExclusive;
		imageInfo.queueFamilyIndexCount = 0;

		image = vk::raii::Image(device, imageInfo);

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);
        imageMemory = vk::raii::DeviceMemory(device, allocInfo);
        image.bindMemory(imageMemory, 0);
	}

	vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags)
	{
		vk::ImageViewCreateInfo viewInfo;
		viewInfo.image				= image;
		viewInfo.viewType			= vk::ImageViewType::e2D;
		viewInfo.format				= format;
		viewInfo.subresourceRange	= { aspectFlags, 0, 1, 0, 1 };

		return vk::raii::ImageView(device, viewInfo);
	}

	void createVertexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

	}

	void createIndexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), (size_t)bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
	}

	void createUniformBuffers()
	{
		uniformBuffers.clear();
		uniformBuffersMemory.clear();
		uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
		{
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
	}

	void createDescriptorPool()
	{
        std::array poolSize
        {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT)
        };

        vk::DescriptorPoolCreateInfo	poolInfo{};
        poolInfo.flags					= vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets				= MAX_FRAMES_IN_FLIGHT;
        poolInfo.poolSizeCount			= static_cast<uint32_t>(poolSize.size());
		poolInfo.pPoolSizes				= poolSize.data();

		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
	}

    void createDescriptorSets() 
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

		vk::DescriptorSetAllocateInfo	allocInfo;
		allocInfo.descriptorPool		= descriptorPool;
		allocInfo.descriptorSetCount	= static_cast<uint32_t>(layouts.size());
		allocInfo.pSetLayouts			= layouts.data();

        descriptorSets.clear();
        descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
		{
			vk::DescriptorBufferInfo bufferInfo;
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			vk::DescriptorImageInfo imageInfo;
			imageInfo.sampler		= textureSampler;
			imageInfo.imageView		= textureImageView;
			imageInfo.imageLayout	= vk::ImageLayout::eShaderReadOnlyOptimal;

            std::array descriptorWrites
			{
				vk::WriteDescriptorSet(descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo),
				vk::WriteDescriptorSet(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler,	&imageInfo)
            };

			device.updateDescriptorSets(descriptorWrites, {});
		}
		
    }

	void createBuffer(
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		vk::raii::Buffer& buffer,
		vk::raii::DeviceMemory& bufferMemory)
	{
		vk::BufferCreateInfo bufferInfo;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = vk::SharingMode::eExclusive;
		buffer = vk::raii::Buffer(device, bufferInfo);

		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType( physicalDevice, memRequirements.memoryTypeBits, properties);

        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(bufferMemory, 0);
	}

	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size)
	{
		vk::CommandBufferAllocateInfo allocInfo;
		allocInfo.commandPool = commandPool;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = 1;
        vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
		
		vk::CommandBufferBeginInfo commandBufferBeginInfo;
		commandBufferBeginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		commandCopyBuffer.begin(commandBufferBeginInfo.flags);
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
		commandCopyBuffer.end();

		vk::SubmitInfo submitInfo;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &*commandCopyBuffer;

		graphicsQueue.submit(submitInfo, nullptr);
		graphicsQueue.waitIdle();
    }

    void transitionImageLayout(const vk::raii::Image& image, 
		vk::ImageLayout oldLayout, vk::ImageLayout newLayout) 
	{
        auto commandBuffer = beginSingleTimeCommands();

		vk::MemoryBarrier		memoryBarrier{};
		vk::BufferMemoryBarrier bufferMemoryBarrier{};

		vk::ImageMemoryBarrier		barrier{};
		barrier.oldLayout			= oldLayout;
		barrier.newLayout			= newLayout;
		barrier.image				= image;
        barrier.subresourceRange	= { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

        vk::PipelineStageFlags		sourceStage;
        vk::PipelineStageFlags		destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) 
		{
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage			= vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage	= vk::PipelineStageFlagBits::eTransfer;
			barrier.sType		= vk::StructureType::eImageMemoryBarrier;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) 
		{
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage			= vk::PipelineStageFlagBits::eTransfer;
            destinationStage	= vk::PipelineStageFlagBits::eFragmentShader;
        }
        else 
		{
            throw std::invalid_argument("Unsupported layout transition!");
        }
        commandBuffer->pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);

        endSingleTimeCommands(*commandBuffer);
    }

    void copyBufferToImage(const vk::raii::Buffer& buffer, 
		vk::raii::Image& image, uint32_t width, uint32_t height) 
	{
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();

		vk::BufferImageCopy region;
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
		region.imageOffset.x = 0;
		region.imageOffset.y = 0;
		region.imageOffset.z = 0;
		region.imageExtent.width = width;
		region.imageExtent.height = height;
		region.imageExtent.depth = 1;

		commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);

        endSingleTimeCommands(*commandBuffer);
    }

	void createCommandBuffers()
	{
		commandBuffers.clear();
		vk::CommandBufferAllocateInfo allocInfo;
		allocInfo.commandPool = commandPool;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	void recordCommandBuffer(uint32_t imageIndex)
	{
		commandBuffers[currentFrame].begin({});
		// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
		transition_image_layout(
			imageIndex,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal,
			{},                                                        // srcAccessMask (no need to wait for previous operations)
			vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
			vk::PipelineStageFlagBits2::eTopOfPipe,                   //  srcStage
			vk::PipelineStageFlagBits2::eColorAttachmentOutput        //  dstStage
		);
		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);

		vk::RenderingAttachmentInfo attachmentInfo;
		attachmentInfo.imageView = swapChainImageViews[imageIndex];
		attachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
		attachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
		attachmentInfo.clearValue = clearColor;

		vk::RenderingInfo renderingInfo;
		renderingInfo.renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent };
		renderingInfo.layerCount = 1;
		renderingInfo.colorAttachmentCount = 1;
		renderingInfo.pColorAttachments = &attachmentInfo;

		commandBuffers[currentFrame].beginRendering(renderingInfo);
		commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
		commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
		commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
		commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, { 0 });
		commandBuffers[currentFrame].bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);
		commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
		commandBuffers[currentFrame].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
		commandBuffers[currentFrame].endRendering();

		// After rendering, transition the swapchain image to PRESENT_SRC
		transition_image_layout(
			imageIndex,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,                 // srcAccessMask
			{},                                                        //  dstAccessMask
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
			vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
		);
		commandBuffers[currentFrame].end();
	}

	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands()
	{
        vk::CommandBufferAllocateInfo allocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = 
			std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        commandBuffer->begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) 
	{
        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &*commandBuffer;

        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

	void createSyncObjects()
	{
		presentCompleteSemaphore.clear();
		renderFinishedSemaphore.clear();
		inFlightFences.clear();

		vk::FenceCreateInfo fenceCreateInfo;
		fenceCreateInfo.flags = vk::FenceCreateFlagBits::eSignaled;

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			presentCompleteSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
			renderFinishedSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			inFlightFences.emplace_back(device, fenceCreateInfo.flags);
		}
	}

	void drawFrame()
	{
		while (vk::Result::eTimeout == device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX));
		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphore[semaphoreIndex], nullptr);


		if (result == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapChain();
			return;
		}
		if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
		{
			throw std::runtime_error("Failed to acquire swap chain image!");
		}
		updateUniformBuffer(currentFrame);

		device.resetFences(*inFlightFences[currentFrame]);

		commandBuffers[currentFrame].reset();
		recordCommandBuffer(imageIndex);

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		vk::SubmitInfo submitInfo;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &*presentCompleteSemaphore[currentFrame];
		submitInfo.pWaitDstStageMask = &waitDestinationStageMask;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &*commandBuffers[currentFrame];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &*renderFinishedSemaphore[currentFrame];
		graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);


		vk::PresentInfoKHR presentInfoKHR;
		presentInfoKHR.waitSemaphoreCount = 1;
		presentInfoKHR.pWaitSemaphores = &*renderFinishedSemaphore[imageIndex];
		presentInfoKHR.swapchainCount = 1;
		presentInfoKHR.pSwapchains = &*swapChain;
		presentInfoKHR.pImageIndices = &imageIndex;

		result = presentQueue.presentKHR(presentInfoKHR);

		switch (result)
		{
		case vk::Result::eSuccess:
			break;
		case vk::Result::eSuboptimalKHR:
			std::cout << "vk::Queue::presentKHR returned vk::Result::eSuboptimalKHR !\n";
			break;
		default:
			break;  // an unexpected result is returned!
		}
		semaphoreIndex = (semaphoreIndex + 1) % presentCompleteSemaphore.size();
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void updateUniformBuffer(uint32_t currentImage)
	{
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo{};
		ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	void transition_image_layout(uint32_t imageIndex,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask
	) {
		vk::ImageMemoryBarrier2					barrier;
		barrier.srcStageMask								= src_stage_mask;
		barrier.srcAccessMask								= src_access_mask;
		barrier.dstStageMask								= dst_stage_mask;
		barrier.dstAccessMask								= dst_access_mask;
		barrier.oldLayout									= old_layout;
		barrier.newLayout									= new_layout;
		barrier.srcQueueFamilyIndex							= VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex							= VK_QUEUE_FAMILY_IGNORED;
		barrier.image										= swapChainImages[imageIndex];
		barrier.subresourceRange.aspectMask					= vk::ImageAspectFlagBits::eColor;
		barrier.subresourceRange.baseMipLevel				= 0;
		barrier.subresourceRange.levelCount					= 1;
		barrier.subresourceRange.baseArrayLayer				= 0;
		barrier.subresourceRange.layerCount					= 1;

		vk::DependencyInfo dependency_info;
		dependency_info.dependencyFlags						= {};
		dependency_info.imageMemoryBarrierCount				= 1;
		dependency_info.pImageMemoryBarriers				= &barrier;

		commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
	}

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const
    {
        vk::ShaderModuleCreateInfo createInfo;

        createInfo.codeSize = code.size() * sizeof(char);
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        vk::raii::ShaderModule shaderModule{ device, createInfo };

        return shaderModule;
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers || disableValidationLayer) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT;
        debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags;
        debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags;
        debugUtilsMessengerCreateInfoEXT.pfnUserCallback = &debugCallback;

        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }
};